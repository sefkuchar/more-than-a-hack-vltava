"""External player-only live/dead transfer with fixed long temporal context."""

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from games import half_video_blocks
from possession.features import build_features, tracking
from possession.model import fit_model, predict
from experiments.duration_model import duration_decode

SHIFTS = (0, -10, 10, -20, 20, -50, 50, -100, 100, -500, 500, -1000, 1000, -2000, 2000)


def player_features(players, teams, blocks):
    summaries = []
    centroids = []
    for team in (0, 1):
        values = players[:, teams == team]
        valid = np.isfinite(values).all(axis=2) & (values[:, :, 2] >= 0)
        speed = np.where(valid, values[:, :, 2], np.nan)
        xy = np.where(valid[:, :, None], values[:, :, :2], np.nan)
        count = np.maximum(valid.sum(axis=1), 1)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            quantiles = np.nanquantile(speed, (.25, .5, .75, .9), axis=1).T
            centroid = np.nanmean(xy, axis=1)
            spread = np.nanstd(xy, axis=1)
            mean = np.nanmean(speed, axis=1)
        moving = np.stack([(speed > threshold).sum(axis=1) / count for threshold in (1, 2, 4)], axis=1)
        summaries.append(np.column_stack([quantiles, moving, mean, centroid, spread]))
        centroids.append(centroid)
    first, second = summaries
    base = np.column_stack([np.minimum(first, second), np.maximum(first, second),
                            np.linalg.norm(centroids[0] - centroids[1], axis=1)])
    velocity = np.zeros((len(players), 2))
    for a, b in blocks:
        ids = np.arange(a, b)
        for team in (0, 1):
            delta = centroids[team][np.clip(ids + 3, a, b - 1)] - centroids[team][np.clip(ids - 3, a, b - 1)]
            velocity[ids, team] = np.linalg.norm(delta, axis=1) * 25 / 6
    base = np.column_stack([base, velocity.min(axis=1), velocity.max(axis=1)])
    return np.nan_to_num(base, nan=0, posinf=0, neginf=0).astype(np.float32)


def context(base, blocks, selected):
    lo = np.zeros(len(base), np.int32)
    hi = np.full(len(base), len(base) - 1, np.int32)
    for a, b in blocks:
        lo[a:b], hi[a:b] = a, b - 1
    return np.concatenate([base[np.clip(selected + shift, lo[selected], hi[selected])]
                           for shift in SHIFTS], axis=1)


if __name__ == '__main__':
    external_path = ROOT / 'data/external/idsse/J03WOH_arrays.npz'
    with np.load(external_path) as data:
        ep, et, eh, status = data['players'], data['teams'], data['half'], data['status']
    eb = [(int(np.flatnonzero(eh == half)[0]), int(np.flatnonzero(eh == half)[-1]) + 1)
          for half in np.unique(eh)]
    external = player_features(ep, et, eb)
    ids = np.arange(0, len(external), 25)
    ex = context(external, eb, ids)
    ey = (status[ids] == 0).astype(np.int8)
    teacher = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7, max_depth=3,
        learning_rate=.07, l2_regularization=50, min_samples_leaf=50, random_state=0)
    teacher.fit(ex, ey)
    players, _, teams = tracking()
    blocks = [(a, b + 1) for _, a, b in half_video_blocks('BAR-ATH')]
    target = player_features(players, teams, blocks)
    dead_probability = np.zeros(len(target))
    for a, b in blocks:
        for lo in range(a, b, 10000):
            sample = np.arange(lo, min(lo + 10000, b))
            dead_probability[sample] = teacher.predict_proba(context(target, blocks, sample))[:, 1]
    x, names, _ = build_features(broadcast=True)
    assert len(names) == 77
    x = np.column_stack([x, dead_probability]).astype(np.float32)
    names += ['player_only_dead_probability']
    # Public labels enter only after the external teacher and all target features exist.
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    x, y = x[start:stop], gt[start:stop]
    with np.load(ROOT / 'experiments/cache/external_duration_validation.npz') as prior:
        density, survival = prior['log_density'], prior['log_survival']
    contexts = np.empty((5, len(y), 3))
    prediction = np.empty(len(y), np.int8)
    for fold in range(5):
        a, b = fold * 1500, (fold + 1) * 1500
        train = np.ones(len(y), bool)
        train[max(0, a - 250):min(len(y), b + 250)] = False
        ids = np.flatnonzero(train)[::3]
        model = fit_model(x[ids], y[ids], names)
        p = predict(model, x, names)
        contexts[fold] = p
        result = duration_decode(np.log(p.clip(1e-5, 1)) / 5, density, survival, np.arange(len(y) + 1))
        prediction[a:b] = result[a:b]
        print('player dead teacher fold', fold, flush=True)
    score = pm.evaluate(y, prediction)
    diagnostic = dict(dead_f1=float(f1_score(y == 2, dead_probability[start:stop] >= .5)),
        dead_roc_auc=float(roc_auc_score(y == 2, dead_probability[start:stop])))
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
        ('experiments/player_dead_teacher.py', 'possession/features.py', 'possession/model.py',
         'experiments/duration_model.py', 'experiments/align_broadcast.py')}
    hashes.update(external_arrays=hashlib.sha256(external_path.read_bytes()).hexdigest(),
        teacher_train_features=hashlib.sha256(ex.tobytes()).hexdigest(),
        teacher_train_labels=hashlib.sha256(ey.tobytes()).hexdigest(),
        target_features=hashlib.sha256(x.tobytes()).hexdigest())
    report = dict(method='external_player_only_dead_teacher', teacher_parameters=teacher.get_params(),
        teacher_inputs='player positions and speed only; team-symmetric summaries; no ball, tags, labels, clock',
        shifted_context_frames=SHIFTS, external_train_frames=len(ey),
        external_class_counts=np.bincount(ey).tolist(), diagnostic_after_teacher_fit=diagnostic,
        source_input_sha256=hashes, **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
    print(report, flush=True)
    (ROOT / 'experiments/player_dead_teacher_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/player_dead_teacher_validation.npz',
        teacher_dead_probability=dead_probability, fold_probability=contexts, prediction=prediction)
