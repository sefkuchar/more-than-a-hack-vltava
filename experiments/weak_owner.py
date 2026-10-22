"""Transductive ownership teacher fitted solely to automatic attacking tags."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from games import half_video_blocks, require
from starter.load_smart_tagging import load as load_tags
from possession.features import build_features
from possession.model import fit_model, predict, swap_teams
from possession.decode import viterbi
from experiments.duration_model import duration_decode


def fit_weak_teacher(x, names):
    codes = {'BUILD UP', 'PROGRESSION', 'ATTACKING TRANSITION', 'LONG BALL', 'CREATING CHANCES'}
    team_map = {'Barcelona': 0, 'Atletic Club': 1}
    active = np.zeros((len(x), 2), bool)
    for event in load_tags():
        team = team_map.get(event['tags'].get('Team'))
        if event['code'] in codes and team is not None:
            a, b = max(0, event['start_frame']), min(len(x), event['end_frame'])
            active[a:b, team] = True
    allowed = np.zeros(len(x), bool)
    for _, a, b in half_video_blocks('BAR-ATH'):
        allowed[a:b + 1] = True
    exclusive = allowed & (active.sum(axis=1) == 1)
    ids = np.flatnonzero(exclusive)[::25]
    weak_y = active[ids].argmax(axis=1)
    keep = [j for j, name in enumerate(names) if name not in ('phase_H', 'phase_A', 'restart')]
    teacher_names = [names[j] for j in keep]
    assert not any(name.startswith('phase') or name == 'restart' for name in teacher_names)
    assert set(np.unique(weak_y)) == {0, 1}
    physical = x[:, keep]
    train_x = np.concatenate([physical[ids], swap_teams(physical[ids], teacher_names)])
    train_y = np.concatenate([weak_y, 1 - weak_y])
    teacher = HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7,
        max_depth=3, learning_rate=.07, l2_regularization=50,
        min_samples_leaf=50, random_state=0)
    teacher.fit(train_x, train_y)
    assert np.array_equal(teacher.classes_, [0, 1])
    probability = (teacher.predict_proba(physical) +
                   teacher.predict_proba(swap_teams(physical, teacher_names))[:, ::-1]) / 2
    return probability, exclusive, active.argmax(axis=1), ids, weak_y, teacher_names, teacher.get_params()


if __name__ == '__main__':
    x, names, _ = build_features(broadcast=True)
    assert len(names) == 77
    weak_p, exclusive, weak_labels, ids, weak_y, teacher_names, teacher_params = fit_weak_teacher(x, names)
    augmented = np.column_stack([x, weak_p]).astype(np.float32)
    augmented_names = names + ['weak_owner_H', 'weak_owner_A']
    # Public labels are unavailable to the teacher function and read only after fitting.
    gt_path = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    start, stop = pm.valid_runs(gt)[0]
    features, y = augmented[start:stop], gt[start:stop]
    with np.load(ROOT / 'experiments/cache/external_duration_validation.npz') as prior:
        density, survival = prior['log_density'], prior['log_survival']
    contexts = np.empty((5, len(y), 3))
    predictions = {method: np.empty(len(y), np.int8) for method in ('viterbi32', 'external_duration')}
    for fold in range(5):
        a, b = fold * 1500, (fold + 1) * 1500
        train = np.ones(len(y), bool)
        train[max(0, a - 250):min(len(y), b + 250)] = False
        train_ids = np.flatnonzero(train)[::3]
        model = fit_model(features[train_ids], y[train_ids], augmented_names)
        p = predict(model, features, augmented_names)
        contexts[fold] = p
        evidence = np.log(p.clip(1e-5, 1))
        predictions['viterbi32'][a:b] = viterbi(evidence, 32)[a:b]
        predictions['external_duration'][a:b] = duration_decode(evidence / 5, density, survival, np.arange(len(y) + 1))[a:b]
        print('weak owner fold', fold, flush=True)
    results = []
    for method, pred in predictions.items():
        score = pm.evaluate(y, pred)
        row = dict(method=method, **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
        results.append(row)
        print(row, flush=True)
    mask = exclusive[start:stop]
    live = y < 2
    diagnostic = dict(public_exclusive_frames=int(mask.sum()), public_exclusive_live_frames=int((mask & live).sum()),
        weak_tag_agreement_including_dead=float(np.mean(weak_labels[start:stop][mask] == y[mask])),
        teacher_live_owner_accuracy=float(np.mean(weak_p[start:stop].argmax(axis=1)[live] == y[live])))
    paths = [Path(__file__), ROOT / 'possession/features.py', ROOT / 'possession/model.py',
             ROOT / 'possession/decode.py', ROOT / 'experiments/duration_model.py',
             ROOT / 'experiments/align_broadcast.py', Path(require(['smart_tagging'])[0])]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    hashes.update(features=hashlib.sha256(features.tobytes()).hexdigest(),
        weak_train_features=hashlib.sha256(x[ids][:, [names.index(name) for name in teacher_names]].tobytes()).hexdigest(),
        weak_train_labels=hashlib.sha256(weak_y.tobytes()).hexdigest(),
        public_labels=hashlib.sha256(y.tobytes()).hexdigest(),
        external_priors=hashlib.sha256(density.tobytes() + survival.tobytes()).hexdigest())
    report = dict(transductive_weak_training=True, teacher_label_source='automatic attacking SmartTagging only',
        weak_samples=len(ids), weak_sample_counts=np.bincount(weak_y, minlength=2).tolist(),
        teacher_features=teacher_names, teacher_parameters=teacher_params,
        diagnostics_after_fitting=diagnostic, source_input_sha256=hashes, results=results)
    (ROOT / 'experiments/weak_owner_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/weak_owner_validation.npz',
        fold_probability=contexts, weak_owner_probability=weak_p, names=augmented_names, **predictions)
