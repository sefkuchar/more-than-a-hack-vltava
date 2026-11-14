"""Ordered lag snapshots for release/reception evidence, with fixed offsets."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from games import half_video_blocks
from possession.features import build_features
from possession.model import fit_model, predict, swap_teams
from possession.duration import decode, PRIOR_PATH
from experiments.physical_boundary_snap import control_onsets, snap

SIGNALS = ('contact_H', 'contact_A', 'stable_contact_H', 'stable_contact_A',
           'distance_H', 'distance_A', 'ball_speed', 'observed',
           'broadcast_contact_H', 'broadcast_contact_A', 'broadcast_visible')
LAGS = (-50, -25, -10, 10, 25, 50)


def lagged_features(x, names):
    low = np.zeros(len(x), np.int32)
    high = np.full(len(x), len(x) - 1, np.int32)
    for _, a, b in half_video_blocks('BAR-ATH'):
        low[a:b + 1], high[a:b + 1] = a, b
    ids = np.arange(len(x))
    columns, new_names = [x], list(names)
    source = [names.index(name) for name in SIGNALS]
    for lag in LAGS:
        columns.append(x[np.clip(ids + lag, low, high)][:, source])
        prefix = ('past' if lag < 0 else 'future') + str(abs(lag)) + '_'
        new_names.extend(prefix + name for name in SIGNALS)
    return np.concatenate(columns, axis=1).astype(np.float32), new_names


if __name__ == '__main__':
    base, names, raw = build_features(broadcast=True)
    assert len(names) == 77
    stable = base[:, [names.index('stable_contact_H'), names.index('stable_contact_A')]]
    onsets = control_onsets(raw['nearest'], stable, raw['observed'])
    x, expanded_names = lagged_features(base, names)
    probe = np.arange(11185, 11285)
    mirrored, _ = lagged_features(swap_teams(base, names), names)
    assert np.array_equal(mirrored[probe], swap_teams(x[probe], expanded_names)), 'Lag pairs must swap exactly'
    del mirrored
    # No labels are used in constructing temporal evidence or control onsets.
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    x, y = x[start:stop], gt[start:stop]
    contexts = np.empty((5, len(y), 3))
    duration_pred = np.empty(len(y), np.int8)
    snap_pred = np.empty(len(y), np.int8)
    for fold in range(5):
        a, b = fold * 1500, (fold + 1) * 1500
        train = np.ones(len(y), bool)
        train[max(0, a - 250):min(len(y), b + 250)] = False
        ids = np.flatnonzero(train)[::3]
        model = fit_model(x[ids], y[ids], expanded_names)
        p = predict(model, x, expanded_names)
        contexts[fold] = p
        original = decode(p)
        snapped, _ = snap(original, onsets, start)
        duration_pred[a:b], snap_pred[a:b] = original[a:b], snapped[a:b]
        print('lagged evidence fold', fold, flush=True)
    results = []
    for kind, pred in [('duration', duration_pred), ('duration_snap', snap_pred)]:
        score = pm.evaluate(y, pred)
        row = dict(method=kind, **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
        results.append(row)
        print(row, flush=True)
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
        ('experiments/lagged_evidence.py', 'possession/features.py', 'possession/model.py',
         'possession/duration.py', 'experiments/physical_boundary_snap.py', 'experiments/align_broadcast.py')}
    hashes.update(features=hashlib.sha256(x.tobytes()).hexdigest(), labels=hashlib.sha256(y.tobytes()).hexdigest(),
                  prior=hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest())
    report = dict(results=results, source_input_sha256=hashes, feature_count=len(expanded_names),
        lag_frames=LAGS, signals=SIGNALS, folds=5, embargo=250, stride=3,
        offline_future_context=True, selection_optimism=True,
        snapped_heldout_frames=int(np.sum(duration_pred != snap_pred)))
    (ROOT / 'experiments/lagged_evidence_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/lagged_evidence_validation.npz',
        fold_probability=contexts, duration=duration_pred, duration_snap=snap_pred, names=expanded_names)
