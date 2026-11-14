"""Unchanged physical snap on four previously tied feature-model contexts."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from possession.duration import decode, PRIOR_PATH
from experiments.physical_boundary_snap import control_onsets, snap


if __name__ == '__main__':
    x, names, raw = build_features(broadcast=True)
    stable = x[:, [names.index('stable_contact_H'), names.index('stable_contact_A')]]
    onsets = control_onsets(raw['nearest'], stable, raw['observed'])
    start, stop = 11185, 18685
    models = {
        'latent': ('latent_context_validation.npz', 'fold_probability'),
        'ball_target': ('ball_possession_validation_target.npz', 'fold_probability'),
        'base_latent': ('context_ensemble_validation.npz', 'base_latent_fold_probability'),
        'base_latent_ball': ('context_ensemble_validation.npz', 'base_latent_ball_fold_probability'),
    }
    arrays, hashes = {}, {}
    for method, (filename, key) in models.items():
        path = ROOT / 'experiments/cache' / filename
        hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as saved:
            p = saved[key]
        assert p.shape == (5, 7500, 3) and np.isfinite(p).all()
        original, prediction = np.empty(7500, np.int8), np.empty(7500, np.int8)
        originals, snapped = [], []
        for fold in range(5):
            base = decode(p[fold])
            fixed, _ = snap(base, onsets, start)
            a, b = fold * 1500, (fold + 1) * 1500
            original[a:b], prediction[a:b] = base[a:b], fixed[a:b]
            originals.append(base)
            snapped.append(fixed)
        arrays[method] = prediction
        arrays[method + '_original'] = original
        arrays[method + '_fold_prediction'] = np.stack(snapped)
        arrays[method + '_fold_original'] = np.stack(originals)
    truth = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[start:stop]
    results = []
    for method in models:
        score = pm.evaluate(truth, arrays[method])
        row = dict(method=method,
            changed_heldout_frames=int(np.sum(arrays[method] != arrays[method + '_original'])),
            **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
        results.append(row)
        print(row, flush=True)
    hashes.update({path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
        ('experiments/context_boundary_snap.py', 'experiments/physical_boundary_snap.py',
         'possession/duration.py', 'possession/features.py', 'experiments/align_broadcast.py')})
    hashes.update(features=hashlib.sha256(x[start:stop].tobytes()).hexdigest(),
        labels=hashlib.sha256(truth.tobytes()).hexdigest(), prior=hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest())
    report = dict(results=results, source_input_sha256=hashes, selection_optimism=True,
        protocol='Same fixed physical onsets/snap for each complete label-excluded fold context before retaining heldout core. No refitting or parameter search.')
    (ROOT / 'experiments/context_boundary_snap_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/context_boundary_snap_validation.npz', **arrays)
