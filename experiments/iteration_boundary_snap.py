"""Apply the unchanged physical boundary snap to cached iteration contexts."""

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
    provenance = json.loads((ROOT / 'experiments/iteration_path_results.json').read_text())
    assert hashlib.sha256(x[start:stop].tobytes()).hexdigest() == provenance['source_input_sha256']['features']
    path = ROOT / 'experiments/cache/iteration_path_validation.npz'
    results, arrays = [], {}
    with np.load(path) as cached:
        for iteration in (60, 90, 120, 150, 180):
            p = cached[f'iterations_{iteration}_fold_probability']
            assert p.shape == (5, 7500, 3)
            original, output = np.empty(7500, np.int8), np.empty(7500, np.int8)
            snapped_context, original_context = [], []
            for fold in range(5):
                base = decode(p[fold])
                fixed, _ = snap(base, onsets, start)
                a, b = fold * 1500, (fold + 1) * 1500
                original[a:b], output[a:b] = base[a:b], fixed[a:b]
                snapped_context.append(fixed)
                original_context.append(base)
            arrays[f'iterations_{iteration}'] = output
            arrays[f'iterations_{iteration}_original'] = original
            arrays[f'iterations_{iteration}_fold_prediction'] = np.stack(snapped_context)
            arrays[f'iterations_{iteration}_fold_original'] = np.stack(original_context)
    truth = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[start:stop]
    for iteration in (60, 90, 120, 150, 180):
        original, output = arrays[f'iterations_{iteration}_original'], arrays[f'iterations_{iteration}']
        score = pm.evaluate(truth, output)
        row = dict(iterations=iteration, changed_heldout_frames=int(np.sum(original != output)),
                   **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
        results.append(row)
        print(row, flush=True)
    assert np.isclose(results[2]['spell_f1_mean'], .64)
    hashes = {source: hashlib.sha256((ROOT / source).read_bytes()).hexdigest() for source in
        ('experiments/iteration_boundary_snap.py', 'experiments/physical_boundary_snap.py', 'possession/duration.py')}
    hashes.update(cache=hashlib.sha256(path.read_bytes()).hexdigest(),
        priors=hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),
        features=hashlib.sha256(x[start:stop].tobytes()).hexdigest(), labels=hashlib.sha256(truth.tobytes()).hexdigest())
    report = dict(results=results, source_input_sha256=hashes, selection_optimism=True,
        protocol='Unchanged physical onsets/snap applied to full same-fold duration decoding before heldout core extraction; no new models or physical parameters.')
    (ROOT / 'experiments/iteration_boundary_snap_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/iteration_boundary_snap_validation.npz', **arrays)
