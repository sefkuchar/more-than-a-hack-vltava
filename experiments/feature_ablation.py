"""Four fixed feature-group ablations under isolated full-context decoding."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from possession.model import fit_model, predict
from experiments.duration_model import duration_decode


if __name__ == '__main__':
    x, names, _ = build_features(broadcast=True)
    assert len(names) == 77
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    x, y = x[start:stop], gt[start:stop]
    with np.load(ROOT / 'experiments/cache/external_duration_validation.npz') as prior:
        density, survival = prior['log_density'], prior['log_survival']
    remove = {
        'no_broadcast': lambda name: name.startswith('broadcast'),
        'no_tags': lambda name: name in ('phase_H', 'phase_A', 'restart'),
        'no_ball_motion': lambda name: name.startswith(('ball_speed', 'stationary_', 'boundary')) or name == 'trajectory_turn',
        'no_short_smoothing': lambda name: any(name.endswith('_' + radius + suffix)
            for radius in ('5', '12') for suffix in ('', '_H', '_A')),
    }
    outputs, results = {}, []
    for kind, predicate in remove.items():
        keep = [j for j, name in enumerate(names) if not predicate(name)]
        subset_names = [names[j] for j in keep]
        features = x[:, keep]
        contexts = np.empty((5, len(y), 3))
        prediction = np.empty(len(y), np.int8)
        for fold in range(5):
            a, b = fold * 1500, (fold + 1) * 1500
            train = np.ones(len(y), bool)
            train[max(0, a - 250):min(len(y), b + 250)] = False
            ids = np.flatnonzero(train)[::3]
            model = fit_model(features[ids], y[ids], subset_names)
            p = predict(model, features, subset_names)
            contexts[fold] = p
            decoded = duration_decode(np.log(p.clip(1e-5, 1)) / 5, density, survival, np.arange(len(y) + 1))
            prediction[a:b] = decoded[a:b]
        score = pm.evaluate(y, prediction)
        row = dict(method=kind, feature_count=len(keep), removed=[name for name in names if predicate(name)],
                   **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
        results.append(row)
        outputs[kind] = prediction
        outputs[kind + '_fold_probability'] = contexts
        print(row, flush=True)
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
        ('experiments/feature_ablation.py', 'possession/features.py', 'possession/model.py',
         'experiments/duration_model.py', 'experiments/align_broadcast.py')}
    hashes.update(features=hashlib.sha256(x.tobytes()).hexdigest(), labels=hashlib.sha256(y.tobytes()).hexdigest(),
                  priors=hashlib.sha256(density.tobytes() + survival.tobytes()).hexdigest())
    report = dict(results=results, source_input_sha256=hashes, decoder='external duration exact25Hz',
                  temperature=5, folds=5, embargo=250, stride=3)
    (ROOT / 'experiments/feature_ablation_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/feature_ablation_validation.npz', **outputs)
