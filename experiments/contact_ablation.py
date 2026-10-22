"""Fixed current-contact feature subsets, excluding learned history pairs."""

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
    feature_path = ROOT / 'experiments/cache/ball_target_features_target_contact.npz'
    with np.load(feature_path) as saved:
        extra, extra_names = saved['features'], saved['names'].tolist()
    assert x.shape[1] == 77 and extra.shape == (len(x), 14)
    current_names = ['neural_contact_H', 'neural_contact_A', 'neural_contact_entropy']
    subsets = {'current_contact': current_names,
               'ball_current_contact': [name for name in extra_names if name.startswith('reconstructed_')] + current_names}
    # All feature construction precedes public label access.
    variants = {kind: (np.column_stack([x, extra[:, [extra_names.index(name) for name in selected]]]).astype(np.float32),
                       names + selected) for kind, selected in subsets.items()}
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    y = gt[start:stop]
    with np.load(ROOT / 'experiments/cache/external_duration_validation.npz') as prior:
        density, survival = prior['log_density'], prior['log_survival']
    results, outputs = [], {}
    for kind, (features, feature_names) in variants.items():
        features = features[start:stop]
        contexts = np.empty((5, len(y), 3))
        prediction = np.empty(len(y), np.int8)
        for fold in range(5):
            a, b = fold * 1500, (fold + 1) * 1500
            train = np.ones(len(y), bool)
            train[max(0, a - 250):min(len(y), b + 250)] = False
            ids = np.flatnonzero(train)[::3]
            model = fit_model(features[ids], y[ids], feature_names)
            p = predict(model, features, feature_names)
            contexts[fold] = p
            decoded = duration_decode(np.log(p.clip(1e-5, 1)) / 5, density, survival, np.arange(len(y) + 1))
            prediction[a:b] = decoded[a:b]
        score = pm.evaluate(y, prediction)
        row = dict(method=kind, feature_count=len(feature_names), added=subsets[kind],
            feature_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
            **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
        results.append(row)
        outputs[kind] = prediction
        outputs[kind + '_fold_probability'] = contexts
        print(row, flush=True)
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
        ('experiments/contact_ablation.py', 'possession/features.py', 'possession/model.py',
         'experiments/duration_model.py', 'experiments/align_broadcast.py')}
    hashes.update(extra_cache=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        labels=hashlib.sha256(y.tobytes()).hexdigest(), priors=hashlib.sha256(density.tobytes() + survival.tobytes()).hexdigest())
    report = dict(results=results, input_source_sha256=hashes,
        decoder='external duration exact25Hz', temperature=5, folds=5, embargo=250, stride=3)
    (ROOT / 'experiments/contact_ablation_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/contact_ablation_validation.npz', **outputs)
