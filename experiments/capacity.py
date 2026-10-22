"""Two predefined Hist capacities with the same features and duration decoder."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from possession.model import predict, swap_teams
from experiments.duration_model import duration_decode


if __name__ == '__main__':
    x, names, _ = build_features(broadcast=True)
    assert len(names) == 77
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    x, y = x[start:stop], gt[start:stop]
    with np.load(ROOT / 'experiments/cache/external_duration_validation.npz') as prior:
        density, survival = prior['log_density'], prior['log_survival']
    results, outputs = [], {}
    for name, depth, leaves, minimum in [('smaller', 2, 4, 70), ('larger', 4, 15, 15)]:
        contexts = np.empty((5, len(y), 3))
        prediction = np.empty(len(y), np.int8)
        parameters = dict(max_iter=120, learning_rate=.07, l2_regularization=10,
                          max_depth=depth, max_leaf_nodes=leaves, min_samples_leaf=minimum, random_state=0)
        for fold in range(5):
            a, b = fold * 1500, (fold + 1) * 1500
            train = np.ones(len(y), bool)
            train[max(0, a - 250):min(len(y), b + 250)] = False
            ids = np.flatnonzero(train)[::3]
            train_x = np.concatenate([x[ids], swap_teams(x[ids], names)])
            train_y = np.concatenate([y[ids], np.where(y[ids] < 2, 1 - y[ids], y[ids])])
            model = HistGradientBoostingClassifier(**parameters).fit(train_x, train_y)
            p = predict(model, x, names)
            contexts[fold] = p
            decoded = duration_decode(np.log(p.clip(1e-5, 1)) / 5, density, survival, np.arange(len(y) + 1))
            prediction[a:b] = decoded[a:b]
        score = pm.evaluate(y, prediction)
        row = dict(method=name, parameters=parameters,
            **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
        results.append(row)
        outputs[name] = prediction
        outputs[name + '_fold_probability'] = contexts
        print(row, flush=True)
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
        ('experiments/capacity.py', 'possession/features.py', 'possession/model.py',
         'experiments/duration_model.py', 'experiments/align_broadcast.py')}
    hashes.update(features=hashlib.sha256(x.tobytes()).hexdigest(), labels=hashlib.sha256(y.tobytes()).hexdigest(),
                  priors=hashlib.sha256(density.tobytes() + survival.tobytes()).hexdigest())
    report = dict(results=results, source_input_sha256=hashes, feature_count=77,
        decoder='external duration exact25Hz', temperature=5, folds=5, embargo=250, stride=3)
    (ROOT / 'experiments/capacity_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/capacity_validation.npz', **outputs)
