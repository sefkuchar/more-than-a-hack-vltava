"""Fixed boosting iteration neighbors; warm-start separately within each fold."""

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

ITERATIONS = (60, 90, 120, 150, 180)


if __name__ == '__main__':
    x, names, _ = build_features(broadcast=True)
    assert len(names) == 77
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    x, y = x[start:stop], gt[start:stop]
    with np.load(ROOT / 'experiments/cache/external_duration_validation.npz') as prior:
        density, survival = prior['log_density'], prior['log_survival']
    contexts = {iteration: np.empty((5, len(y), 3)) for iteration in ITERATIONS}
    predictions = {iteration: np.empty(len(y), np.int8) for iteration in ITERATIONS}
    parameters = dict(learning_rate=.07, l2_regularization=10, max_depth=3,
                      max_leaf_nodes=7, min_samples_leaf=35, random_state=0, warm_start=True)
    for fold in range(5):
        a, b = fold * 1500, (fold + 1) * 1500
        train = np.ones(len(y), bool)
        train[max(0, a - 250):min(len(y), b + 250)] = False
        ids = np.flatnonzero(train)[::3]
        train_x = np.concatenate([x[ids], swap_teams(x[ids], names)])
        train_y = np.concatenate([y[ids], np.where(y[ids] < 2, 1 - y[ids], y[ids])])
        model = HistGradientBoostingClassifier(max_iter=60, **parameters)
        for iteration in ITERATIONS:
            model.set_params(max_iter=iteration)
            model.fit(train_x, train_y)
            p = predict(model, x, names)
            contexts[iteration][fold] = p
            decoded = duration_decode(np.log(p.clip(1e-5, 1)) / 5, density, survival, np.arange(len(y) + 1))
            predictions[iteration][a:b] = decoded[a:b]
        print('iteration path fold', fold, flush=True)
    results = []
    for iteration in ITERATIONS:
        score = pm.evaluate(y, predictions[iteration])
        row = dict(iterations=iteration,
            **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
        results.append(row)
        print(row, flush=True)
    with np.load(ROOT / 'experiments/cache/booster_ensemble.npz') as baseline:
        difference = float(np.max(np.abs(contexts[120] - baseline['hist_fold_probability'])))
    assert difference < 1e-10, 'Warm-start 120 differs from cached baseline'
    hashes = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
        ('experiments/iteration_path.py', 'possession/features.py', 'possession/model.py',
         'experiments/duration_model.py', 'experiments/align_broadcast.py')}
    hashes.update(features=hashlib.sha256(x.tobytes()).hexdigest(), labels=hashlib.sha256(y.tobytes()).hexdigest(),
                  priors=hashlib.sha256(density.tobytes() + survival.tobytes()).hexdigest())
    report = dict(results=results, parameters=parameters, baseline120_max_probability_difference=difference,
        source_input_sha256=hashes, decoder='external duration exact25Hz', temperature=5,
        folds=5, embargo=250, stride=3, selection_optimism=True)
    (ROOT / 'experiments/iteration_path_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/iteration_path_validation.npz',
        **{f'iterations_{iteration}': predictions[iteration] for iteration in ITERATIONS},
        **{f'iterations_{iteration}_fold_probability': contexts[iteration] for iteration in ITERATIONS})
