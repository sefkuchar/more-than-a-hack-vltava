"""One RBF SVC experiment under contiguous, embargoed public validation."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from possession.decode import viterbi
from possession.model import predict, swap_teams
from experiments.compare import metrics


if __name__ == '__main__':
    x, names, _ = build_features(broadcast=True)
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    x, y = x[start:stop], gt[start:stop]
    assert len(y) == 7500
    decoded = {cost: np.empty(len(y), np.int8) for cost in (16, 32)}
    oof = np.empty((len(y), 3))
    for fold in range(5):
        a, b = fold * 1500, (fold + 1) * 1500
        train = np.ones(len(y), bool)
        train[max(0, a - 250):min(len(y), b + 250)] = False
        ids = np.flatnonzero(train)[::3]
        train_x = np.concatenate([x[ids], swap_teams(x[ids], names)])
        train_y = np.concatenate([y[ids], np.where(y[ids] < 2, 1 - y[ids], y[ids])])
        model = make_pipeline(StandardScaler(), SVC(C=3, gamma='scale', probability=True, random_state=0))
        model.fit(train_x, train_y)
        probability = predict(model, x, names)
        oof[a:b] = probability[a:b]
        for cost in decoded:
            decoded[cost][a:b] = viterbi(np.log(probability.clip(1e-5, 1)), cost)[a:b]
        print('svm isolated fold', fold, 'features', len(names), flush=True)
    results = []
    for cost, pred in decoded.items():
        row = dict(method='rbf_svc', penalty=cost, **metrics(y, pred))
        results.append(row)
        print(row, flush=True)
    cache = ROOT / 'experiments/cache'
    cache.mkdir(exist_ok=True)
    np.savez_compressed(cache / 'svm_validation.npz', probability=oof,
                        **{f'penalty_{cost}': pred for cost, pred in decoded.items()})
    (ROOT / 'experiments/svm_results.json').write_text(json.dumps(results, indent=2) + '\n')
