"""Fixed CatBoost / XGBoost comparison with fully isolated fold decoding."""

import hashlib
import json
import sys
from pathlib import Path

import catboost
import numpy as np
import xgboost
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from possession.decode import viterbi
from possession.model import predict, swap_teams
from experiments.compare import metrics


def learner(name):
    if name == 'catboost':
        return CatBoostClassifier(iterations=300, depth=4, learning_rate=.05,
            l2_leaf_reg=10, loss_function='MultiClass', random_seed=0,
            thread_count=4, verbose=False, allow_writing_files=False)
    return XGBClassifier(objective='multi:softprob', num_class=3,
        n_estimators=200, max_depth=3, learning_rate=.05, subsample=.8,
        colsample_bytree=.8, reg_lambda=10, min_child_weight=10,
        tree_method='hist', n_jobs=4, random_state=0)


if __name__ == '__main__':
    x, names, _ = build_features(broadcast=True)
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    x, y = x[start:stop], gt[start:stop]
    assert len(y) == 7500 and len(names) == 77
    source_paths = ['experiments/boosters.py', 'possession/features.py',
                    'possession/model.py', 'possession/decode.py', 'experiments/align_broadcast.py']
    fingerprints = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths}
    fingerprints['feature_matrix'] = hashlib.sha256(x.tobytes()).hexdigest()
    fingerprints['labels'] = hashlib.sha256(y.tobytes()).hexdigest()
    cache = ROOT / 'experiments/cache'
    cache.mkdir(exist_ok=True)
    results = []
    for kind in ('catboost', 'xgboost'):
        contexts = np.empty((5, len(y), 3))
        predictions = {cost: np.empty(len(y), np.int8) for cost in (16, 32)}
        for fold in range(5):
            a, b = fold * 1500, (fold + 1) * 1500
            train = np.ones(len(y), bool)
            train[max(0, a - 250):min(len(y), b + 250)] = False
            ids = np.flatnonzero(train)[::3]
            train_x = np.concatenate([x[ids], swap_teams(x[ids], names)])
            train_y = np.concatenate([y[ids], np.where(y[ids] < 2, 1 - y[ids], y[ids])])
            model = learner(kind)
            model.fit(train_x, train_y)
            probability = predict(model, x, names)
            assert np.isfinite(probability).all() and np.allclose(probability.sum(axis=1), 1)
            contexts[fold] = probability
            for cost in predictions:
                predictions[cost][a:b] = viterbi(np.log(probability.clip(1e-5, 1)), cost)[a:b]
            print(kind, 'fold', fold, flush=True)
        for cost, pred in predictions.items():
            row = dict(method=kind, penalty=cost, **metrics(y, pred))
            results.append(row)
            print(row, flush=True)
        np.savez_compressed(cache / f'{kind}_validation.npz', fold_probability=contexts,
                            names=names, **{f'penalty_{cost}': pred for cost, pred in predictions.items()})
    report = dict(results=results, versions=dict(catboost=catboost.__version__, xgboost=xgboost.__version__),
                  source_fingerprints=fingerprints, stride=3, embargo_frames=250,
                  folds=5, feature_count=len(names), parameters={kind: learner(kind).get_params()
                      for kind in ('catboost', 'xgboost')})
    (ROOT / 'experiments/boosters_results.json').write_text(json.dumps(report, indent=2) + '\n')
