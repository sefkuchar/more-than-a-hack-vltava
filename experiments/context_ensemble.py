"""Fixed equal-weight Hist ensembles, decoding each fold's full context."""

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
    cache = ROOT / 'experiments/cache'
    files = {'base': ('booster_ensemble.npz', 'hist_fold_probability'),
             'latent': ('latent_context_validation.npz', 'fold_probability'),
             'ball': ('ball_possession_validation_target.npz', 'fold_probability')}
    contexts = {}
    hashes = {}
    for kind, (filename, key) in files.items():
        with np.load(cache / filename) as saved:
            p = saved[key]
        assert p.shape == (5, 7500, 3) and np.isfinite(p).all()
        assert np.allclose(p.sum(axis=2), 1)
        contexts[kind] = p
        hashes[filename] = hashlib.sha256((cache / filename).read_bytes()).hexdigest()
    with np.load(cache / 'external_duration_validation.npz') as prior:
        density, survival = prior['log_density'], prior['log_survival']
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    y = gt[start:stop]
    assert len(y) == 7500
    predictions = {}
    results = []
    def evaluate(name, probabilities):
        pred = np.empty(7500, np.int8)
        for fold in range(5):
            result = duration_decode(np.log(probabilities[fold].clip(1e-5, 1)) / 5,
                                     density, survival, np.arange(7501))
            a, b = fold * 1500, (fold + 1) * 1500
            pred[a:b] = result[a:b]
        predictions[name] = pred
        score = pm.evaluate(y, pred)
        row = dict(method=name, **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
        results.append(row)
        print(row, flush=True)
    for name, keys in [('base_latent', ('base', 'latent')), ('base_latent_ball', ('base', 'latent', 'ball'))]:
        probabilities = np.mean([contexts[key] for key in keys], axis=0)
        contexts[name] = probabilities
        evaluate(name, probabilities)
    if max(row['spell_f1_mean'] for row in results) < .65:
        x, names, _ = build_features(broadcast=True)
        x = x[start:stop]
        feature_hash = hashlib.sha256(x.tobytes()).hexdigest()
        old = json.loads((ROOT / 'experiments/booster_ensemble_results.json').read_text())
        assert feature_hash == old['feature_sha256'], 'Base input features changed since cached contexts'
        hashes['base_features'] = feature_hash
        bagged = np.empty((5, 7500, 3))
        for fold in range(5):
            a, b = fold * 1500, (fold + 1) * 1500
            train = np.ones(7500, bool)
            train[max(0, a - 250):min(7500, b + 250)] = False
            ids = np.flatnonzero(train)
            probabilities = []
            for offset in range(3):
                sample = ids[offset::3]
                model = fit_model(x[sample], y[sample], names)
                probabilities.append(predict(model, x, names))
            bagged[fold] = np.mean(probabilities, axis=0)
            print('interleaved bagging fold', fold, flush=True)
        contexts['interleaved_bagging'] = bagged
        evaluate('interleaved_bagging', bagged)
    hashes['script'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    hashes['labels'] = hashlib.sha256(y.tobytes()).hexdigest()
    hashes['duration_priors'] = hashlib.sha256(density.tobytes() + survival.tobytes()).hexdigest()
    report = dict(results=results, decoder='external duration exact 25Hz', temperature=5,
                  fixed_equal_weights=True, fold_context_shape=[5, 7500, 3], input_sha256=hashes)
    (ROOT / 'experiments/context_ensemble_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(cache / 'context_ensemble_validation.npz', **predictions,
        **{name + '_fold_probability': p for name, p in contexts.items() if name not in files})
