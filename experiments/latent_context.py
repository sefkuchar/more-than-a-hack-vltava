"""Transductive unlabeled motion regimes; fixed K=8, no label pretraining."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from games import half_video_blocks
from possession.features import build_features
from possession.model import fit_model, predict, swap_teams
from possession.decode import viterbi
from experiments.duration_model import duration_decode


if __name__ == '__main__':
    x, names, _ = build_features(broadcast=True)
    assert x.shape[1] == 77
    ids = np.concatenate([np.arange(a, b + 1, 25) for _, a, b in half_video_blocks('BAR-ATH')])
    unlabeled = np.concatenate([x[ids], swap_teams(x[ids], names)])
    scaler = StandardScaler().fit(unlabeled)
    gmm = GaussianMixture(n_components=8, covariance_type='diag', reg_covar=.1,
                          random_state=0, max_iter=200).fit(scaler.transform(unlabeled))
    def latent(values):
        p = gmm.predict_proba(scaler.transform(values))
        q = gmm.predict_proba(scaler.transform(swap_teams(values, names)))
        return np.stack([p, q], axis=2).reshape(len(values), 16)
    augmented_names = names + [f'component{i}_{side}' for i in range(8) for side in ('H', 'A')]
    z = latent(x)
    augmented = np.column_stack([x, z]).astype(np.float32)
    sample = ids[:100]
    expected = np.column_stack([swap_teams(x[sample], names), latent(swap_teams(x[sample], names))]).astype(np.float32)
    assert np.array_equal(swap_teams(augmented[sample], augmented_names), expected), 'Latent pairs must be exactly equivariant'
    # Label access begins only after the whole-match unlabeled representation exists.
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    features, y = augmented[start:stop], gt[start:stop]
    assert len(y) == 7500
    with np.load(ROOT / 'experiments/cache/external_duration_validation.npz') as prior:
        density, survival = prior['log_density'], prior['log_survival']
    contexts = np.empty((5, 7500, 3))
    predictions = {method: np.empty(7500, np.int8) for method in ('viterbi32', 'external_duration')}
    for fold in range(5):
        a, b = fold * 1500, (fold + 1) * 1500
        train = np.ones(7500, bool)
        train[max(0, a - 250):min(7500, b + 250)] = False
        train_ids = np.flatnonzero(train)[::3]
        model = fit_model(features[train_ids], y[train_ids], augmented_names)
        p = predict(model, features, augmented_names)
        contexts[fold] = p
        evidence = np.log(p.clip(1e-5, 1))
        predictions['viterbi32'][a:b] = viterbi(evidence, 32)[a:b]
        predictions['external_duration'][a:b] = duration_decode(evidence / 5, density, survival, np.arange(7501))[a:b]
        print('latent isolated fold', fold, flush=True)
    results = []
    for method, pred in predictions.items():
        score = pm.evaluate(y, pred)
        row = dict(method=method, **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
        results.append(row)
        print(row, flush=True)
    fingerprints = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in
        ('experiments/latent_context.py', 'possession/features.py', 'possession/model.py',
         'possession/decode.py', 'experiments/duration_model.py', 'experiments/align_broadcast.py')}
    fingerprints['unlabeled_features'] = hashlib.sha256(unlabeled.tobytes()).hexdigest()
    fingerprints['supervised_features'] = hashlib.sha256(features.tobytes()).hexdigest()
    fingerprints['labels'] = hashlib.sha256(y.tobytes()).hexdigest()
    fingerprints['external_duration_priors'] = hashlib.sha256(density.tobytes() + survival.tobytes()).hexdigest()
    learned = dict(means=gmm.means_.tolist(), variances=gmm.covariances_.tolist(),
                   weights=gmm.weights_.tolist(), scaler_mean=scaler.mean_.tolist(),
                   scaler_scale=scaler.scale_.tolist())
    report = dict(transductive=True, unlabeled_rows=len(unlabeled), feature_count=93,
        gmm_parameters=gmm.get_params(), learned=learned, converged=bool(gmm.converged_),
        iterations=int(gmm.n_iter_), source_input_sha256=fingerprints, results=results)
    parameter_bytes = json.dumps(learned, sort_keys=True).encode()
    report['learned_parameter_sha256'] = hashlib.sha256(parameter_bytes).hexdigest()
    (ROOT / 'experiments/latent_context_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/latent_context_validation.npz',
                        fold_probability=contexts, names=augmented_names, **predictions)
