"""Transfer external football spell durations, independent of target labels."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp
from sklearn.mixture import GaussianMixture

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from experiments.duration_model import duration_decode


def external_priors(labels, halves, length):
    samples = [[], []]
    for half in np.unique(halves):
        seq = labels[halves == half]
        boundaries = np.r_[0, np.flatnonzero(seq[1:] != seq[:-1]) + 1, len(seq)]
        # Half edges censor the first and last spells.
        for a, b in zip(boundaries[1:-2], boundaries[2:-1]):
            if seq[a] in ('H', 'A', 'D'):
                samples[int(seq[a] == 'D')].append(int(b - a))
    d = np.maximum(np.arange(length + 1, dtype=float), 1)
    density, survival, parameters = [], [], []
    for group, durations in zip(('possession', 'dead'), samples):
        values = np.log(durations).reshape(-1, 1)
        model = GaussianMixture(n_components=2, reg_covar=.25, random_state=0).fit(values)
        mu = model.means_[:, 0]
        sigma = np.sqrt(model.covariances_[:, 0, 0])
        weights = model.weights_
        z = (np.log(d)[None, :] - mu[:, None]) / sigma[:, None]
        log_weight = np.log(weights)[:, None]
        density.append(logsumexp(log_weight - np.log(d)[None, :]
                       - np.log(sigma[:, None] * np.sqrt(2 * np.pi)) - .5 * z * z, axis=0))
        survival.append(logsumexp(log_weight + log_ndtr(-z), axis=0))
        parameters.append(dict(group=group, count=len(durations),
            mean_duration_seconds=float(np.mean(durations) / 25),
            median_duration_seconds=float(np.median(durations) / 25),
            weights=weights.tolist(), log_frame_means=mu.tolist(), log_frame_sigmas=sigma.tolist()))
    return np.array([density[0], density[0], density[1]]), np.array([survival[0], survival[0], survival[1]]), parameters


if __name__ == '__main__':
    external_path = ROOT / 'data/external/idsse/J03WOH_arrays.npz'
    probability_path = ROOT / 'experiments/cache/booster_ensemble.npz'
    with np.load(external_path) as external:
        labels, halves = external['labels'], external['half']
    with np.load(probability_path) as cached:
        probability = cached['hist_fold_probability']
    assert probability.shape == (5, 7500, 3)
    density, survival, parameters = external_priors(labels, halves, 7500)
    assert np.isfinite(density).all() and np.isfinite(survival).all()
    print('external priors', parameters, flush=True)
    prediction = np.empty(7500, np.int8)
    for fold in range(5):
        result = duration_decode(np.log(probability[fold].clip(1e-5, 1)) / 5,
                                 density, survival, np.arange(7501))
        a, b = fold * 1500, (fold + 1) * 1500
        prediction[a:b] = result[a:b]
        print('external duration fold', fold, flush=True)
    # Target labels are read only after the fixed external-prior predictions exist.
    gt_path = ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'
    gt = pm.label_codes(pm.read_label_csv(gt_path))
    start, stop = pm.valid_runs(gt)[0]
    score = pm.evaluate(gt[start:stop], prediction)
    fingerprints = {}
    for path in (Path(__file__), ROOT / 'experiments/duration_model.py', external_path, probability_path, gt_path):
        fingerprints[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(method='external_duration_gmm2', emission_temperature=5,
                  parameters=parameters, input_sha256=fingerprints,
                  **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')})
    print(report, flush=True)
    np.savez_compressed(ROOT / 'experiments/cache/external_duration_validation.npz',
                        prediction=prediction, log_density=density, log_survival=survival)
    (ROOT / 'experiments/external_duration_results.json').write_text(json.dumps(report, indent=2) + '\n')
