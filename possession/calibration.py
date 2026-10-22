"""Team-symmetric calibration fitted only on blocked held-out training rows."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from possession.model import fit_model, predict


def calibrated(probability, parameters):
    p = np.asarray(probability, dtype=float)
    parameters = np.asarray(parameters, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Expected finite H/A/D probabilities')
    if parameters.shape != (2,) or not np.isfinite(parameters).all():
        raise ValueError('Expected two finite calibration parameters')
    log_scale, dead_intercept = parameters
    logits = np.exp(log_scale) * np.log(p.clip(1e-5, 1))
    logits[:, 2] += dead_intercept
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def fit_calibration(probability, labels):
    p = np.asarray(probability, dtype=float)
    y = np.asarray(labels)
    if p.shape != (len(y), 3) or not len(y) or not np.isin(y, (0, 1, 2)).all():
        raise ValueError('Expected calibration examples with H/A/D labels')
    if not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Nonfinite calibration probabilities')
    logp = np.log(p.clip(1e-5, 1))
    rows = np.arange(len(y))

    def objective(parameters):
        log_scale, intercept = parameters
        scale = np.exp(log_scale)
        logits = scale * logp
        logits[:, 2] += intercept
        logz = logsumexp(logits, axis=1)
        loss = np.mean(logz - logits[rows, y]) + .01 * (log_scale ** 2 + intercept ** 2)
        residual = np.exp(logits - logz[:, None])
        residual[rows, y] -= 1
        gradient = np.array([
            np.mean(np.sum(residual * (scale * logp), axis=1)) + .02 * log_scale,
            np.mean(residual[:, 2]) + .02 * intercept])
        return loss, gradient

    result = minimize(objective, np.zeros(2), method='L-BFGS-B', jac=True,
                      bounds=[(np.log(.25), np.log(4)), (-2, 2)])
    if not result.success:
        raise ValueError('Calibration failed: ' + result.message)
    return result.x, float(result.fun)


def blocked_calibration(x, y, names, allowed=None):
    """Inner minute folds exclude the outer holdout and both ten-second embargoes."""
    allowed = np.ones(len(y), bool) if allowed is None else np.asarray(allowed, dtype=bool)
    if allowed.shape != (len(y),) or len(x) != len(y):
        raise ValueError('Calibration mask and training arrays have different lengths')
    probabilities, targets, folds = [], [], []
    for inner, ids in enumerate(np.array_split(np.arange(len(y)), 5)):
        a, b = int(ids[0]), int(ids[-1]) + 1
        valid = ids[allowed[ids]]
        if not len(valid):
            continue
        train = allowed.copy()
        train[max(0, a - 250):min(len(y), b + 250)] = False
        sample = np.flatnonzero(train)[::3]
        if set(np.unique(y[sample])) != {0, 1, 2}:
            continue
        model = fit_model(x[sample], y[sample], names)
        probabilities.append(predict(model, x[valid], names))
        targets.append(y[valid])
        folds.append({'inner_fold': inner, 'training_frames': len(sample),
                      'calibration_frames': len(valid)})
    if not probabilities:
        raise ValueError('Not enough classes for blocked probability calibration')
    p, labels = np.concatenate(probabilities), np.concatenate(targets)
    parameters, objective = fit_calibration(p, labels)
    return parameters, {'folds': folds, 'calibration_frames': len(labels),
                        'regularized_objective': objective}
