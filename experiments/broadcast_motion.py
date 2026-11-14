"""One fixed experiment adding camera ball kinematics to the existing model."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from possession.features import build_features, interpolate, rolling
from possession.decode import viterbi
from possession.model import fit_model, predict
from experiments.align_broadcast import aligned_broadcast
from experiments.compare import metrics


def add_motion_features(x, names, raw):
    broadcast = aligned_broadcast()['ball']
    seen = np.isfinite(broadcast).all(axis=1)
    ball = interpolate(broadcast)
    tactical = interpolate(raw['ball'])
    def velocity(values):
        return (np.roll(values, -3, axis=0) - np.roll(values, 3, axis=0)) * (25 / 6)
    v = velocity(ball)
    speed = np.linalg.norm(v, axis=1).clip(0, 50)
    boundary = np.minimum.reduce([ball[:, 0], 105.3 - ball[:, 0], ball[:, 1], 68 - ball[:, 1]])
    idx = np.arange(len(ball))
    previous = np.maximum.accumulate(np.where(seen, idx, -len(ball)))
    following = np.minimum.accumulate(np.where(seen, idx, 2 * len(ball))[::-1])[::-1]
    age = np.minimum(idx - previous, following - idx) / 25
    columns = [speed * seen, boundary.clip(-10, 25) * seen, age.clip(0, 30)]
    added = ['broadcast_ball_speed', 'broadcast_ball_boundary', 'broadcast_observation_age']
    for radius in (12, 50):
        visibility = rolling(seen, radius)
        denominator = np.maximum(visibility, 1e-6)
        columns.extend([rolling((speed < 1.5) * seen, radius) / denominator,
                        rolling(speed * seen, radius) / denominator])
        added.extend([f'broadcast_ball_stationarity_{radius}', f'broadcast_ball_speed_mean_{radius}'])
    discrepancy = np.linalg.norm(v - velocity(tactical), axis=1).clip(0, 50)
    columns.append(discrepancy * seen * raw['observed'])
    added.append('broadcast_velocity_discrepancy')
    result = np.column_stack([x, *columns]).astype(np.float32)
    assert np.isfinite(result).all()
    return result, names + added


if __name__ == '__main__':
    x, names, raw = build_features(broadcast=True)
    x, names = add_motion_features(x, names, raw)
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    x, y = x[start:stop], gt[start:stop]
    assert len(y) == 7500
    source_paths = ['experiments/broadcast_motion.py', 'possession/features.py',
                    'possession/model.py', 'possession/decode.py', 'experiments/align_broadcast.py']
    fingerprints = {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in source_paths}
    fingerprints['feature_matrix'] = hashlib.sha256(x.tobytes()).hexdigest()
    fingerprints['labels'] = hashlib.sha256(y.tobytes()).hexdigest()
    oof = np.empty((len(y), 3))
    decoded = np.empty(len(y), np.int8)
    for fold in range(5):
        a, b = fold * 1500, (fold + 1) * 1500
        train = np.ones(len(y), bool)
        train[max(0, a - 250):min(len(y), b + 250)] = False
        ids = np.flatnonzero(train)[::3]
        model = fit_model(x[ids], y[ids], names)
        probability = predict(model, x, names)
        oof[a:b] = probability[a:b]
        decoded[a:b] = viterbi(np.log(probability.clip(1e-5, 1)), 32)[a:b]
        print('broadcast motion fold', fold, 'features', len(names), flush=True)
    result = dict(method='broadcast_motion', penalty=32, feature_count=len(names),
                  stride=3, folds=5, embargo_frames=250, source_fingerprints=fingerprints,
                  **metrics(y, decoded))
    print(result, flush=True)
    (ROOT / 'experiments/broadcast_motion_results.json').write_text(json.dumps(result, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/broadcast_motion_validation.npz',
                        probability=oof, prediction=decoded, names=names)
