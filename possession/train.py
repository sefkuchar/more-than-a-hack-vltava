"""Train the possession model on external IDSSE matches only (rules of 23 Oct 2026).

* Classifier: IDSSE J03WOH + J03WQQ (DFL/Sportec provider labels, other games).
* Calibration, decoder temperature and boundary refinement: chosen on IDSSE J03WOY,
  a third match the classifier never sees.
* The provided BAR-ATH match contributes no labels. Only its unlabelled
  ball-visibility pattern is used to corrupt the external ball track.
"""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn

from evaluation import possession_metrics as pm
from possession.calibration import calibrated, fit_calibration
from possession.duration import PRIOR_PATH
from possession.external import (TRAIN_MATCHES, VALIDATION_MATCH, corrupt_ball, load_match,
                                 match_features, segments, target_visibility)
from possession.features import tracking
from possession.model import fit_model, predict
from possession.pipeline import decode_segments, onsets_from

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models/possession_model.joblib'
TEMPERATURES = (1., 2., 3., 5., 8.)
TRAIN_STRIDE = 4


def evaluation_codes(codes, half):
    out = np.array(codes, dtype=np.int8, copy=True)
    out[np.flatnonzero(np.diff(half)) + 1] = pm.X  # keep halves as separate scored runs
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    prior = json.loads(PRIOR_PATH.read_text())
    _, target_ball, _ = tracking()
    visibility = target_visibility(target_ball)
    print('target ball visibility in live halves: {:.3f}'.format(visibility.mean()), flush=True)

    xs, ys, names, counts = [], [], None, {}
    for k, match in enumerate(TRAIN_MATCHES):
        m = load_match(match)
        for variant, ball in (('clean', m['ball']), ('corrupt', corrupt_ball(m['ball'], visibility, seed=k))):
            x, names, _ = match_features(m, ball, names)
            rows = np.flatnonzero(m['labels'] < 3)[::TRAIN_STRIDE]
            xs.append(x[rows])
            ys.append(m['labels'][rows])
            counts['{}_{}'.format(match, variant)] = int(len(rows))
            print('train', match, variant, len(rows), flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    classifier = fit_model(x, y, names)

    v = load_match(VALIDATION_MATCH)
    vx, _, vraw = match_features(v, corrupt_ball(v['ball'], visibility, seed=100), names)
    raw_p = predict(classifier, vx, names)
    rows = np.flatnonzero(v['labels'] < 3)[::2]
    calibration, objective = fit_calibration(raw_p[rows], v['labels'][rows])
    p = calibrated(raw_p, calibration)
    onsets = onsets_from(vx, names, vraw)
    gt = evaluation_codes(v['labels'], v['half'])
    grid = []
    for refine in (False, True):
        for temperature in TEMPERATURES:
            codes, refined = decode_segments(p, segments(v['half']), prior, temperature, onsets, refine)
            score = pm.evaluate(gt, evaluation_codes(codes, v['half']))
            grid.append({'temperature': temperature, 'refine': refine, 'refined_edges': refined,
                         'spell_f1': score['spell_f1_mean'], 'macro_f1': score['macro_f1']})
            print(json.dumps(grid[-1]), flush=True)
    best = max(grid, key=lambda g: (round(g['spell_f1'], 6), round(g['macro_f1'], 6), -g['temperature']))

    bundle = {'format_version': 3, 'classifier': classifier, 'features': names,
              'calibration': calibration, 'duration_prior': prior,
              'emission_temperature': best['temperature'], 'refine': best['refine'],
              'use_tags': False, 'broadcast': False, 'sklearn_version': sklearn.__version__,
              'training_data': {'train': list(TRAIN_MATCHES), 'validation': VALIDATION_MATCH,
                                'source': 'IDSSE / DFL-Sportec open data, CC-BY-4.0 (other games)'}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    report = {'model': args.out.name, 'training_rows': counts, 'feature_count': len(names),
              'calibration_parameters': calibration.tolist(), 'calibration_objective': objective,
              'validation_match': VALIDATION_MATCH, 'validation_grid': grid, 'selected': best,
              'target_ball_visibility': float(visibility.mean()),
              'provided_game_labels_used': False, 'sklearn_version': sklearn.__version__,
              'model_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest()}
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('selected', 'calibration_parameters', 'model_sha256')}, indent=2))


if __name__ == '__main__':
    main()
