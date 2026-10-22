"""Save the best classifier, calibrator and duration prior inside this repository."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn

from evaluation import possession_metrics as pm
from possession.calibration import blocked_calibration
from possession.duration import PRIOR_PATH
from possession.features import build_features
from possession.model import fit_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models/possession_model.joblib'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    x, names, _ = build_features(broadcast=True)
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    runs = pm.valid_runs(gt)
    if len(runs) != 1:
        raise ValueError('Expected the published contiguous five-minute training window')
    start, stop = runs[0]
    core, y = x[start:stop], gt[start:stop]
    parameters, details = blocked_calibration(core, y, names)
    ids = np.arange(len(y))[::3]
    classifier = fit_model(core[ids], y[ids], names)
    prior = json.loads(PRIOR_PATH.read_text())
    bundle = {'format_version': 1, 'classifier': classifier, 'features': names,
              'calibration': parameters, 'duration_prior': prior,
              'sklearn_version': sklearn.__version__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    report = {'model': args.out.name, 'training_frames': len(ids),
              'feature_count': len(names), 'calibration_parameters': parameters.tolist(),
              'calibration_training': details, 'sklearn_version': sklearn.__version__,
              'feature_sha256': hashlib.sha256(core.tobytes()).hexdigest(),
              'public_labels_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
              'model_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
              'external_data_required': False,
              'note': 'Classifier and calibrator use allowed public labels. Use possession.validate for held-out performance.'}
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
