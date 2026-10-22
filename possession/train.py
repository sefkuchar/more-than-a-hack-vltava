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
from possession.duration import PRIOR_PATH, EMISSION_TEMPERATURE
from possession.features import build_features
from possession.model import fit_model

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models/possession_model.joblib'
BASELINE_MODEL_PATH = ROOT / 'models/possession_baseline.joblib'
MODEL_TEMPERATURE = 2.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--baseline', action='store_true', help='Rebuild the previous single-classifier model separately')
    args = parser.parse_args()
    args.paper_blend = not args.baseline
    args.out = args.out or (BASELINE_MODEL_PATH if args.baseline else MODEL_PATH)
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
              'emission_temperature': EMISSION_TEMPERATURE if args.baseline else MODEL_TEMPERATURE,
              'sklearn_version': sklearn.__version__}
    if args.paper_blend:
        paper_x, paper_names, _ = build_features(broadcast=True, paper_evidence=True)
        paper_core = paper_x[start:stop]
        paper_parameters, paper_details = blocked_calibration(paper_core, y, paper_names)
        bundle['format_version'] = 2
        bundle['paper_blend'] = {'classifier': fit_model(paper_core[ids], y[ids], paper_names),
                                 'features': paper_names, 'calibration': paper_parameters}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.out, compress=3)
    report = {'model': args.out.name, 'training_frames': len(ids),
              'emission_temperature': bundle['emission_temperature'],
              'feature_count': len(names), 'calibration_parameters': parameters.tolist(),
              'calibration_training': details, 'sklearn_version': sklearn.__version__,
              'feature_sha256': hashlib.sha256(core.tobytes()).hexdigest(),
              'public_labels_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
              'model_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
              'external_data_required': False,
              'note': 'Classifier and calibrator use allowed public labels. Use possession.validate for held-out performance.'}
    if args.paper_blend:
        report['paper_blend'] = {'weight': .5, 'fusion': 'mean conditional H/A probabilities, retain base D',
                                'boundary_refinement': 'sustained contact or observed SG7 acceleration-confirmed contact',
                                'feature_count': len(paper_names),
                                'feature_sha256': hashlib.sha256(paper_core.tobytes()).hexdigest(),
                                'calibration_parameters': paper_parameters.tolist(),
                                'calibration_training': paper_details}
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
