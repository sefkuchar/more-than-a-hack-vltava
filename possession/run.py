"""Load the saved IDSSE-trained model and label every frame of the provided match."""

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import sklearn

from games import GAME, half_video_blocks
from evaluation import possession_metrics as pm
from possession.calibration import calibrated
from possession.duration import PRIOR_PATH
from possession.external import kick_acceleration
from possession.features import build_features
from possession.model import predict
from possession.pipeline import decode_segments, onsets_from
from possession.train import MODEL_PATH
from starter.baseline import write_submission

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'Vltava.csv')
    parser.add_argument('--model', type=Path, default=MODEL_PATH)
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError('Saved model missing. Run: python -m possession.train')
    bundle = joblib.load(args.model)
    if bundle.get('format_version') != 3 or bundle.get('sklearn_version') != sklearn.__version__:
        raise ValueError('Model format/version mismatch; install requirements-model.txt or retrain')
    features, names, raw = build_features(use_tags=bundle['use_tags'], broadcast=bundle['broadcast'])
    if names != bundle['features']:
        raise ValueError('Feature schema differs from the saved model; retrain with possession.train')
    blocks = [(start, end + 1) for _, start, end in half_video_blocks(GAME)]
    half = np.zeros(len(features), dtype=np.int8)
    for k, (start, stop) in enumerate(blocks, 1):
        half[start:stop] = k
    raw['kick_accel'] = np.where(half > 0, kick_acceleration(raw['ball'], half), 0).astype(np.float32)
    probability = calibrated(predict(bundle['classifier'], features, names), bundle['calibration'])
    codes, refined = decode_segments(probability, blocks, bundle['duration_prior'],
                                     bundle['emission_temperature'], onsets_from(features, names, raw),
                                     bundle['refine'])
    labels = np.array(pm.LABELS)[codes]
    write_submission(str(args.out), labels)
    reason = pm.validate_submission(pm.read_label_csv(args.out), len(features))
    if reason:
        raise ValueError(reason)
    report = {
        'model': 'team-symmetric gradient boosting trained on IDSSE (other games), calibrated and '
                 'tuned on a held-out IDSSE match, duration semi-Markov decoding',
        'training_data': bundle['training_data'],
        'provided_game_labels_used': False,
        'emission_temperature': bundle['emission_temperature'], 'refine': bundle['refine'],
        'contact_refined_boundaries': refined,
        'model_sha256': hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'duration_prior_sha256': hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),
        'submission_sha256': hashlib.sha256(args.out.read_bytes()).hexdigest(),
    }
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Wrote', args.out, flush=True)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
