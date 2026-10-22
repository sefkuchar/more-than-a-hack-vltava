"""Load the saved best model and generate every frame of the submission."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import joblib
import sklearn

from games import GAME, half_video_blocks
from evaluation import possession_metrics as pm
from possession.features import build_features
from possession.model import predict
from possession.duration import decode,PRIOR_PATH,EMISSION_TEMPERATURE
from possession.control import control_onsets,snap_control_boundaries
from possession.calibration import calibrated
from possession.train import MODEL_PATH
from starter.baseline import write_submission

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=ROOT/'evaluation/results/possession_model.csv')
    parser.add_argument('--model',type=Path,default=MODEL_PATH)
    args=parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError('Saved model missing. Run: python -m possession.train')
    bundle=joblib.load(args.model)
    if bundle.get('format_version')!=1 or bundle.get('sklearn_version')!=sklearn.__version__:
        raise ValueError('Model format/version mismatch; install requirements-model.txt or retrain')
    features,names,raw=build_features(broadcast=True)
    if names!=bundle['features']:
        raise ValueError('Feature schema differs from the saved model; retrain with possession.train')
    stable=features[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    onsets=control_onsets(raw['nearest'],stable,raw['observed'])
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    print('Loaded saved model; {} features'.format(len(names)),flush=True)
    probability=calibrated(predict(bundle['classifier'],features,names),bundle['calibration'])
    codes=np.full(len(features),pm.X,dtype=np.int8)
    refined=0
    for _,start,end in half_video_blocks(GAME):
        print('Decoding frames {}..{}'.format(start,end),flush=True)
        codes[start:end+1],changes=snap_control_boundaries(decode(probability[start:end+1],bundle['duration_prior']),onsets,start)
        refined+=len(changes)
    labels=np.array(pm.LABELS)[codes]
    write_submission(str(args.out),labels)
    table=pm.read_label_csv(args.out)
    reason=pm.validate_submission(table,len(features))
    if reason:
        raise ValueError(reason)
    score=pm.evaluate(gt,codes)
    report={
        'model':'saved team-symmetric gradient boosting, probability calibration, duration semi-Markov, contact boundary refinement',
        'features':names,'emission_temperature':EMISSION_TEMPERATURE,
        'model_sha256':hashlib.sha256(args.model.read_bytes()).hexdigest(),
        'calibration_parameters':bundle['calibration'].tolist(),
        'duration_prior_sha256':hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),
        'contact_refined_boundaries':refined,
        'public_training_score_spell_f1':score['spell_f1_mean'],
        'public_training_score_macro_f1':score['macro_f1'],
        'score_warning':'Public labels were used in training. These are resubstitution scores, not held-out performance.',
        'submission_sha256':hashlib.sha256(args.out.read_bytes()).hexdigest(),
        'private_full_match_score':None,
    }
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Wrote',args.out,flush=True)
    print(json.dumps({k:v for k,v in report.items() if k!='features'},indent=2),flush=True)


if __name__=='__main__':
    main()
