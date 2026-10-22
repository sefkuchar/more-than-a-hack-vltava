"""Five contiguous minute folds, ten-second embargo, and isolated temporal decoding."""

import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from possession.duration import decode,PRIOR_PATH,EMISSION_TEMPERATURE
from possession.features import build_features
from possession.model import fit_model,predict
from possession.control import control_onsets,snap_control_boundaries
from possession.calibration import blocked_calibration,calibrated
from starter.baseline import write_submission

ROOT=Path(__file__).resolve().parents[1]


def main():
    x,names,raw=build_features(broadcast=True)
    stable=x[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    onsets=control_onsets(raw['nearest'],stable,raw['observed'])
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    runs=pm.valid_runs(gt)
    if len(runs)!=1:
        raise ValueError('This validation protocol expects the published contiguous public window')
    start,stop=runs[0]
    core=x[start:stop]
    y=gt[start:stop]
    prediction=np.full(len(gt),pm.X,dtype=np.int8)
    folds=[]
    for fold,ids in enumerate(np.array_split(np.arange(len(y)),5)):
        a,b=int(ids[0]),int(ids[-1])+1
        train=np.ones(len(y),bool)
        train[max(0,a-250):min(len(y),b+250)]=False
        sample=np.flatnonzero(train)[::3]
        model=fit_model(core[sample],y[sample],names)
        # The SAME label-excluded model supplies every emission in this decoding.
        probability=predict(model,core,names)
        parameters,calibration=blocked_calibration(core,y,names,train)
        probability=calibrated(probability,parameters)
        decoded,changes=snap_control_boundaries(decode(probability),onsets,start)
        prediction[start+a:start+b]=decoded[a:b]
        folds.append({'fold':fold,'held_out_first_frame':start+a,
            'held_out_last_frame':start+b-1,'training_frames':len(sample),
            'calibration_parameters':parameters.tolist(),'inner_calibration':calibration,
            'contact_refined_boundaries_full_context':len(changes)})
        print('Isolated validation fold',fold,'complete',flush=True)
    score=pm.evaluate(gt,prediction,bootstrap_n=1000)
    fields=('spell_f1_mean','macro_f1','n_gt_spells','n_pred_spells',
            'spell_bootstrap_low','spell_bootstrap_high','bootstrap_low','bootstrap_high')
    report={'protocol':'5 contiguous minute folds; 250-frame training embargo on both sides; each fold decoded with its own label-excluded model',
        'method':'team-symmetric gradient boosting with nested probability calibration',
        'decoder':'calibrated probabilities, external duration semi-Markov at25Hz, then observed-contact refinement of existing H/A boundaries',
        'emission_temperature':EMISSION_TEMPERATURE,
        'duration_prior_sha256':hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),
        'features':len(names),'folds':folds,'model':{k:score[k] for k in fields},
        'caveats':['Only 17 public possessions; this protocol does not establish full-match performance.',
            'Features and hyperparameters were compared on this same small sample, so selection optimism remains.',
            'Fold boundaries can clip or merge predicted spells; no random frame validation was used.'],
        'baselines':{}}
    for path in sorted((ROOT/'evaluation/results').glob('baseline*.csv')):
        p=pm.label_codes(pm.read_label_csv(path))
        m=pm.evaluate(gt,p)
        report['baselines'][path.stem]={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}
    out=ROOT/'experiments/validation_predictions.csv'
    write_submission(str(out),np.array(pm.LABELS)[prediction])
    report['prediction_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    (ROOT/'experiments/validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['model'],indent=2),flush=True)


if __name__=='__main__':
    main()
