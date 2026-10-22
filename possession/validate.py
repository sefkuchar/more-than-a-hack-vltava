"""Five-fold blocked cross-validation with a ten-second training embargo."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from possession.duration import decode,PRIOR_PATH,EMISSION_TEMPERATURE
from possession.features import build_features
from possession.model import fit_model,predict
from possession.control import control_onsets,snap_control_boundaries,recover_camera_spells
from possession.calibration import blocked_calibration,calibrated,blend_teams
from possession.train import MODEL_TEMPERATURE
from starter.baseline import write_submission

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',action='store_true',help='Validate the previous single-classifier model separately')
    args=parser.parse_args()
    temperature=EMISSION_TEMPERATURE if args.baseline else MODEL_TEMPERATURE
    x,names,raw=build_features(broadcast=True)
    if not args.baseline:
        paper_x,paper_names,raw=build_features(broadcast=True,paper_evidence=True)
    stable=x[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    onsets=control_onsets(raw['nearest'],stable,raw['observed'],raw.get('kick_accel'))
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    runs=pm.valid_runs(gt)
    if len(runs)!=1:
        raise ValueError('This validation protocol expects the published contiguous public window')
    start,stop=runs[0]
    core=x[start:stop]
    if not args.baseline:
        paper_core=paper_x[start:stop]
    y=gt[start:stop]
    prediction=np.full(len(gt),pm.X,dtype=np.int8)
    folds=[]
    metric_names=('spell_f1_mean','macro_f1')
    for fold,ids in enumerate(np.array_split(np.arange(len(y)),5)):
        a,b=int(ids[0]),int(ids[-1])+1
        train=np.ones(len(y),bool)
        train[max(0,a-250):min(len(y),b+250)]=False
        assert not train[ids].any(), 'Cross-validation test frames entered training'
        sample=np.flatnonzero(train)[::3]
        model=fit_model(core[sample],y[sample],names)
        # The SAME label-excluded model supplies every emission in this decoding.
        probability=predict(model,core,names)
        parameters,calibration=blocked_calibration(core,y,names,train)
        probability=calibrated(probability,parameters)
        paper_details={}
        if not args.baseline:
            paper_model=fit_model(paper_core[sample],y[sample],paper_names)
            paper_parameters,paper_calibration=blocked_calibration(paper_core,y,paper_names,train)
            paper_probability=calibrated(predict(paper_model,paper_core,paper_names),paper_parameters)
            probability=blend_teams(probability,paper_probability)
            paper_details={'paper_calibration_parameters':paper_parameters.tolist(),
                           'paper_inner_calibration':paper_calibration}
        decoded,changes=snap_control_boundaries(decode(probability,temperature=temperature),onsets,start)
        recovered=0
        if not args.baseline:
            decoded,spans=recover_camera_spells(
                decoded,probability,raw['camera_corrected_nearest'][start:stop],
                raw['camera_visible'][start:stop],raw['camera_correction'][start:stop])
            recovered=len(spans)
        prediction[start+a:start+b]=decoded[a:b]
        fold_score=pm.evaluate(y[ids],decoded[ids])
        folds.append({'fold':fold,'held_out_first_frame':start+a,
            'held_out_last_frame':start+b-1,'training_frames':len(sample),
            'metrics':{k:fold_score[k] for k in (*metric_names,'n_gt_spells','n_pred_spells','per_class')},
            'calibration_parameters':parameters.tolist(),'inner_calibration':calibration,
            'contact_refined_boundaries_full_context':len(changes),
            'camera_recovered_spells_full_context':recovered,**paper_details})
        print('Cross-validation fold',fold+1,{k:fold_score[k] for k in metric_names},flush=True)
    score=pm.evaluate(gt,prediction,bootstrap_n=1000)
    fold_scores=np.array([[fold['metrics'][k] for k in metric_names] for fold in folds])
    cv={'n_splits':len(folds),'embargo_frames':250,
        'pooled_out_of_fold':{k:score[k] for k in metric_names},
        'mean_per_fold':dict(zip(metric_names,fold_scores.mean(axis=0).tolist())),
        'std_per_fold':dict(zip(metric_names,fold_scores.std(axis=0).tolist())),
        'std_convention':'Population standard deviation (ddof=0)',
        'aggregation_note':'Pooled scores use the official metric on all out-of-fold predictions. Fold means score each minute separately: spells are clipped at fold edges, and absent H/A/D classes have F1 zero under the official macro metric.'}
    fields=('spell_f1_mean','macro_f1','n_gt_spells','n_pred_spells',
            'spell_bootstrap_low','spell_bootstrap_high','bootstrap_low','bootstrap_high')
    report={'validation_type':'blocked_cross_validation','cross_validation':cv,
        'protocol':'5-fold blocked cross-validation: contiguous minute folds; 250-frame training embargo on both sides; each fold decoded with its own label-excluded model; nested blocked probability calibration',
        'method':'team-symmetric gradient boosting with nested probability calibration' if args.baseline else 'team-symmetric base/cross-camera classifiers, nested probability calibration, equal conditional H/A blend retaining base D probability',
        'decoder':'calibrated probabilities, external duration semi-Markov at25Hz, then observed-contact refinement of existing H/A boundaries' + ('' if args.baseline else ' with SG7 acceleration-confirmed contact candidates and camera-corrected short-spell recovery'),
        'emission_temperature':temperature,
        'duration_prior_sha256':hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),
        'features':len(names) if args.baseline else len(paper_names),'folds':folds,'model':{k:score[k] for k in fields},
        'per_class':score['per_class'],'spell_f1':score['spell_f1'],
        'caveats':['Only 17 public possessions; this protocol does not establish full-match performance.',
            'Features and hyperparameters were compared on this same small sample, so selection optimism remains.',
            'Fold boundaries can clip or merge predicted spells; no random frame validation was used.'],
        'baselines':{}}
    for path in sorted((ROOT/'evaluation/results').glob('baseline*.csv')):
        p=pm.label_codes(pm.read_label_csv(path))
        m=pm.evaluate(gt,p)
        report['baselines'][path.stem]={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}
    prefix='baseline_' if args.baseline else ''
    out=ROOT/('experiments/'+prefix+'validation_predictions.csv')
    write_submission(str(out),np.array(pm.LABELS)[prediction])
    report['prediction_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    (ROOT/('experiments/'+prefix+'validation_report.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(cv,indent=2),flush=True)


if __name__=='__main__':
    main()
