"""Unified nested probability calibration and dead-boundary timing calibration."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
from possession.model import fit_model,predict
from possession.duration import decode,PRIOR_PATH
from experiments.physical_boundary_snap import control_onsets,snap
from experiments.nested_probability_calibration import fit_calibration,calibrated
from experiments.dead_boundary_calibration import offsets,move_dead_edges,summary
from evaluation import possession_metrics as pm


def main():
    x,names,raw=build_features(broadcast=True);start,stop=11185,18685;xs=x[start:stop]
    provenance=json.loads((ROOT/'experiments/booster_ensemble_results.json').read_text())
    featurehash=hashlib.sha256(xs.tobytes()).hexdigest();assert featurehash==provenance['feature_sha256']
    assert hashlib.sha256((ROOT/'experiments/booster_ensemble.py').read_bytes()).hexdigest()==provenance['source_sha256']
    stable=x[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    onsets=control_onsets(raw['nearest'],stable,raw['observed'])
    cache=ROOT/'experiments/cache/booster_ensemble.npz'
    with np.load(cache) as saved:outerprobs=saved['hist_fold_probability']
    y=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[start:stop]
    assert hashlib.sha256(y.tobytes()).hexdigest()==provenance['label_sha256']
    plain=np.empty(7500,np.int8);timed=np.empty_like(plain);contexts=[];records=[];innercache={}
    for outer in range(5):
        a,b=outer*1500,(outer+1)*1500
        outertrain=np.ones(7500,bool);outertrain[max(0,a-250):min(7500,b+250)]=False
        predictions={};pp=[];yy=[];calids=[]
        for inner in range(5):
            if inner==outer:continue
            c,d=inner*1500,(inner+1)*1500
            train=outertrain.copy();train[max(0,c-250):min(7500,d+250)]=False
            ids=np.flatnonzero(train)[::3]
            assert not train[a:b].any() and not train[c:d].any()
            if set(np.unique(y[ids]))!={0,1,2}:continue
            model=fit_model(xs[ids],y[ids],names)
            p=predict(model,xs,names);predictions[inner]=p
            valid=np.arange(c,d);valid=valid[outertrain[valid]]
            pp.append(p[valid]);yy.append(y[valid]);calids.extend(valid.tolist())
            innercache[f'outer{outer}_inner{inner}_probability']=p
        pp=np.concatenate(pp);yy=np.concatenate(yy);calids=np.array(calids)
        assert outertrain[calids].all()
        parameters,loss=fit_calibration(pp,yy)
        perturbed=y.copy();perturbed[~outertrain]=(perturbed[~outertrain]+1)%3
        assert np.array_equal(yy,perturbed[calids])
        # Refit on the same allowed references to verify deterministic isolation.
        check_parameters,_=fit_calibration(pp,perturbed[calids])
        assert np.allclose(parameters,check_parameters,rtol=0,atol=1e-12)
        samples={'into_dead':[],'out_of_dead':[]};inners=[]
        for inner,p in predictions.items():
            c,d=inner*1500,(inner+1)*1500
            pred,_=snap(decode(calibrated(p,parameters)),onsets,start)
            found=offsets(y,pred,outertrain,c,d)
            assert found==offsets(perturbed,pred,outertrain,c,d)
            for key in samples:samples[key].extend(found[key])
            inners.append(dict(inner_fold=inner,offset_counts={k:len(v) for k,v in found.items()}))
        bias={k:int(np.clip(np.rint(np.median(v)),-50,50)) if len(v)>=2 else 0 for k,v in samples.items()}
        probability=calibrated(outerprobs[outer],parameters)
        pred,_=snap(decode(probability),onsets,start)
        shifted,count,skipped=move_dead_edges(pred,bias)
        plain[a:b]=pred[a:b];timed[a:b]=shifted[a:b];contexts.append(shifted)
        records.append(dict(outer_fold=outer,scale=float(np.exp(parameters[0])),log_scale=float(parameters[0]),
             dead_intercept=float(parameters[1]),calibration_frames=len(yy),regularized_objective=loss,
             bias_frames=bias,offset_statistics={k:summary(v) for k,v in samples.items()},inner_folds=inners,
             moved_dead_edges_full_context=count,skipped_crossing_edges=skipped))
        print('unifiednested fold',outer,'bias',bias,flush=True)
    scores={}
    for name,pred in [('nested_probability_calibration_only',plain),('nested_calibration_and_timing',timed)]:
        m=pm.evaluate(y,pred);scores[name]={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}
    sources=['experiments/nested_calibration_timing.py','experiments/nested_probability_calibration.py',
             'experiments/dead_boundary_calibration.py','experiments/physical_boundary_snap.py',
             'possession/features.py','possession/model.py','possession/duration.py']
    report=dict(results=scores,folds=records,changed_oof_frames=int(np.sum(timed!=plain)),
        source_sha256={s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in sources},
        feature_sha256=featurehash,prior_sha256=hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),
        base_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        protocol='Sameoutertrain-onlyinnerOOF references learnregularized2parameter probabilitycalibration. Then calibratedinnerfullcontexts duration5+fixedHAsnap yieldtypedbinaryDtimingoffsetmatches inallowedinnercore/bothneighborsoutertrain. Medianmin2 clipped±50. ApplybothcalibrationandDoffsetstolabel-excludedoutermodel. No newparameters/grid; outerlabelperturbation leavescalibratorparametersandoffsetsexactunchanged. ExistingDedgeordering/positivity/labelsequenceandpureHApositionschecked.')
    np.savez_compressed(ROOT/'experiments/cache/nested_calibration_timing.npz',prediction=timed,
        calibration_only=plain,fold_prediction=np.stack(contexts),**innercache)
    (ROOT/'experiments/nested_calibration_timing_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
