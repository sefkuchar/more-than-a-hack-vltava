"""Two-parameter team-symmetric probability calibration using nested held-out rows."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
from possession.model import fit_model,predict
from possession.duration import decode,PRIOR_PATH
from experiments.physical_boundary_snap import control_onsets,snap
from evaluation import possession_metrics as pm


def calibrated(p,parameters):
    loga,b=parameters
    logits=np.exp(loga)*np.log(p.clip(1e-5,1))
    logits[:,2]+=b
    return np.exp(logits-logsumexp(logits,axis=1,keepdims=True))


def fit_calibration(p,y):
    logp=np.log(p.clip(1e-5,1));n=len(y);rows=np.arange(n)
    def objective(parameters):
        loga,b=parameters;a=np.exp(loga)
        logits=a*logp;logits[:,2]+=b
        logz=logsumexp(logits,axis=1)
        loss=np.mean(logz-logits[rows,y])+.01*(loga**2+b**2)
        q=np.exp(logits-logz[:,None]);q[rows,y]-=1
        gradient=np.array([np.mean(np.sum(q*(a*logp),axis=1))+.02*loga,
                           np.mean(q[:,2])+.02*b])
        return loss,gradient
    result=minimize(objective,np.zeros(2),method='L-BFGS-B',jac=True,
                    bounds=[(np.log(.25),np.log(4)),(-2,2)])
    if not result.success:raise ValueError('Calibration optimization failed: '+result.message)
    return result.x,float(result.fun)


def main():
    x,names,raw=build_features(broadcast=True);start,stop=11185,18685;xs=x[start:stop]
    provenance=json.loads((ROOT/'experiments/booster_ensemble_results.json').read_text())
    featurehash=hashlib.sha256(xs.tobytes()).hexdigest();assert featurehash==provenance['feature_sha256']
    assert hashlib.sha256((ROOT/'experiments/booster_ensemble.py').read_bytes()).hexdigest()==provenance['source_sha256']
    cache=ROOT/'experiments/cache/booster_ensemble.npz'
    with np.load(cache) as saved:baseprob=saved['hist_fold_probability']
    stable=x[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    onsets=control_onsets(raw['nearest'],stable,raw['observed'])
    y=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[start:stop]
    assert hashlib.sha256(y.tobytes()).hexdigest()==provenance['label_sha256']
    out=np.empty(7500,np.int8);baseline=np.empty_like(out);contexts=[];records=[];nested_inputs=[]
    for outer in range(5):
        a,b=outer*1500,(outer+1)*1500
        outertrain=np.ones(7500,bool);outertrain[max(0,a-250):min(7500,b+250)]=False
        pp=[];yy=[];calibration_ids=[];inners=[]
        for inner in range(5):
            if inner==outer:continue
            c,d=inner*1500,(inner+1)*1500
            train=outertrain.copy();train[max(0,c-250):min(7500,d+250)]=False
            ids=np.flatnonzero(train)[::3]
            assert not train[a:b].any() and not train[c:d].any()
            if set(np.unique(y[ids]))!={0,1,2}:
                inners.append(dict(inner_fold=inner,skipped='missing training class'));continue
            model=fit_model(xs[ids],y[ids],names)
            valid=np.arange(c,d);valid=valid[outertrain[valid]]
            p=predict(model,xs[valid],names)
            pp.append(p);yy.append(y[valid]);calibration_ids.extend(valid.tolist())
            inners.append(dict(inner_fold=inner,train_frames=len(ids),calibration_frames=len(valid)))
        pp=np.concatenate(pp);yy=np.concatenate(yy);calibration_ids=np.array(calibration_ids)
        assert outertrain[calibration_ids].all()
        parameters,loss=fit_calibration(pp,yy)
        # Outer-label perturbation cannot change calibrator training references.
        changed=y.copy();changed[~outertrain]=(changed[~outertrain]+1)%3
        assert np.array_equal(y[calibration_ids],changed[calibration_ids])
        transformed=calibrated(baseprob[outer],parameters)
        base,_=snap(decode(baseprob[outer]),onsets,start)
        pred,_=snap(decode(transformed),onsets,start)
        baseline[a:b]=base[a:b];out[a:b]=pred[a:b];contexts.append(transformed)
        nested_inputs.append(dict(probability=pp,labels=yy,ids=calibration_ids))
        logp=np.log(pp.clip(1e-5,1))
        before=-float(logp[np.arange(len(yy)),yy].mean())
        after=-float(np.log(calibrated(pp,parameters).clip(1e-12,1))[np.arange(len(yy)),yy].mean())
        records.append(dict(outer_fold=outer,log_scale=float(parameters[0]),scale=float(np.exp(parameters[0])),
            dead_intercept=float(parameters[1]),calibration_frames=len(yy),regularized_objective=loss,
            calibration_log_loss_before=before,calibration_log_loss_after=after,inner_folds=inners))
        print('nestedcalibration fold',outer,parameters,'frames',len(yy),flush=True)
    scores={}
    for name,pred in [('uncalibrated_snap',baseline),('nested_probability_calibrated_snap',out)]:
        m=pm.evaluate(y,pred);scores[name]={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}
    report=dict(results=scores,folds=records,changed_oof_frames=int(np.sum(out!=baseline)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),feature_sha256=featurehash,
        prior_sha256=hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        protocol='Strictnestedoutertrain+innerheldminute+250embargo. Onlyinnerheldrowsinsideoutertrain supplycalibratorlabels. exp(a*logp+b*I_D); a=exp(loga), logaboundslog.25..log4,b[-2,2];meanlogloss+.01*(loga²+b²),L-BFGS-Binitial0,0. SharedHA scale preserves teamsymmetry. Applyouterp,sameT5duration+fixedcontactsnap. No outertimingbias/grid/featurechanges. Outerlabelisolationmaskcheckpassed.')
    arrays=dict(prediction=out,baseline=baseline,fold_probability=np.stack(contexts))
    for i,v in enumerate(nested_inputs):
        for k,value in v.items():arrays[f'inner_outer{i}_{k}']=value
    np.savez_compressed(ROOT/'experiments/cache/nested_probability_calibration.npz',**arrays)
    (ROOT/'experiments/nested_probability_calibration_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
