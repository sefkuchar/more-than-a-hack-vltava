"""One predefined physics-confidence temperature experiment; no parameter search."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from possession.duration import duration_decode,prior_tables,PRIOR_PATH


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def temperatures(p,stable):
    winner=p[:,:2].argmax(axis=1);i=np.arange(len(p))
    quality=stable[i,winner]*(1-stable[i,1-winner])*(1-p[:,2])*np.abs(p[:,0]-p[:,1])
    return 5-4*np.clip(quality,0,1)


def main():
    x,names,_=build_features(broadcast=True)
    truth=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(truth)[0];x=x[start:stop];y=truth[start:stop]
    provenance=json.loads((ROOT/'experiments/booster_ensemble_results.json').read_text())
    checks=dict(source_sha256=sha(ROOT/'experiments/booster_ensemble.py'),
                feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),
                label_sha256=hashlib.sha256(y.tobytes()).hexdigest())
    if any(checks[k]!=provenance[k] for k in checks):raise ValueError('Cached histogram provenance mismatch')
    if len(names)!=77:raise ValueError('Expected77 features')
    path=ROOT/'experiments/cache/booster_ensemble.npz'
    with np.load(path) as z:folds=z['hist_fold_probability']
    if folds.shape!=(5,len(y),3):raise ValueError('Unexpected fold probability shape')
    stable=x[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    density,survival=prior_tables(json.loads(PRIOR_PATH.read_text()),len(y))
    times=np.arange(len(y)+1);out=np.empty(len(y),np.int8);base=np.empty_like(out);ts=[]
    for fold,p in enumerate(folds):
        a,b=fold*1500,(fold+1)*1500
        t=temperatures(p,stable)
        logp=np.log(p.clip(1e-5,1))
        base[a:b]=duration_decode(logp/5,density,survival,times)[a:b]
        out[a:b]=duration_decode(logp/t[:,None],density,survival,times)[a:b]
        ts.append(t)
        print('physics temperature fold',fold,flush=True)
    results={}
    for name,pred in [('static_5',base),('physics_adaptive',out)]:
        m=pm.evaluate(y,pred)
        results[name]={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}
    report=dict(results=results,cache_provenance=checks,cache_sha256=sha(path),
                duration_prior_sha256=sha(PRIOR_PATH),source_sha256=sha(Path(__file__)),
                formula='quality=stable[winner]*(1-stable[other])*(1-pD)*abs(pH-pA); T=5-4*clip(quality,0,1)',
                temperature_quantiles=np.quantile(ts,[0,.25,.5,.75,1]).tolist(),
                protocol='Existing5 isolated60second folds and10second embargo; each fold decoded acrossfull7500 thenheldoutcore retained. Quality uses predictions andtrackingfeaturesonly; no label-based threshold or temperature search.')
    (ROOT/'experiments/physics_temperature_results.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(ROOT/'experiments/cache/physics_temperature.npz',prediction=out,static_prediction=base,fold_temperature=np.stack(ts))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
