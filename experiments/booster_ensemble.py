"""Fixed equal-weight ensembles; each fold uses only its label-excluded models."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from possession.model import fit_model,predict
from possession.decode import viterbi


if __name__=='__main__':
    x,names,_=build_features(broadcast=True)
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(gt)[0];x,y=x[start:stop],gt[start:stop]
    with np.load(ROOT/'experiments/cache/xgboost_validation.npz') as z:
        xp=z['fold_probability']
    with np.load(ROOT/'experiments/cache/catboost_validation.npz') as z:
        cp=z['fold_probability']
    out={m:np.empty(len(y),np.int8) for m in ('hist_xgb','hist_xgb_cat')}
    hp=[]
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500;train=np.ones(len(y),bool)
        train[max(0,a-250):min(len(y),b+250)]=False;ids=np.flatnonzero(train)[::3]
        hist=fit_model(x[ids],y[ids],names);p=predict(hist,x,names);hp.append(p)
        for m,prob in (('hist_xgb',(p+xp[fold])/2),('hist_xgb_cat',(p+xp[fold]+cp[fold])/3)):
            out[m][a:b]=viterbi(np.log(prob.clip(1e-5,1)),32)[a:b]
        print('ensemble isolated fold',fold,flush=True)
    results=[]
    for m,pred in out.items():
        score=pm.evaluate(y,pred)
        row=dict(method=m,penalty=32,**{k:score[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')})
        results.append(row);print(row,flush=True)
    np.savez_compressed(ROOT/'experiments/cache/booster_ensemble.npz',hist_fold_probability=np.array(hp),**out)
    (ROOT/'experiments/booster_ensemble_results.json').write_text(json.dumps({
        'results':results,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'feature_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
        'label_sha256':hashlib.sha256(y.tobytes()).hexdigest()},indent=2)+'\n')
