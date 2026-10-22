"""Decode each held-out minute entirely with its own label-excluded model."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from possession.decode import viterbi
from experiments.compare import fit_model, predict, metrics
from experiments.transfer import external_features


if __name__ == '__main__':
    x,names,_=build_features(broadcast=True)
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(gt)[0]
    x,y=x[start:stop],gt[start:stop]
    ex,ey=external_features(names)
    external=fit_model(ex,ey,names)
    q=predict(external,x,names)
    costs=(8,16,32)
    methods=('target','mean','product')
    decoded={m:{c:np.empty(len(y),dtype=np.int8) for c in costs} for m in methods}
    oof=np.empty_like(q)
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500
        train=np.ones(len(y),dtype=bool)
        train[max(0,a-250):min(len(y),b+250)]=False
        ids=np.flatnonzero(train)[::3]
        model=fit_model(x[ids],y[ids],names)
        p=predict(model,x,names)
        oof[a:b]=p[a:b]
        for name,prob in (('target',p),('mean',(p+q)/2),('product',np.sqrt(p*q))):
            for cost in costs:
                decoded[name][cost][a:b]=viterbi(np.log(prob.clip(1e-5,1)),cost)[a:b]
        print('isolated fold',fold,flush=True)
    results=[]
    arrays=dict(probability=oof,external_probability=q)
    for method in methods:
        for cost in costs:
            pred=decoded[method][cost]
            row=dict(method=method,penalty=cost,**metrics(y,pred))
            results.append(row);print(row,flush=True)
            arrays[method+'_'+str(cost)]=pred
    np.savez_compressed(ROOT/'experiments/cache/isolated_validation.npz',**arrays)
    (ROOT/'experiments/isolated_results.json').write_text(json.dumps(results,indent=2)+'\n')
