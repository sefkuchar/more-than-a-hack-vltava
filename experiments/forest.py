"""Compare a smoother randomized-tree learner using the same isolated folds."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from possession.decode import viterbi
from experiments.compare import fit_model,predict,metrics


if __name__=='__main__':
    x,names,_=build_features(broadcast=True)
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(gt)[0];x,y=x[start:stop],gt[start:stop]
    outputs={};costs=(8,16,32)
    decoded={m:{c:np.empty(len(y),np.int8) for c in costs} for m in ('forest','ensemble')}
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500;train=np.ones(len(y),bool)
        train[max(0,a-250):min(len(y),b+250)]=False;ids=np.flatnonzero(train)[::3]
        hist=fit_model(x[ids],y[ids],names);p=predict(hist,x,names)
        forest=fit_model(x[ids],y[ids],names,'extra');q=predict(forest,x,names)
        for method,prob in (('forest',q),('ensemble',(p+q)/2)):
            for c in costs:
                decoded[method][c][a:b]=viterbi(np.log(prob.clip(1e-5,1)),c)[a:b]
        print('forest isolated fold',fold,flush=True)
    results=[]
    for m in decoded:
        for c,pred in decoded[m].items():
            outputs[m+'_'+str(c)]=pred
            row=dict(method=m,penalty=c,**metrics(y,pred));results.append(row);print(row,flush=True)
    np.savez_compressed(ROOT/'experiments/cache/forest_validation.npz',**outputs)
    (ROOT/'experiments/forest_results.json').write_text(json.dumps(results,indent=2)+'\n')
