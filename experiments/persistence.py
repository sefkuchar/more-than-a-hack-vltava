"""Bracket the earlier best persistence penalty with isolated-fold decoding."""

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
from experiments.compare import metrics


if __name__=='__main__':
    x,names,_=build_features(broadcast=True)
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(gt)[0];x,y=x[start:stop],gt[start:stop]
    decoded={c:np.empty(len(y),np.int8) for c in (24,48,64)}
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500;train=np.ones(len(y),bool)
        train[max(0,a-250):min(len(y),b+250)]=False;ids=np.flatnonzero(train)[::3]
        model=fit_model(x[ids],y[ids],names);p=predict(model,x,names)
        for c in decoded:
            decoded[c][a:b]=viterbi(np.log(p.clip(1e-5,1)),c)[a:b]
        print('persistence isolated fold',fold,flush=True)
    rows=[]
    for c,pred in decoded.items():
        row=dict(method='target_persistence',penalty=c,**metrics(y,pred));rows.append(row);print(row,flush=True)
    np.savez_compressed(ROOT/'experiments/cache/persistence_validation.npz',
                        **{'target_'+str(c):p for c,p in decoded.items()})
    (ROOT/'experiments/persistence_results.json').write_text(json.dumps(rows,indent=2)+'\n')
