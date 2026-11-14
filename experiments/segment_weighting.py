"""Weight training spells instead of letting long possessions dominate the loss."""

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


def spell_weights(y, train):
    masked=np.where(train,y,3).astype(np.int8)
    weight=np.zeros(len(y))
    for a,b,k in pm.segments(masked,pm.valid_runs(masked)):
        weight[a:b]=np.clip(np.sqrt(250./(b-a)),.3,5.)
    return weight


if __name__=='__main__':
    x,names,_=build_features(broadcast=True)
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(gt)[0]
    x,y=x[start:stop],gt[start:stop]
    results=[];outputs={}
    for method in ('uniform','sqrt'):
        decoded={c:np.empty(len(y),dtype=np.int8) for c in (8,16,32)}
        probability=np.empty((len(y),3))
        for fold in range(5):
            a,b=fold*1500,(fold+1)*1500
            train=np.ones(len(y),bool)
            train[max(0,a-250):min(len(y),b+250)]=False
            weights=None if method=='uniform' else spell_weights(y,train)[train]
            model=fit_model(x[train],y[train],names,sample_weight=weights)
            p=predict(model,x,names)
            probability[a:b]=p[a:b]
            for c in decoded:
                decoded[c][a:b]=viterbi(np.log(p.clip(1e-5,1)),c)[a:b]
            print(method,'isolated fold',fold,flush=True)
        outputs[method+'_probability']=probability
        for c,pred in decoded.items():
            row=dict(method=method,penalty=c,**metrics(y,pred))
            results.append(row);print(row,flush=True)
            outputs[method+'_'+str(c)]=pred
    np.savez_compressed(ROOT/'experiments/cache/weighted_validation.npz',**outputs)
    (ROOT/'experiments/weighted_results.json').write_text(json.dumps(results,indent=2)+'\n')
