"""Separate live/dead and conditional team learning under the existing validation."""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from possession.model import fit_model,predict,swap_teams
from possession.decode import viterbi


def joint(team,dead):
    return np.column_stack([team*(1-dead[:,None]),dead])


if __name__=='__main__':
    x,names,_=build_features(broadcast=True)
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(gt)[0];x,y=x[start:stop],gt[start:stop]
    methods=('hierarchical','team_head','dead_head')
    out={m:np.empty(len(y),np.int8) for m in methods}
    oof={m:np.empty((len(y),3)) for m in methods}
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500;train=np.ones(len(y),bool)
        train[max(0,a-250):min(len(y),b+250)]=False;ids=np.flatnonzero(train)[::3]
        base=fit_model(x[ids],y[ids],names);p=predict(base,x,names)
        tx=np.concatenate([x[ids],swap_teams(x[ids],names)])
        ty=np.concatenate([y[ids],np.where(y[ids]<2,1-y[ids],2)])
        dead=HistGradientBoostingClassifier(**base.get_params()).fit(tx,ty==2)
        alive=ty<2
        team=HistGradientBoostingClassifier(**base.get_params()).fit(tx[alive],ty[alive])
        sx=swap_teams(x,names)
        pd=(dead.predict_proba(x)[:,1]+dead.predict_proba(sx)[:,1])/2
        pt=(team.predict_proba(x)+team.predict_proba(sx)[:,[1,0]])/2
        base_team=p[:,:2]/p[:,:2].sum(axis=1)[:,None]
        for m,prob in (('hierarchical',joint(pt,pd)),('team_head',joint(pt,p[:,2])),
                       ('dead_head',joint(base_team,pd))):
            out[m][a:b]=viterbi(np.log(prob.clip(1e-5,1)),32)[a:b]
            oof[m][a:b]=prob[a:b]
        print('hierarchical isolated fold',fold,flush=True)
    results=[]
    for m,pred in out.items():
        score=pm.evaluate(y,pred)
        row=dict(method=m,penalty=32,**{k:score[k] for k in
            ('spell_f1_mean','macro_f1','n_pred_spells','alive_dead_f1','team_acc_both_alive')})
        results.append(row);print(row,flush=True)
    np.savez_compressed(ROOT/'experiments/cache/hierarchical_validation.npz',
        **{m+'_32':p for m,p in out.items()},**{m+'_probability':p for m,p in oof.items()})
    (ROOT/'experiments/hierarchical_results.json').write_text(json.dumps(results,indent=2)+'\n')
