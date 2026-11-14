"""Propose brief control events using posterior uncertainty plus confirmed contact."""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import possession_metrics as pm
from possession.features import build_features
from experiments.duration_model import duration_decode


def brief_control(labels,probability,nearest,observed):
    """Three-spell proposals require independent physical confirmation.

    A proposal can split one existing spell into three; the minority posterior
    threshold is the requested .65 F1 operating point divided by two. It must
    have 0.32 s exclusive observed contact, lie inside one existing possession,
    and last at most four seconds. No existing D frame can become a possession.
    """
    out=labels.copy()
    for team in (0,1):
        plausible=(probability[:,team]>=.65/2)&(labels==1-team)
        bounds=np.flatnonzero(np.diff(np.r_[False,plausible,False].astype(int)))
        physical=observed&(nearest[:,team]<1.5)&(nearest[:,1-team]>3)
        for a,b in zip(bounds[::2],bounds[1::2]):
            if b-a>100 or a<25 or b+25>len(out):
                continue
            if not np.all(labels[a-25:b+25]==1-team):
                continue
            runs=np.flatnonzero(np.diff(np.r_[False,physical[a:b],False].astype(int)))
            if len(runs) and np.max(runs[1::2]-runs[::2])>=8:
                out[a:b]=team
    return out


if __name__=='__main__':
    with np.load(ROOT/'experiments/cache/booster_ensemble.npz') as z:
        probability=z['hist_fold_probability']
    with np.load(ROOT/'experiments/cache/external_duration_validation.npz') as z:
        density,survival=z['log_density'],z['log_survival']
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(gt)[0];y=gt[start:stop]
    _,_,raw=build_features(broadcast=True)
    nearest=raw['nearest'][start:stop];observed=raw['observed'][start:stop]
    prediction=np.empty(len(y),np.int8)
    for fold in range(5):
        p=probability[fold]
        base=duration_decode(np.log(p.clip(1e-5,1))/5,density,survival,np.arange(len(y)+1))
        refined=brief_control(base,p,nearest,observed)
        a,b=fold*1500,(fold+1)*1500
        prediction[a:b]=refined[a:b]
        print('brief control isolated fold',fold,'changes',int(np.sum(base[a:b]!=refined[a:b])),flush=True)
    score=pm.evaluate(y,prediction)
    report=dict(method='brief_control',**{k:score[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    print(report,flush=True)
    np.savez_compressed(ROOT/'experiments/cache/brief_control_validation.npz',prediction=prediction)
    (ROOT/'experiments/brief_control_results.json').write_text(json.dumps(report,indent=2)+'\n')
