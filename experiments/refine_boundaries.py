"""One fixed physical-contact postprocessor; labels used only after predictions exist."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
import possession_metrics as pm


def exclusive_control_anchors(prediction, nearest, observed, radius=1.5, opponent_radius=3., minimum_frames=8):
    """Override only sustained, observed, exclusive ball-player proximity.

    Fixed physical scales: 1.5 m control zone, 3 m opponent exclusion and
    0.32 s persistence at 25 Hz. No label/time inputs and no fitted thresholds.
    Missing frames break confirmation rather than silently interpolate it.
    """
    out=prediction.copy()
    for team in (0,1):
        confident=observed & (nearest[:,team]<radius) & (nearest[:,1-team]>opponent_radius)
        edges=np.diff(np.r_[False,confident,False].astype(int))
        for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
            if b-a>=minimum_frames and np.all(prediction[a:b]!=2):out[a:b]=team
    return out

if __name__=='__main__':
    z=np.load(ROOT/'experiments/cache/isolated_validation.npz')
    base=z['target_32']
    _,_,raw=build_features()
    # Public range comes from documented input coverage, not event boundaries.
    start=11185;stop=start+len(base)
    out=exclusive_control_anchors(base,raw['nearest'][start:stop],raw['observed'][start:stop])
    np.savez_compressed(ROOT/'data/tactical/contact_refinement.npz',prediction=out)
    truth=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[start:stop]
    for name,pred in [('isolated_base',base),('exclusive_contact',out)]:
        m=pm.evaluate(truth,pred)
        print(name,{k:m[k] for k in ['spell_f1_mean','macro_f1','n_pred_spells']})
    change=np.flatnonzero(out!=base)
    print('changed_frames',len(change),'corrected',int(np.sum((out==truth)&(base!=truth))),
          'broken',int(np.sum((base==truth)&(out!=truth))))
