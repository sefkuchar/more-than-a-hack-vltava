"""Fixed two-camera control anchors; labels opened only after decoding."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
from possession.duration import duration_decode,prior_tables,PRIOR_PATH
from experiments.physical_boundary_snap import control_onsets,snap
from evaluation import possession_metrics as pm


def sustained(mask,minimum=8):
    out=np.zeros_like(mask)
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
        if b-a>=minimum:out[a:b]=True
    return out


def main():
    x,names,raw=build_features(broadcast=True)
    def col(name):return x[:,names.index(name)]
    stable=x[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    tactical=raw['nearest'];broadcast=x[:,[names.index('broadcast_distance_H'),names.index('broadcast_distance_A')]]
    masks=[]
    for k in (0,1):
        mask=raw['observed']&(tactical[:,k]<1.5)&(tactical[:,1-k]>3)&(stable[:,k]>.2)
        mask&=(col('broadcast_visible')>0)&(broadcast[:,k]<1.5)&(broadcast[:,1-k]>3)&(col('broadcast_ball_error')<2)
        masks.append(sustained(mask))
    masks=np.stack(masks,axis=1)
    assert not np.any(masks[:,0]&masks[:,1])
    onsets=control_onsets(raw['nearest'],stable,raw['observed'])
    start,stop=11185,18685;xs=x[start:stop];anchors=masks[start:stop]
    prov=json.loads((ROOT/'experiments/booster_ensemble_results.json').read_text())
    featurehash=hashlib.sha256(xs.tobytes()).hexdigest()
    assert featurehash==prov['feature_sha256']
    assert hashlib.sha256((ROOT/'experiments/booster_ensemble.py').read_bytes()).hexdigest()==prov['source_sha256']
    cache=ROOT/'experiments/cache/booster_ensemble.npz'
    with np.load(cache) as z:probs=z['hist_fold_probability']
    density,survival=prior_tables(json.loads(PRIOR_PATH.read_text()),7500);times=np.arange(7501)
    methods=['duration_baseline','anchor_only','anchor_then_snap']
    oof={name:np.empty(7500,np.int8) for name in methods};contexts={name:[] for name in methods};records=[]
    for fold,p in enumerate(probs):
        a,b=fold*1500,(fold+1)*1500
        logp=np.log(p.clip(1e-5,1));forced=logp.copy()
        eligible=anchors&((p[:,0]+p[:,1]>p[:,2])[:,None])
        for k in (0,1):forced[eligible[:,k]]=-12;forced[eligible[:,k],k]=0
        baseline=duration_decode(logp/5,density,survival,times)
        predicted=duration_decode(forced/5,density,survival,times)
        snapped,changes=snap(predicted,onsets,start)
        for name,v in zip(methods,[baseline,predicted,snapped]):oof[name][a:b]=v[a:b];contexts[name].append(v)
        def spells(v):return int(np.sum((np.r_[True,v[1:]!=v[:-1]])&(v<2)))
        records.append(dict(fold=fold,eligible_anchor_frames_full_context=int(eligible.any(1).sum()),
            eligible_anchor_frames_heldout=int(eligible[a:b].any(1).sum()),
            baseline_spells_full_context=spells(baseline),anchor_spells_full_context=spells(predicted),
            snap_changed_boundaries_full_context=len(changes)))
    # The pretrained/physical inputs and all output sequences above are fixed.
    truth=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[start:stop]
    assert hashlib.sha256(truth.tobytes()).hexdigest()==prov['label_sha256']
    scores={}
    for name,v in oof.items():
        m=pm.evaluate(truth,v);scores[name]={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}
    report=dict(results=scores,global_anchor_frames=masks.sum(0).tolist(),public_anchor_frames=anchors.sum(0).tolist(),folds=records,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),feature_sha256=featurehash,
        prior_sha256=hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        protocol='Singlepredefinedtwo-cameraagreementcontrolmask,sustained8frames. Tactical1.5/3m+stable>.2, broadcast1.5/3m+ballerror<2m. SamefoldpLive>pDgate. Anchoredlogp0/-12,temperature5,exactexternaldurationDP. Secondpredeclaredoutputappliesunchangedphysicalboundarysnap. No parametersearch; no GTtiming inanchors.')
    arrays={f'{name}_prediction':oof[name] for name in methods}
    arrays.update({f'{name}_fold_prediction':np.stack(contexts[name]) for name in methods})
    np.savez_compressed(ROOT/'experiments/cache/camera_control_anchors.npz',**arrays,anchors=anchors)
    (ROOT/'experiments/camera_control_anchors_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
