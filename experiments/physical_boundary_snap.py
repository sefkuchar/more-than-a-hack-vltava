"""One fixed contact-onset snap of existing H/A boundaries; no spell insertion."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
from possession.duration import decode,PRIOR_PATH
from evaluation import possession_metrics as pm


def control_onsets(nearest,stable,observed):
    onsets=[]
    for team in (0,1):
        mask=observed&(nearest[:,team]<1.5)&(nearest[:,1-team]>3)&(stable[:,team]>.2)
        edges=np.diff(np.r_[False,mask,False].astype(np.int8))
        starts=np.flatnonzero(edges==1);stops=np.flatnonzero(edges==-1)
        onsets.append(starts[stops-starts>=8])
    return onsets


def snap(prediction,onsets,global_start):
    out=prediction.copy();edges=np.r_[0,np.flatnonzero(out[1:]!=out[:-1])+1,len(out)]
    labels=out[edges[:-1]].copy();changed=[]
    for i in range(1,len(edges)-1):
        old=int(edges[i]);before=int(labels[i-1]);after=int(labels[i])
        if before>=2 or after>=2:continue
        candidates=onsets[after]-global_start
        candidates=candidates[(np.abs(candidates-old)<=50)&(candidates>edges[i-1])&(candidates<edges[i+1])]
        if not len(candidates):continue
        new=int(candidates[np.argmin(np.abs(candidates-old))])
        edges[i]=new
        if new!=old:changed.append((old,new))
    for i,label in enumerate(labels):out[edges[i]:edges[i+1]]=label
    original_edges=np.r_[0,np.flatnonzero(prediction[1:]!=prediction[:-1])+1,len(out)]
    new_edges=np.r_[0,np.flatnonzero(out[1:]!=out[:-1])+1,len(out)]
    assert len(original_edges)==len(new_edges)
    assert np.array_equal(prediction[original_edges[:-1]],out[new_edges[:-1]])
    assert np.all(np.diff(new_edges)>0)
    assert np.array_equal(prediction==2,out==2)
    return out,changed


def main():
    x,names,raw=build_features(broadcast=True)
    # The physical onsets use the full raw sequence before any context slicing.
    stable=x[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    onsets=control_onsets(raw['nearest'],stable,raw['observed'])
    start,stop=11185,18685;xs=x[start:stop]
    provenance=json.loads((ROOT/'experiments/booster_ensemble_results.json').read_text())
    digest=hashlib.sha256(xs.tobytes()).hexdigest()
    assert digest==provenance['feature_sha256']
    assert hashlib.sha256((ROOT/'experiments/booster_ensemble.py').read_bytes()).hexdigest()==provenance['source_sha256']
    path=ROOT/'experiments/cache/booster_ensemble.npz'
    with np.load(path) as saved:probability=saved['hist_fold_probability']
    assert probability.shape==(5,7500,3)
    original=np.empty(7500,np.int8);out=np.empty_like(original);contexts=[];basecontexts=[];records=[]
    for fold,p in enumerate(probability):
        base=decode(p);pred,changes=snap(base,onsets,start)
        a,b=fold*1500,(fold+1)*1500
        original[a:b]=base[a:b];out[a:b]=pred[a:b];contexts.append(pred);basecontexts.append(base)
        shift=np.array([b-a for a,b in changes])
        records.append(dict(fold=fold,changed_boundaries_full_context=len(changes),
             changed_frames_full_context=int(np.sum(pred!=base)),
             absolute_shift_frames=int(np.abs(shift).sum()),
             changed_boundaries_heldout=sum(a<=old<b or a<=new<b for old,new in changes)))
    # Possession labels are opened only after all predictions are fixed.
    truth=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[start:stop]
    assert hashlib.sha256(truth.tobytes()).hexdigest()==provenance['label_sha256']
    results={}
    for name,pred in [('static_duration',original),('contact_boundary_snap',out)]:
        m=pm.evaluate(truth,pred);results[name]={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}
    report=dict(results=results,folds=records,changed_oof_frames=int(np.sum(out!=original)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        feature_sha256=digest,duration_prior_sha256=hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),
        cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        global_control_onsets=[len(v) for v in onsets],
        protocol='Fixed existing directH/A boundary snap to nearest actual global newteam control onset within50frames. Distances1.5/3m,stable>.2,observed,run>=8frames. Orderedpositive spells andDmask preserved perfullfoldcontext. No new spells,GTtiming or parameter search. Each label-excluded fold decoded+snapped in fullcontext before core retained.')
    np.savez_compressed(ROOT/'experiments/cache/physical_boundary_snap.npz',prediction=out,
        original=original,fold_prediction=np.stack(contexts),fold_original=np.stack(basecontexts))
    (ROOT/'experiments/physical_boundary_snap_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
