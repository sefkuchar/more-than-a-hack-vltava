"""One fixed label-free adaptive frame grouping experiment."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
from possession.model import swap_teams
from possession.duration import decode,PRIOR_PATH
from experiments.physical_boundary_snap import control_onsets,snap
from evaluation import possession_metrics as pm


def groups_from_features(x,names):
    selected=['contact_H','contact_A','stable_contact_H','stable_contact_A',
              'ball_speed','moving_players','mean_player_speed','observed',
              'distance_H','distance_A']
    signal=x[:,[names.index(k) for k in selected]].astype(float)
    # Scaling uses unlabeled observations only. Each 5-frame mean is one5Hzsample.
    center=signal.mean(0);scale=signal.std(0).clip(.05)
    z=(signal-center)/scale
    bins=[z[a:min(a+5,len(z))].mean(0) for a in range(0,len(z),5)]
    bins=np.asarray(bins);starts=[0];a=0
    while a<len(bins):
        b=min(a+2,len(bins))
        while b<len(bins) and b-a<10:
            shift=np.sqrt(np.mean((bins[b]-bins[a:b].mean(0))**2))
            if shift>.75:break
            b+=1
        if len(bins)-b==1:
            b=len(bins) if b-a<10 else b-1
        starts.append(min(b*5,len(x)));a=b
    edges=np.asarray(starts,dtype=int)
    assert edges[0]==0 and edges[-1]==len(x) and np.all(np.diff(edges)>0)
    assert np.diff(edges).min()>=10 and np.diff(edges).max()<=50
    return edges,selected,center,scale


def main():
    full,names,raw=build_features(broadcast=True)
    start,stop=11185,18685;x=full[start:stop]
    edges,selected,center,scale=groups_from_features(x,names)
    gx=np.stack([x[a:b].mean(0) for a,b in zip(edges[:-1],edges[1:])])
    mirrored=swap_teams(gx,names)
    stable=full[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    onsets=control_onsets(raw['nearest'],stable,raw['observed'])
    # Group geometry and aggregate inputs are fixed before labels are opened.
    y=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[start:stop]
    predictions={name:np.empty(len(y),np.int8) for name in ('group_duration','group_duration_snap')}
    contexts=[];group_probs=[];folds=[]
    lengths=np.diff(edges)
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500
        train=np.ones(len(y),bool);train[max(0,a-250):min(len(y),b+250)]=False
        # Whole-group exclusion removes boundary ambiguity without consultingGT.
        selected_groups=np.array([j for j,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])) if train[lo:hi].all()])
        targets=[];weights=[]
        for j in selected_groups:
            lo,hi=edges[j:j+2]
            counts=np.bincount(y[lo:hi],minlength=3)
            targets.append(counts.argmax());weights.append((hi-lo)/3)
        targets=np.asarray(targets);weights=np.asarray(weights)
        assert set(np.unique(targets))=={0,1,2}
        model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,max_depth=3,
            learning_rate=.07,l2_regularization=10,min_samples_leaf=5,random_state=0)
        trainx=np.concatenate([gx[selected_groups],mirrored[selected_groups]])
        trainy=np.r_[targets,np.where(targets<2,1-targets,targets)]
        model.fit(trainx,trainy,sample_weight=np.tile(weights,2))
        p=(model.predict_proba(gx)+model.predict_proba(mirrored)[:,[1,0,2]])/2
        expanded=np.repeat(p,lengths,axis=0)
        plain=decode(expanded);snapped,changes=snap(plain,onsets,start)
        predictions['group_duration'][a:b]=plain[a:b]
        predictions['group_duration_snap'][a:b]=snapped[a:b]
        contexts.append(expanded);group_probs.append(p)
        folds.append(dict(fold=fold,training_groups=len(selected_groups),training_raw_frames=int(lengths[selected_groups].sum()),
                          snap_changes_full_context=len(changes)))
    scores={}
    for name,pred in predictions.items():
        m=pm.evaluate(y,pred);scores[name]={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}
    report=dict(results=scores,groups=len(gx),duration_frames_quantiles=np.quantile(lengths,[0,.25,.5,.75,1]).tolist(),
        grouping_features=selected,group_scale=scale.tolist(),group_center=center.tolist(),folds=folds,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        feature_sha256=hashlib.sha256(x.tobytes()).hexdigest(),prior_sha256=hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),
        protocol='Label-free5Hzstandardizedphysicalfeatures; RMSshift>.75 fromcurrentgroupmean createsboundaryaftermin.4s,max2s; finaltailredistributedwithoutviolatinglimits. Mean77featurespergroup; majoritytraininglabelonlyforwholegroupsinsidetrainmask,weights=rawlength/3; minleaf5 rescales35stridesamples togroups. Strict5minuteCV10sembargo;samefoldfullcontext probabilitiesexpand25Hz thenexternaldurationT5. Secondfixedoutputusesunchangedphysicalsnap. No groupingthresholdsearch/GTchangepoints.')
    cache=ROOT/'experiments/cache'
    np.savez_compressed(cache/'adaptive_groups.npz',edges=edges,fold_probability=np.stack(contexts),fold_group_probability=np.stack(group_probs),**predictions)
    (ROOT/'experiments/adaptive_groups_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
