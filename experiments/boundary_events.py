"""Auxiliary boundary likelihood learned only on outer-training label changes."""
import hashlib
import inspect
import json
from pathlib import Path
import sys
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
from possession.model import swap_teams
from experiments.duration_model import duration_decode
import possession_metrics as pm


def event_features(x,names):
    wanted=['contact_H','contact_A','stable_contact_H','stable_contact_A',
            'broadcast_contact_H','broadcast_contact_A','distance_H','distance_A',
            'ball_speed','moving_players','mean_player_speed','boundary','observed',
            'broadcast_visible','broadcast_quality']
    selected=x[:,[names.index(name) for name in wanted]].astype(float)
    columns=[selected];output=list(wanted)
    n=len(x);i=np.arange(n)
    sums=np.vstack([np.zeros(selected.shape[1]),selected.cumsum(axis=0)])
    for radius in (12,25):
        lo=np.maximum(0,i-radius);hi=np.minimum(n,i+radius)
        past=(sums[i]-sums[lo])/np.maximum(i-lo,1)[:,None]
        future=(sums[hi]-sums[i])/np.maximum(hi-i,1)[:,None]
        columns.append(np.abs(future-past))
        output.extend([f'change_{radius}_{name}' for name in wanted])
    return np.concatenate(columns,axis=1).astype(np.float32),output


def expanded_events(y,train):
    targets=np.zeros(len(y),np.int8)
    # Only access labels where BOTH consecutive frames belong to outer train.
    pair=np.flatnonzero(train[1:]&train[:-1])+1
    boundaries=pair[y[pair]!=y[pair-1]]
    for t in boundaries:targets[max(0,t-8):min(len(y),t+9)]=1
    targets[~train]=0
    return targets,boundaries


def fingerprint():
    h=hashlib.sha256()
    for name in ('experiments/boundary_events.py','experiments/duration_model.py',
                 'possession/features.py','possession/model.py','experiments/align_broadcast.py'):
        h.update((ROOT/name).read_bytes())
    for pattern in ('data/tactical/*atd.*','data/broadcast/*atd.*','data/tactical/*smart_tagging.xml',
                    'evaluation/gt/*public.csv','experiments/cache/external_duration_validation.npz',
                    'experiments/cache/booster_ensemble.npz'):
        for p in sorted(ROOT.glob(pattern)):
            s=p.stat();h.update(f'{p.relative_to(ROOT)}:{s.st_size}:{s.st_mtime_ns}'.encode())
    return h.hexdigest()


def main():
    if 'boundary' not in inspect.signature(duration_decode).parameters:
        raise RuntimeError('Waiting for duration_decode boundary API')
    cache=ROOT/'experiments/cache';fp=fingerprint();path=cache/'boundary_events_results.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('source_fingerprint')==fp:print(json.dumps(old,indent=2));return
    x,names,_=build_features(broadcast=True)
    ex,event_names=event_features(x,names)
    mirrored,_=event_features(swap_teams(x,names),names)
    truth=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(truth)[0];y=truth[start:stop]
    ex=ex[start:stop];mirrored=mirrored[start:stop]
    with np.load(cache/'booster_ensemble.npz') as z:base=z['hist_fold_probability']
    with np.load(cache/'external_duration_validation.npz') as z:
        density=z['log_density'];survival=z['log_survival']
    assert base.shape==(5,len(y),3)
    out=np.empty(len(y),np.int8);event_oof=np.empty(len(y));folds=[];probabilities=[];bonuses=[]
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500
        train=np.ones(len(y),bool);train[max(0,a-250):min(len(y),b+250)]=False
        targets,changes=expanded_events(y,train)
        ids=np.flatnonzero(train)[::3]
        tx=np.concatenate([ex[ids],mirrored[ids]])
        ty=np.tile(targets[ids],2)
        prior=float(targets[train].mean())
        model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=5,max_depth=3,
                learning_rate=.07,l2_regularization=10,min_samples_leaf=20,random_state=0)
        model.fit(tx,ty)
        q=(model.predict_proba(ex)[:,1]+model.predict_proba(mirrored)[:,1])/2
        qc=q.clip(1e-5,1-1e-5);pc=np.clip(prior,1e-5,1-1e-5)
        loglr=np.clip(np.log(qc/(1-qc))-np.log(pc/(1-pc)),-4,4)
        bonus=np.r_[loglr,0.];bonus[0]=0.
        pred=duration_decode(np.log(base[fold].clip(1e-5,1))/5,density,survival,np.arange(len(y)+1),boundary=bonus)
        out[a:b]=pred[a:b];event_oof[a:b]=q[a:b]
        # Diagnostic reference has X outside this fold; never used in fitting.
        foldtruth=np.where((np.arange(len(y))>=a)&(np.arange(len(y))<b),y,3).astype(np.int8)
        m=pm.evaluate(foldtruth,pred)
        folds.append(dict(fold=fold,event_prior=prior,training_boundaries=len(changes),
                          train_frames=len(ids),**{k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}))
        probabilities.append(q);bonuses.append(bonus)
        print('boundary fold',fold,folds[-1],flush=True)
    m=pm.evaluate(y,out)
    report=dict(method='auxiliary_boundary_events',source_fingerprint=fp,event_features=event_names,
                temperature=5,change_expansion_frames=8,loglr_clip=4,folds=folds,
                metrics={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')},
                protocol='Fixed event model and prior; samefoldlabel-excluded fullcontext decoding. Event targets access only adjacent outertrain label pairs before expanding±8 and remaskingtrain. No target-based parameter search.')
    np.savez_compressed(cache/'boundary_events_oof.npz',prediction=out,event_probability=event_oof,
                        fold_event_probability=np.stack(probabilities),fold_boundary_bonus=np.stack(bonuses),source_fingerprint=fp)
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
