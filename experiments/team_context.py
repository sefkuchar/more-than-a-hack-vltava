"""One fixed team movement/shape experiment with isolated fold decoding."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features,tracking,interpolate,rolling
from possession.model import fit_model,predict
from possession.decode import viterbi
import possession_metrics as pm


def augment(x,names,players,ball,teams):
    """Team-symmetric geometry only; no labels or event-time information."""
    xy=players[:,:,:2];valid=np.isfinite(players).all(axis=2)
    b=interpolate(ball)
    delta=b[:,None,:]-xy
    distance=np.linalg.norm(delta,axis=2)
    velocity=(np.roll(xy,-3,axis=0)-np.roll(xy,3,axis=0))*(25/6)
    # Endpoints cannot use wrapped motion from the other end of the recording.
    velocity[:3]=np.nan;velocity[-3:]=np.nan
    radial=np.sum(delta*velocity,axis=2)/np.maximum(distance,1e-5)
    radial=np.clip(radial,-10,10)
    cols=[];outnames=list(names)
    def mean(value,mask):
        good=mask&np.isfinite(value)
        return np.where(good,value,0).sum(axis=1)/np.maximum(good.sum(axis=1),1)
    features={key:[] for key in ('second_distance','nearby_fraction','mean_speed',
                                'radial','near_radial','spread')}
    for team in (0,1):
        mask=valid&(teams[None,:]==team)
        ds=np.where(mask,distance,np.inf)
        second=np.partition(ds,1,axis=1)[:,1]
        features['second_distance'].append(np.where(np.isfinite(second),np.minimum(second,100),100))
        features['nearby_fraction'].append((mask&(distance<10)).sum(axis=1)/np.maximum(mask.sum(axis=1),1))
        features['mean_speed'].append(mean(players[:,:,2],mask))
        features['radial'].append(mean(radial,mask))
        features['near_radial'].append(mean(radial,mask&(distance<15)))
        count=np.maximum(mask.sum(axis=1),1)
        center=np.where(mask[:,:,None],xy,0).sum(axis=1)/count[:,None]
        square=np.sum((xy-center[:,None,:])**2,axis=2)
        features['spread'].append(np.sqrt(mean(square,mask)))
    for key,values in features.items():
        pair=np.stack(values,axis=1)
        cols.append(pair);outnames.extend([f'team_{key}_H',f'team_{key}_A'])
        if key in ('mean_speed','radial'):
            cols.append(rolling(pair,12))
            outnames.extend([f'team_{key}_mean25_H',f'team_{key}_mean25_A'])
    out=np.concatenate([x,*cols],axis=1).astype(np.float32)
    if not np.isfinite(out).all():raise ValueError('Nonfinite team context')
    return out,outnames


def fingerprint():
    h=hashlib.sha256()
    for f in ('experiments/team_context.py','possession/features.py','possession/model.py',
              'possession/decode.py','experiments/align_broadcast.py'):
        h.update((ROOT/f).read_bytes())
    for pattern in ('data/tactical/*atd.*','data/broadcast/*atd.*','data/tactical/*smart_tagging.xml',
                    'evaluation/gt/*public.csv'):
        for p in sorted(ROOT.glob(pattern)):
            s=p.stat();h.update(f'{p.relative_to(ROOT)}:{s.st_size}:{s.st_mtime_ns}'.encode())
    return h.hexdigest()


def main():
    cache=ROOT/'experiments/cache';cache.mkdir(parents=True,exist_ok=True)
    path=cache/'team_context_results.json';fp=fingerprint()
    if path.exists():
        old=json.loads(path.read_text())
        if old.get('source_fingerprint')==fp:print(json.dumps(old,indent=2));return
    x,names,_=build_features(broadcast=True);nbase=len(names)
    x,names=augment(x,names,*tracking())
    y=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(y)[0];xs=x[start:stop];y=y[start:stop]
    if len(y)!=7500:raise ValueError('Expected five public minutes')
    prediction=np.empty(len(y),np.int8);oof=np.empty((len(y),3));foldprob=[]
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500
        train=np.ones(len(y),bool);train[max(0,a-250):min(len(y),b+250)]=False
        ids=np.flatnonzero(train)[::3]
        model=fit_model(xs[ids],y[ids],names)
        p=predict(model,xs,names)
        prediction[a:b]=viterbi(np.log(p.clip(1e-5,1)),32)[a:b]
        oof[a:b]=p[a:b];foldprob.append(p)
        print('fold done',fold,flush=True)
    m=pm.evaluate(y,prediction)
    result=dict(method='team_context',base_features=nbase,total_features=len(names),
                source_fingerprint=fp,decoder_cost=32,folds=5,fold_frames=1500,
                embargo_frames=250,train_stride=3,
                metrics={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')},
                note='One fixed physical feature specification; no parameter search. Labels only train/score. Each fold model predicts/decodes whole7500 before held-out core retention. Repeated method selection on17 spells remains optimistic.')
    np.savez_compressed(cache/'team_context_oof.npz',prediction=prediction,probability=oof,
                        fold_probability=np.stack(foldprob),names=names,source_fingerprint=fp)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
