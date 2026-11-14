"""Fixed contact recency experiment with strictly isolated temporal-fold decoding."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
from possession.model import fit_model,predict
from possession.decode import viterbi
import possession_metrics as pm


def contact_ages(contact, fps=25, cap_seconds=20):
    """Past/future contact distance in seconds; missing contacts saturate at cap."""
    n=len(contact);i=np.arange(n)[:,None];cap=int(fps*cap_seconds)
    previous=np.maximum.accumulate(np.where(contact,i,-cap),axis=0)
    following=np.minimum.accumulate(np.where(contact,i,n+cap)[::-1],axis=0)[::-1]
    return np.minimum(i-previous,cap)/fps,np.minimum(following-i,cap)/fps


def augment(x,names,raw):
    extra=[];outnames=list(names)
    observed=raw['observed'][:,None]
    signals={'raw':(raw['contact']>np.exp(-.5))&observed,
             'stable':(x[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]>.4)&observed}
    for signal,contact in signals.items():
        for direction,age in zip(('since','until'),contact_ages(contact)):
            extra.append(age)
            outnames.extend([f'control_{signal}_{direction}_H',f'control_{signal}_{direction}_A'])
    return np.concatenate([x,*extra],axis=1).astype(np.float32),outnames


def fingerprint():
    source=['experiments/control_memory.py','possession/features.py','possession/model.py',
            'possession/decode.py','experiments/align_broadcast.py']
    h=hashlib.sha256()
    for file in source:h.update((ROOT/file).read_bytes())
    for pattern in ('data/tactical/*atd.*','data/broadcast/*atd.*','data/tactical/*smart_tagging.xml',
                    'evaluation/gt/*public.csv'):
        for p in sorted(ROOT.glob(pattern)):
            s=p.stat();h.update(f'{p.relative_to(ROOT)}:{s.st_size}:{s.st_mtime_ns}'.encode())
    return h.hexdigest()


def main():
    cache=ROOT/'experiments/cache';cache.mkdir(exist_ok=True,parents=True)
    result_path=cache/'control_memory_results.json';fp=fingerprint()
    if result_path.exists():
        old=json.loads(result_path.read_text())
        if old.get('source_fingerprint')==fp:
            print(json.dumps(old,indent=2));return
    x,names,raw=build_features(broadcast=True)
    original=len(names);x,names=augment(x,names,raw)
    truth=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(truth)[0];xs=x[start:stop];y=truth[start:stop]
    if len(y)!=7500:raise ValueError('Expected five public minutes')
    out=np.empty(len(y),np.int8);oof=np.empty((len(y),3));folds=[]
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500
        train=np.ones(len(y),bool);train[max(0,a-250):min(len(y),b+250)]=False
        ids=np.flatnonzero(train)[::3]
        model=fit_model(xs[ids],y[ids],names)
        probability=predict(model,xs,names)
        decoded=viterbi(np.log(probability.clip(1e-5,1)),32)
        out[a:b]=decoded[a:b];oof[a:b]=probability[a:b]
        folds.append(dict(fold=fold,train_frames=len(ids),heldout_frames=b-a))
        print('fold done',fold,flush=True)
    m=pm.evaluate(y,out)
    result=dict(method='control_memory',base_features=original,total_features=len(names),
                source_fingerprint=fp,decoder_cost=32,folds=folds,
                metrics={k:m[k] for k in ['spell_f1_mean','macro_f1','n_pred_spells']},
                note='One fixed feature experiment. Five 60s folds,10s label embargo; each same-fold model decodes whole7500 then retains held-out core. Public17 spells cannot establish generalisation; repeated method selection is optimistic.')
    np.savez_compressed(cache/'control_memory_oof.npz',prediction=out,probability=oof,names=names,source_fingerprint=fp)
    result_path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
