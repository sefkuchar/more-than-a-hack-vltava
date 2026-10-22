"""Learn a soft duration distribution from training spells, without a minimum length."""

import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import log_ndtr

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
import possession_metrics as pm


@njit(cache=True)
def duration_decode(evidence,log_duration,log_survival,times,boundary=None):
    n=len(evidence)
    prefix=np.zeros((n+1,3))
    for t in range(n):
        prefix[t+1]=prefix[t]+evidence[t]
    score=np.full((n+1,3),-np.inf)
    exit_score=np.full((n+1,3),-np.inf)
    exit_state=np.zeros((n+1,3),np.int8)
    back_start=np.zeros((n+1,3),np.int32)
    back_state=np.zeros((n+1,3),np.int8)
    exit_score[0]=0.
    for stop in range(1,n+1):
        for k in range(3):
            best=-np.inf
            for start in range(stop):
                d=times[stop]-times[start]
                duration=log_survival[k,d] if start==0 or stop==n else log_duration[k,d]
                value=exit_score[start,k]+prefix[stop,k]-prefix[start,k]+duration
                if boundary is not None and start>0:
                    value+=boundary[times[start]]
                if value>best:
                    best=value;back_start[stop,k]=start;back_state[stop,k]=exit_state[start,k]
            score[stop,k]=best
        for k in range(3):
            for prev in range(3):
                if prev!=k and score[stop,prev]-np.log(2.)>exit_score[stop,k]:
                    exit_score[stop,k]=score[stop,prev]-np.log(2.)
                    exit_state[stop,k]=prev
    labels=np.empty(n,np.int8)
    k=np.argmax(score[n]);stop=n
    while stop>0:
        start=back_start[stop,k]
        labels[start:stop]=k
        k=back_state[stop,k];stop=start
    return labels


def check_decoder():
    rng=np.random.RandomState(7)
    for n in (1,3,5):
        evidence=rng.normal(size=(n,3))
        density=-rng.uniform(size=(3,n+1))
        survival=-rng.uniform(size=(3,n+1))
        def objective(seq):
            seq=np.array(seq)
            bounds=np.r_[0,np.flatnonzero(seq[1:]!=seq[:-1])+1,n]
            score=sum(evidence[t,k] for t,k in enumerate(seq))-np.log(2)*(len(bounds)-2)
            for a,b in zip(bounds[:-1],bounds[1:]):
                prior=survival if a==0 or b==n else density
                score+=prior[seq[a],b-a]
            return score
        actual=objective(duration_decode(evidence,density,survival,np.arange(n+1)))
        expected=max(objective(seq) for seq in itertools.product(range(3),repeat=n))
        assert np.isclose(actual,expected),(n,actual,expected)
        boundary=rng.normal(size=n+1)
        def with_boundary(seq):
            changes=np.flatnonzero(np.array(seq)[1:]!=np.array(seq)[:-1])+1
            return objective(seq)+sum(boundary[changes])
        actual=with_boundary(duration_decode(evidence,density,survival,np.arange(n+1),boundary))
        expected=max(with_boundary(seq) for seq in itertools.product(range(3),repeat=n))
        assert np.isclose(actual,expected),(n,actual,expected)


def decode_on_grid(evidence,density,survival,step=5):
    times=np.r_[np.arange(0,len(evidence),step),len(evidence)]
    cumulative=np.vstack([np.zeros(3),evidence.cumsum(axis=0)])
    blocks=np.diff(cumulative[times],axis=0)
    labels=duration_decode(blocks,density,survival,times)
    return np.repeat(labels,np.diff(times))


def refine_boundaries(labels,evidence,density,survival,radius=4):
    bounds=np.r_[0,np.flatnonzero(labels[1:]!=labels[:-1])+1,len(labels)]
    states=labels[bounds[:-1]]
    cumulative=np.vstack([np.zeros(3),evidence.cumsum(axis=0)])
    for _ in range(2):
        for j in range(1,len(bounds)-1):
            a,b,c=bounds[j-1:j+2]
            candidates=np.arange(max(a+1,b-radius),min(c-1,b+radius)+1)
            left,right=states[j-1:j+1]
            lp=survival if a==0 else density
            rp=survival if c==len(labels) else density
            scores=(cumulative[candidates,left]-cumulative[a,left]
                +cumulative[c,right]-cumulative[candidates,right]
                +lp[left,candidates-a]+rp[right,c-candidates])
            bounds[j]=candidates[np.argmax(scores)]
    out=np.empty_like(labels)
    for a,b,k in zip(bounds[:-1],bounds[1:],states):
        out[a:b]=k
    return out


def duration_priors(y,train):
    masked=np.where(train,y,3).astype(np.int8)
    samples=[[],[]]
    for start,stop,k in pm.segments(masked,pm.valid_runs(masked)):
        # A training-mask edge censors a spell and cannot supply its full duration.
        if start>0 and stop<len(y) and train[start-1] and train[stop]:
            samples[int(k==2)].append(stop-start)
    d=np.arange(len(y)+1,dtype=float);d[0]=1
    log_pdf=np.empty((3,len(d)));log_survival=np.empty_like(log_pdf)
    parameters=[]
    for k in range(3):
        values=np.log(samples[int(k==2)])
        mu=float(values.mean()) if len(values) else np.log(375.)
        sigma=max(float(values.std()) if len(values)>1 else 1.,.75)
        z=(np.log(d)-mu)/sigma
        log_pdf[k]=-np.log(d*sigma*np.sqrt(2*np.pi))-.5*z*z
        log_survival[k]=log_ndtr(-z)
        parameters.append(dict(state=k,mu=mu,sigma=sigma,n_training_spells=len(values)))
    return log_pdf,log_survival,parameters


if __name__=='__main__':
    check_decoder()
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(gt)[0];y=gt[start:stop]
    with np.load(ROOT/'experiments/cache/booster_ensemble.npz') as z:
        probability=z['hist_fold_probability']
    with np.load(ROOT/'experiments/cache/xgboost_validation.npz') as z:
        xp=z['fold_probability']
    methods={'hist':probability,'xgb':xp,'hist_xgb':(probability+xp)/2}
    predictions={m:np.empty(len(y),np.int8) for m in methods}
    predictions['hist_grid5']=np.empty(len(y),np.int8)
    predictions['hist_grid5_refined']=np.empty(len(y),np.int8);parameters=[]
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500;train=np.ones(len(y),bool)
        train[max(0,a-250):min(len(y),b+250)]=False
        density,survival,params=duration_priors(y,train)
        # Five correlated frames contribute one unit of emission evidence.
        for method,p in methods.items():
            result=duration_decode(np.log(p[fold].clip(1e-5,1))/5,density,survival,np.arange(len(y)+1))
            predictions[method][a:b]=result[a:b]
        result=decode_on_grid(np.log(probability[fold].clip(1e-5,1))/5,density,survival)
        predictions['hist_grid5'][a:b]=result[a:b]
        result=refine_boundaries(result,np.log(probability[fold].clip(1e-5,1))/5,density,survival)
        predictions['hist_grid5_refined'][a:b]=result[a:b]
        parameters.append(params)
        print('duration isolated fold',fold,flush=True)
    results=[]
    for method,pred in predictions.items():
        score=pm.evaluate(y,pred)
        row=dict(method=method,**{k:score[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')})
        results.append(row);print(row,flush=True)
    report=dict(results=results,parameters=parameters,emission_temperature=5,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    np.savez_compressed(ROOT/'experiments/cache/duration_validation.npz',**predictions)
    (ROOT/'experiments/duration_results.json').write_text(json.dumps(report,indent=2)+'\n')
