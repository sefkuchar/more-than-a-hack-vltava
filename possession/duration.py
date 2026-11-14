"""Semi-Markov decoding with durations learned from an independent public match."""

import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import log_ndtr,logsumexp

PRIOR_PATH=Path(__file__).with_name('duration_prior.json')
EMISSION_TEMPERATURE=5.


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
                prior=log_survival[k,d] if start==0 or stop==n else log_duration[k,d]
                value=exit_score[start,k]+prefix[stop,k]-prefix[start,k]+prior
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


def prior_tables(prior,length):
    d=np.maximum(np.arange(length+1,dtype=float),1)
    density,survival=[],[]
    if [g['group'] for g in prior['groups']]!=['possession','dead']:
        raise ValueError('Expected possession and dead duration priors')
    for group in prior['groups']:
        weight=np.array(group['weights']);mu=np.array(group['log_frame_means'])
        sigma=np.array(group['log_frame_sigmas'])
        if not (np.isfinite(weight).all() and np.isfinite(mu).all() and np.isfinite(sigma).all()
                and (weight>0).all() and (sigma>0).all() and np.isclose(weight.sum(),1)):
            raise ValueError('Invalid duration mixture')
        z=(np.log(d)[None,:]-mu[:,None])/sigma[:,None]
        lw=np.log(weight)[:,None]
        density.append(logsumexp(lw-np.log(d)[None,:]
            -np.log(sigma[:,None]*np.sqrt(2*np.pi))-.5*z*z,axis=0))
        survival.append(logsumexp(lw+log_ndtr(-z),axis=0))
    return np.array([density[0],density[0],density[1]]),np.array([survival[0],survival[0],survival[1]])


def decode(probability,prior=None,temperature=EMISSION_TEMPERATURE):
    p=np.asarray(probability,dtype=float)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Expected finite H/A/D probabilities')
    if not np.isfinite(temperature) or temperature<=0:
        raise ValueError('Expected a positive finite emission temperature')
    if not len(p):
        return np.empty(0,np.int8)
    prior=json.loads(PRIOR_PATH.read_text()) if prior is None else prior
    density,survival=prior_tables(prior,len(p))
    return duration_decode(np.log(p.clip(1e-5,1))/temperature,
                           density,survival,np.arange(len(p)+1))
