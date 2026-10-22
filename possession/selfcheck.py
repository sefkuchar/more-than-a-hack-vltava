"""Small independent checks for the decoder optimum and team equivariance."""

import itertools

import numpy as np

from possession.decode import viterbi
from possession.duration import duration_decode,decode
from possession.control import control_onsets,snap_control_boundaries
from possession.model import swap_teams,fit_model,predict
from possession.physics import selfcheck as physics_selfcheck
from possession.calibration import blend_teams


def main():
    physics_selfcheck()
    rng=np.random.RandomState(2)
    for n in (1,2,5):
        for cost in (0,1.7):
            for dead in (.5,3):
                e=rng.normal(size=(n,3))
                c=np.full((3,3),cost,dtype=float)
                c[2,:]=dead;c[:,2]=dead;np.fill_diagonal(c,0)
                def objective(seq):
                    return sum(e[t,k] for t,k in enumerate(seq))-sum(c[seq[t-1],seq[t]] for t in range(1,n))
                best=max(objective(s) for s in itertools.product(range(3),repeat=n))
                assert np.isclose(objective(viterbi(e,cost,dead)),best)
    assert len(viterbi(np.empty((0,3))))==0
    for n in (1,3,5):
        evidence=rng.normal(size=(n,3))
        density=-rng.uniform(size=(3,n+1))
        survival=-rng.uniform(size=(3,n+1))
        boundary=rng.normal(size=n+1)
        def duration_objective(seq):
            seq=np.array(seq)
            bounds=np.r_[0,np.flatnonzero(seq[1:]!=seq[:-1])+1,n]
            score=sum(evidence[t,k] for t,k in enumerate(seq))-np.log(2)*(len(bounds)-2)
            for a,b in zip(bounds[:-1],bounds[1:]):
                prior=survival if a==0 or b==n else density
                score+=prior[seq[a],b-a]
                if a>0:
                    score+=boundary[a]
            return score
        actual=duration_decode(evidence,density,survival,np.arange(n+1),boundary)
        expected=max(duration_objective(s) for s in itertools.product(range(3),repeat=n))
        assert np.isclose(duration_objective(actual),expected)
    assert len(decode(np.empty((0,3))))==0
    probability=rng.dirichlet(np.ones(3),6)
    assert np.array_equal(decode(probability),decode(probability,temperature=5.))
    decoded=decode(probability,temperature=2.)
    mirrored=decode(probability[:,[1,0,2]],temperature=2.)
    assert np.array_equal(mirrored,np.where(decoded<2,1-decoded,decoded))
    for invalid in (0.,-1.,np.inf,np.nan):
        try:
            decode(probability,temperature=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid temperature accepted')
    nearest=np.full((40,2),10.)
    stable=np.zeros((40,2));observed=np.ones(40,bool)
    nearest[10:20,0]=.5;stable[10:20,0]=.8
    onsets=control_onsets(nearest,stable,observed)
    assert np.array_equal(onsets[0],[10]) and not len(onsets[1])
    impulse_stability=np.zeros_like(stable)
    acceleration=np.zeros(40);acceleration[12]=8
    assert not control_onsets(nearest,impulse_stability,observed)[0].size
    assert np.array_equal(control_onsets(nearest,impulse_stability,observed,acceleration)[0],[10])
    assert not control_onsets(nearest,impulse_stability,np.zeros(40,bool),acceleration)[0].size
    original=np.r_[np.ones(15,np.int8),np.zeros(15,np.int8),np.full(10,2,np.int8)]
    refined,changes=snap_control_boundaries(original,onsets,0)
    assert changes==[(15,10)] and np.array_equal(refined,np.r_[np.ones(10),np.zeros(20),np.full(10,2)])
    swapped=np.where(original<2,1-original,original)
    mirror,_=snap_control_boundaries(swapped,onsets[::-1],0)
    assert np.array_equal(mirror,np.where(refined<2,1-refined,refined))
    assert not len(snap_control_boundaries(np.empty(0,np.int8),onsets,0)[0])
    names=['contact_H','contact_A','distance_margin','motion']
    x=rng.normal(size=(150,4));y=np.tile(np.arange(3),50)
    assert np.array_equal(swap_teams(swap_teams(x,names),names),x)
    m=fit_model(x,y,names)
    p=predict(m,x,names)
    q=predict(m,swap_teams(x,names),names)[:,[1,0,2]]
    assert np.allclose(p,q)
    assert np.allclose(p.sum(axis=1),1)
    other=rng.dirichlet(np.ones(3),len(p))
    blended=blend_teams(p,other)
    assert np.array_equal(blended[:,2],p[:,2]) and np.allclose(blended.sum(axis=1),1)
    assert np.allclose(blend_teams(p,p),p)
    assert np.allclose(blend_teams(p[:,[1,0,2]],other[:,[1,0,2]]),blended[:,[1,0,2]])
    assert np.allclose(blend_teams(np.array([[.2,.8,0.],[0.,0.,1.]]),np.array([[0.,0.,1.],[.1,.9,0.]])).sum(axis=1),1)
    assert blend_teams(np.empty((0,3)),np.empty((0,3))).shape==(0,3)
    print('PASS: decoder optima, observed/gap-safe physics, sustained/impulse contact refinement, conditional team blending, team symmetry and empty inputs')


if __name__=='__main__':
    main()
