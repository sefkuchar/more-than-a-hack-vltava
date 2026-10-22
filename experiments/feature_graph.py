"""One fixed transductive physical-feature graph; no clock or frame-index edges."""
import hashlib
import json
from pathlib import Path
import sys
import warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.semi_supervised import LabelSpreading
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
from possession.model import swap_teams
from possession.duration import decode,PRIOR_PATH
from experiments.physical_boundary_snap import control_onsets,snap
from games import half_video_blocks
from evaluation import possession_metrics as pm


def safe_probability(p):
    p=np.nan_to_num(p,nan=0.,posinf=0.,neginf=0.).clip(0)
    total=p.sum(axis=1,keepdims=True)
    return np.divide(p,total,out=np.full_like(p,1/3),where=total>0)


def main():
    x,names,raw=build_features(broadcast=True);start,stop=11185,18685
    public=np.arange(start,stop,3)
    unlabeled=np.concatenate([np.arange(lo,hi+1,25) for _,lo,hi in half_video_blocks('BAR-ATH')])
    ids=np.union1d(public,unlabeled)
    nx=x[ids];mx=swap_teams(nx,names)
    # Standardization sees physical observations only, never node labels.
    scaler=StandardScaler().fit(np.vstack([nx,mx]))
    nodes=scaler.transform(np.vstack([nx,mx]))
    query=scaler.transform(x[start:stop]);mirror_query=scaler.transform(swap_teams(x[start:stop],names))
    stable=x[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    onsets=control_onsets(raw['nearest'],stable,raw['observed'])
    prov=json.loads((ROOT/'experiments/booster_ensemble_results.json').read_text())
    featurehash=hashlib.sha256(x[start:stop].tobytes()).hexdigest()
    assert featurehash==prov['feature_sha256']
    assert hashlib.sha256((ROOT/'experiments/booster_ensemble.py').read_bytes()).hexdigest()==prov['source_sha256']
    cache=ROOT/'experiments/cache/booster_ensemble.npz'
    with np.load(cache) as saved:base=saved['hist_fold_probability']
    # Input graph and queries are fixed; only allowed training nodes get labels.
    y=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[start:stop]
    assert hashlib.sha256(y.tobytes()).hexdigest()==prov['label_sha256']
    methods=('graph','graph_snap','equal_mixture','equal_mixture_snap')
    outputs={name:np.empty(7500,np.int8) for name in methods};folds=[];contexts=[]
    public_node=np.isin(ids,public)
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500
        train=np.ones(7500,bool);train[max(0,a-250):min(7500,b+250)]=False
        eligible=public_node&(ids>=start)&(ids<stop)
        selected=np.flatnonzero(eligible)
        selected=selected[train[ids[selected]-start]]
        labels=np.full(len(ids),-1,dtype=int)
        labels[selected]=y[ids[selected]-start]
        assert not np.any((labels>=0)&(ids>=start+a)&(ids<start+b))
        mirrored_labels=np.where(labels<0,-1,np.where(labels<2,1-labels,labels))
        model=LabelSpreading(kernel='knn',n_neighbors=15,alpha=.2,max_iter=100,tol=1e-3,n_jobs=4)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            model.fit(nodes,np.r_[labels,mirrored_labels])
        assert np.array_equal(model.classes_,[0,1,2])
        p=(safe_probability(model.predict_proba(query))+safe_probability(model.predict_proba(mirror_query))[:,[1,0,2]])/2
        p=safe_probability(p);contexts.append(p)
        mixture=(p+base[fold])/2
        a_plain=decode(p);a_snap,changes=snap(a_plain,onsets,start)
        b_plain=decode(mixture);b_snap,mchanges=snap(b_plain,onsets,start)
        for name,v in zip(methods,(a_plain,a_snap,b_plain,b_snap)):outputs[name][a:b]=v[a:b]
        graphmass=np.nan_to_num(model.label_distributions_).sum(axis=1)
        folds.append(dict(fold=fold,labeled_nodes=int(len(selected)*2),iterations=int(model.n_iter_),
            unreachable_nodes=int((graphmass<=0).sum()),warnings=[str(w.message) for w in caught],
            graph_snap_changes_full_context=len(changes),mixture_snap_changes_full_context=len(mchanges)))
        print('featuregraph fold',fold,'iterations',model.n_iter_,flush=True)
    results={}
    for name,v in outputs.items():
        m=pm.evaluate(y,v);results[name]={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}
    report=dict(results=results,folds=folds,original_nodes=len(ids),mirrored_total_nodes=len(nodes),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),feature_sha256=featurehash,
        prior_sha256=hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),base_cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        parameters=dict(kernel='knn',n_neighbors=15,alpha=.2,max_iter=100,tol=.001,n_jobs=4),
        protocol='TransductiveLabelSpreading. Nodesunionpublicstride3+publishedhalvesstride25 andteam-mirrors; StandardScaler fittedunlabeledphysicalfeaturesonly. Onlypublictrainstride3 nodesoutsideouterminute+10sembargo carrylabels. Queryphysicalfeatures77 only,no time/frameindexedges. Puregraph/equalHGBmixture predeclared, eachwithwithoutunchangedphysicalsnap; samefoldfullcontext durationT5beforecoreretention. Uniformfallbackforunreachablequeryprobability. Repeated17spellmethodselectionremainsoptimistic.')
    np.savez_compressed(ROOT/'experiments/cache/feature_graph.npz',fold_probability=np.stack(contexts),node_frames=ids,**outputs)
    (ROOT/'experiments/feature_graph_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
