"""Nested train-only calibration of existing dead-ball edge timing."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
from possession.model import fit_model,predict
from possession.duration import decode,PRIOR_PATH
from experiments.physical_boundary_snap import control_onsets,snap
from evaluation import possession_metrics as pm


def offsets(y,prediction,outertrain,first,last):
    candidate=np.arange(max(1,first),last)
    candidate=candidate[outertrain[candidate-1]&outertrain[candidate]]
    def changes(values):
        before=(values[candidate-1]==2).astype(int);after=(values[candidate]==2).astype(int)
        keep=before!=after
        return list(zip(candidate[keep].tolist(),before[keep].tolist(),after[keep].tolist()))
    pairs,_,_=pm.match_transitions(changes(y),changes(prediction),125,typed='pair')
    result={'into_dead':[],'out_of_dead':[]}
    for truth,pred in pairs:result['into_dead' if truth[2] else 'out_of_dead'].append(int(truth[0]-pred[0]))
    return result


def move_dead_edges(prediction,bias):
    out=prediction.copy();edges=np.r_[0,np.flatnonzero(out[1:]!=out[:-1])+1,len(out)]
    labels=out[edges[:-1]].copy();pure_ha=[int(edges[i]) for i in range(1,len(edges)-1) if labels[i-1]<2 and labels[i]<2]
    changes=0;skipped=0
    for i in range(1,len(edges)-1):
        if labels[i-1]!=2 and labels[i]!=2:continue
        delta=bias['into_dead' if labels[i]==2 else 'out_of_dead']
        new=int(edges[i]+delta)
        if not edges[i-1]<new<edges[i+1]:skipped+=1;continue
        changes+=int(delta!=0);edges[i]=new
    for i,label in enumerate(labels):out[edges[i]:edges[i+1]]=label
    newedges=np.r_[0,np.flatnonzero(out[1:]!=out[:-1])+1,len(out)]
    assert len(edges)==len(newedges) and np.all(np.diff(newedges)>0)
    assert np.array_equal(out[newedges[:-1]],labels)
    assert pure_ha==[int(newedges[i]) for i in range(1,len(newedges)-1) if labels[i-1]<2 and labels[i]<2]
    return out,changes,skipped


def selfcheck():
    y=np.array([0,0,2,2,0,0,2,2,0,0,2,2,0,0,2,2,0,0])
    train=np.ones(len(y),bool);train[5:11]=False
    pred=np.roll(y,1)
    original=offsets(y,pred,train,0,len(y))
    mutated=y.copy();mutated[~train]=1-mutated[~train]
    assert original==offsets(mutated,pred,train,0,len(y)),'Excluded labels leaked into timing calibration'
    moved,_,_=move_dead_edges(y,dict(into_dead=1,out_of_dead=-1))
    assert len(moved)==len(y)


def summary(values):
    return dict(count=len(values),median=float(np.median(values)) if values else None,
                mean=float(np.mean(values)) if values else None,
                minimum=int(min(values)) if values else None,maximum=int(max(values)) if values else None)


def main():
    selfcheck()
    x,names,raw=build_features(broadcast=True);start,stop=11185,18685;xs=x[start:stop]
    provenance=json.loads((ROOT/'experiments/booster_ensemble_results.json').read_text())
    featurehash=hashlib.sha256(xs.tobytes()).hexdigest();assert featurehash==provenance['feature_sha256']
    assert hashlib.sha256((ROOT/'experiments/booster_ensemble.py').read_bytes()).hexdigest()==provenance['source_sha256']
    stable=x[:,[names.index('stable_contact_H'),names.index('stable_contact_A')]]
    onsets=control_onsets(raw['nearest'],stable,raw['observed'])
    cache=ROOT/'experiments/cache/booster_ensemble.npz'
    with np.load(cache) as z:baseprob=z['hist_fold_probability']
    y=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[start:stop]
    out=np.empty(7500,np.int8);baseline=np.empty_like(out);contexts=[];records=[]
    for outer in range(5):
        a,b=outer*1500,(outer+1)*1500
        outertrain=np.ones(7500,bool);outertrain[max(0,a-250):min(7500,b+250)]=False
        samples={'into_dead':[],'out_of_dead':[]};inners=[]
        for inner in range(5):
            if inner==outer:continue
            c,d=inner*1500,(inner+1)*1500
            train=outertrain.copy();train[max(0,c-250):min(7500,d+250)]=False
            ids=np.flatnonzero(train)[::3]
            assert not train[a:b].any() and not train[c:d].any()
            if set(np.unique(y[ids]))!={0,1,2}:
                inners.append(dict(inner_fold=inner,skipped='missing training class'));continue
            model=fit_model(xs[ids],y[ids],names)
            pred,_=snap(decode(predict(model,xs,names)),onsets,start)
            found=offsets(y,pred,outertrain,c,d)
            # A full excluded-label perturbation must leave calibration unchanged.
            changed=y.copy();changed[~outertrain]=(changed[~outertrain]+1)%3
            assert found==offsets(changed,pred,outertrain,c,d)
            for key in samples:samples[key].extend(found[key])
            inners.append(dict(inner_fold=inner,train_frames=len(ids),matches={k:len(v) for k,v in found.items()}))
        bias={key:int(np.clip(np.rint(np.median(values)),-50,50)) if len(values)>=2 else 0 for key,values in samples.items()}
        outerbase,_=snap(decode(baseprob[outer]),onsets,start)
        result,count,skipped=move_dead_edges(outerbase,bias)
        baseline[a:b]=outerbase[a:b];out[a:b]=result[a:b];contexts.append(result)
        records.append(dict(outer_fold=outer,bias_frames=bias,offset_statistics={k:summary(v) for k,v in samples.items()},
                            inner_folds=inners,moved_dead_edges_full_context=count,skipped_crossing_edges=skipped))
        print('nestedD fold',outer,bias,{k:len(v) for k,v in samples.items()},flush=True)
    # Outer metrics are calculated only after every fold's biases/output are fixed.
    scores={}
    for name,pred in [('physical_snap_baseline',baseline),('nested_dead_calibration',out)]:
        m=pm.evaluate(y,pred);scores[name]={k:m[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')}
    report=dict(results=scores,folds=records,changed_oof_frames=int(np.sum(out!=baseline)),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),feature_sha256=featurehash,
        prior_sha256=hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
        protocol='Outer5foldminute+250framepurge. Fourotherminutesinnerheldout; HGBtrain=outertrain intersectnotinner±250, stride3. Sameinnerexcludedmodel fullcontextduration5+fixedHA snap. BinaryD transitions onlywithbothneighborsoutertrain,withininnercore;typedpairmatch125frames. Biasroundedmediantrue-minus-pred if>=2matches clipped±50. MovesonlyexistingD edges,orderedpositive sequence preserved andpureHAedgesunchanged. Excludedlabelperturbation selfcheckpassed. No offsetgrid or manualtimestamps.')
    np.savez_compressed(ROOT/'experiments/cache/dead_boundary_calibration.npz',prediction=out,baseline=baseline,fold_prediction=np.stack(contexts))
    (ROOT/'experiments/dead_boundary_calibration_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
