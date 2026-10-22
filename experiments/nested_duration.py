"""Train-only nested temperature selection for fixed external duration priors."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'evaluation')]
from possession.features import build_features
from possession.model import fit_model,predict
from experiments.duration_model import duration_decode
import possession_metrics as pm
TEMPERATURES=(2.,5.,10.)
TIE_ORDER=(5.,2.,10.)


def fingerprint():
    h=hashlib.sha256()
    for name in ('experiments/nested_duration.py','experiments/duration_model.py',
                 'possession/features.py','possession/model.py','experiments/align_broadcast.py'):
        h.update((ROOT/name).read_bytes())
    for pattern in ('data/tactical/*atd.*','data/broadcast/*atd.*','data/tactical/*smart_tagging.xml',
                    'evaluation/gt/*public.csv','experiments/cache/external_duration_validation.npz'):
        for p in sorted(ROOT.glob(pattern)):
            s=p.stat();h.update(f'{p.relative_to(ROOT)}:{s.st_size}:{s.st_mtime_ns}'.encode())
    return h.hexdigest()


def main():
    cache=ROOT/'experiments/cache';fp=fingerprint();resultfile=cache/'nested_duration_results.json'
    if resultfile.exists():
        report=json.loads(resultfile.read_text())
        if report.get('source_fingerprint')==fp:print(json.dumps(report,indent=2));return
    x,names,_=build_features(broadcast=True)
    gt=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start,stop=pm.valid_runs(gt)[0];x=x[start:stop];y=gt[start:stop]
    if len(y)!=7500 or len(names)!=77:raise ValueError('Expected current77 features on7500frames')
    with np.load(cache/'external_duration_validation.npz') as prior:
        density=prior['log_density'];survival=prior['log_survival']
    if density.shape!=(3,len(y)+1):raise ValueError('External prior dimensions differ')
    times=np.arange(len(y)+1)
    out=np.empty(len(y),np.int8);oof=np.empty((len(y),3));foldprob=[];outer_results=[]
    for outer in range(5):
        a,b=outer*1500,(outer+1)*1500
        outertrain=np.ones(len(y),bool)
        outertrain[max(0,a-250):min(len(y),b+250)]=False
        inner_predictions={t:np.full(len(y),3,np.int8) for t in TEMPERATURES}
        inners=[]
        for inner,core in enumerate(np.array_split(np.flatnonzero(outertrain),3)):
            lo,hi=int(core[0]),int(core[-1])+1
            innertrain=outertrain.copy()
            innertrain[max(0,lo-250):min(len(y),hi+250)]=False
            ids=np.flatnonzero(innertrain)[::3]
            record=dict(inner=inner,core_start=lo,core_stop=hi,core_frames=len(core),train_frames=len(ids))
            # These assertions are about masks only, never outer label values.
            assert not np.any(innertrain[a:b])
            assert not np.any(innertrain[core])
            if set(np.unique(y[ids]))!={0,1,2}:
                record['skipped']='missing training class';inners.append(record);continue
            model=fit_model(x[ids],y[ids],names)
            p=predict(model,x,names);logp=np.log(p.clip(1e-5,1))
            for t in TEMPERATURES:
                decoded=duration_decode(logp/t,density,survival,times)
                inner_predictions[t][core]=decoded[core]
            inners.append(record)
            print('inner done',outer,inner,flush=True)
        # Scoring reference explicitly removes outer held-out AND embargo labels.
        # Skipped cores abstain equally across temperatures and are documented.
        inner_truth=np.where(outertrain,y,3).astype(np.int8)
        scores={t:float(pm.evaluate(inner_truth,inner_predictions[t])['spell_f1_mean']) for t in TEMPERATURES}
        selected=max(TIE_ORDER,key=lambda t:scores[t])
        ids=np.flatnonzero(outertrain)[::3]
        model=fit_model(x[ids],y[ids],names)
        probability=predict(model,x,names)
        decoded=duration_decode(np.log(probability.clip(1e-5,1))/selected,density,survival,times)
        out[a:b]=decoded[a:b];oof[a:b]=probability[a:b];foldprob.append(probability)
        outer_results.append(dict(outer_fold=outer,selected_temperature=selected,
                                  inner_scores={str(t):scores[t] for t in TEMPERATURES},inner_folds=inners))
        print('outer done',outer,'temperature',selected,'inner scores',scores,flush=True)
    metrics=pm.evaluate(y,out)
    report=dict(method='nested_external_duration_temperature',source_fingerprint=fp,
                temperatures=TEMPERATURES,tie_preference=TIE_ORDER,features=len(names),
                outer_folds=outer_results,
                metrics={k:metrics[k] for k in ('spell_f1_mean','macro_f1','n_pred_spells')},
                protocol='Outer5x60s,10s embargo,stride3; inner3 contiguous chunks of available outertrain indices; whole inner span plus10s removed. No outer labels in temperature selection or duration priors. Full-context decoding always uses same label-excluded model. Repeated public method development still causes selection optimism.')
    np.savez_compressed(cache/'nested_duration_oof.npz',prediction=out,probability=oof,
                        fold_probability=np.stack(foldprob),source_fingerprint=fp)
    resultfile.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
