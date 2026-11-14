"""Official PathCRF140 evidence on automatically stitched local tracks.

Upstream MPL2 files/weights remain under ignored data/external/pathcrf.
No target labels enter stitching, input construction or pretrained inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import types
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from possession.features import tracking,build_features
from possession.model import fit_model,predict
from possession.duration import decode
from evaluation import possession_metrics as pm
VENDOR=ROOT/'data/external/pathcrf'
CACHE=ROOT/'experiments/cache'


def load_model(device):
    # Only one small index helper is needed from upstream's PyG-dependent utils.
    # Supply our own indexing implementation; no PyG or copied upstream helper.
    for name in ('models','datatools','set_transformer'):
        package=types.ModuleType(name);package.__path__=[str(VENDOR/name)]
        sys.modules[name]=package
    helpers=types.ModuleType('models.utils')
    def maps(n_players=22,n_outsides=4,device=None):
        n=n_players+n_outsides
        ids=[s*n+d for s in range(n_players) for d in range(n)]
        ids.extend(o*n+o for o in range(n_players,n))
        ids=torch.tensor(ids,dtype=torch.long,device=device)
        lookup=torch.full((n*n,),-1,dtype=torch.long,device=device)
        lookup[ids]=torch.arange(len(ids),device=device)
        return ids,lookup,ids//n,ids%n
    helpers.build_edge_compression_maps=maps;sys.modules['models.utils']=helpers
    from models.set_lstm import SetLSTM
    args=json.loads((VENDOR/'saved/140/args.json').read_text())
    model=SetLSTM(macro_type=args['macro_type'],micro_type=args['micro_type'],
        seq_model_type=args['seq_model'],crf_model_type=args['crf_model'],
        feat_dim=args['node_in_dim'],team_size=args['team_size'],
        macro_ppe_dim=args['coarse_dim'],macro_fpe_dim=args['coarse_dim'],
        micro_enc_dim=args['fine_dim'],macro_seq_dim=args['seq_dim'],
        micro_seq_dim=args['seq_dim'],crf_edge_dim=args['crf_edge_dim'],dropout=args['dropout'])
    path=VENDOR/'saved/140/model/state_dict_best_acc.pt'
    blob=path.read_bytes()
    digest=hashlib.sha1(f'blob {len(blob)}\0'.encode()+blob).hexdigest()
    if digest!='f8e48952d01d6d70db986948962822478884a336':raise ValueError('Official checkpoint blob mismatch')
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
    return model.to(device).eval(),hashlib.sha256(blob).hexdigest()


def deduplicate(points):
    kept=[]
    for point in points:
        if not kept or np.linalg.norm(np.asarray(kept)-point,axis=1).min()>=.7:kept.append(point)
    return np.asarray(kept).reshape(-1,2)


def choose_players(points,count):
    if len(points)<=count:return points
    # Geometric coverage selection avoids arbitrary excess-slot truncation.
    ids=[int(np.argmin(points[:,0]))]
    while len(ids)<count:
        ds=np.linalg.norm(points[:,None]-points[ids][None],axis=2).min(1)
        ds[ids]=-1;ids.append(int(ds.argmax()))
    return points[ids]


def stitch_team(observations,diagnostics=None,count=11,max_gap=5):
    counts=np.array([len(p) for p in observations])
    if counts.max()<count:
        if diagnostics is not None:diagnostics['insufficient_seed_observations']=diagnostics.get('insufficient_seed_observations',0)+1
        return None
    seed=int(np.flatnonzero(counts>=count)[0]);n=len(observations)
    xy=np.full((n,count,2),np.nan)
    xy[seed]=choose_players(observations[seed],count)
    for direction in (1,-1):
        last=xy[seed].copy();velocity=np.zeros((count,2));age=np.zeros(count,int)
        for frame in range(seed+direction,n if direction==1 else -1,direction):
            candidate=observations[frame]
            predicted=last+velocity*.2*direction
            age+=1
            if len(candidate):
                distance=np.linalg.norm(predicted[:,None]-candidate[None],axis=2)
                rows,cols=linear_sum_assignment(distance)
                for r,c in zip(rows,cols):
                    if distance[r,c]<=4.:
                        xy[frame,r]=candidate[c]
                        speed=(candidate[c]-last[r])/(.2*max(1,age[r]))*direction
                        velocity[r]=speed if np.linalg.norm(speed)<12 else 0
                        last[r]=candidate[c];age[r]=0
            # Carry prediction for assignment only; missing coordinates stay NaN.
            last[age>0]=predicted[age>0]
    quality=np.isfinite(xy).all(2).mean()
    for player in range(count):
        good=np.isfinite(xy[:,player]).all(1);ids=np.flatnonzero(good)
        edges=np.r_[0,np.flatnonzero(good[1:]!=good[:-1])+1,n]
        for a,b in zip(edges[:-1],edges[1:]):
            if good[a]:continue
            if b-a>max_gap:
                reason='long_interior_gap' if a>0 and b<n else 'long_edge_gap'
                if diagnostics is not None:diagnostics[reason]=diagnostics.get(reason,0)+1
                return None
        for axis in (0,1):xy[:,player,axis]=np.interp(np.arange(n),ids,xy[ids,player,axis])
    return xy,quality


def window_input(players,teams,left_team,diagnostics=None):
    real=np.isfinite(players).all(2)
    clouds=[];quality=[];keeper_missing=[];failed=False
    for team in (left_team,1-left_team):
        observations=[deduplicate(frame[mask&(teams==team),:2])
                      for frame,mask in zip(players,real)]
        result=stitch_team(observations,diagnostics,count=10,max_gap=25)
        if result is None:
            failed=True;continue
        xy,q=result
        # NaN-speed keeper candidates are uncertain context, never raw contacts.
        # A wrong-goal placeholder is rejected by geometry, not possession labels.
        goal_x=5. if team==left_team else 100.3
        keeper=np.full((len(players),2),np.nan)
        for frame in range(len(players)):
            valid=np.isfinite(players[frame,:,:2]).all(1)&~np.isfinite(players[frame,:,2])&(teams==team)
            valid&=np.abs(players[frame,:,0]-goal_x)<25
            ids=np.flatnonzero(valid)
            if len(ids):
                distances=np.linalg.norm(players[frame,ids,:2]-[goal_x,34],axis=1)
                keeper[frame]=players[frame,ids[distances.argmin()],:2]
        available=np.isfinite(keeper).all(1)
        keeper_missing.append(float((~available).mean()))
        if available.any():
            for axis in (0,1):keeper[:,axis]=np.interp(np.arange(len(keeper)),np.flatnonzero(available),keeper[available,axis])
        else:keeper[:]=[goal_x,34]
        clouds.append(np.concatenate([xy,keeper[:,None]],axis=1));quality.append(q*10/11)
    if failed:return None
    xy=np.concatenate(clouds,axis=1);xy[:,:,0]*=105/105.3
    velocity=np.gradient(xy,.2,axis=0);speed=np.linalg.norm(velocity,axis=2)
    accel=np.gradient(speed,.2,axis=0)
    indicators=np.zeros((len(xy),22,2));indicators[:,:11,0]=-1;indicators[:,11:,0]=1
    keepers=(10,21)
    indicators[:,keepers,1]=1
    feat=np.concatenate([indicators,xy,velocity,speed[:,:,None],accel[:,:,None]],axis=2)
    outside=np.zeros((len(xy),4,8));outside[:,:,2:4]=[[0,34],[105,34],[52.5,0],[52.5,68]]
    out=np.concatenate([feat,outside],axis=1).astype(np.float32)
    assert out.shape==(50,26,8) and np.isfinite(out).all()
    return out,float(np.mean(quality)),float(np.mean(keeper_missing))


def evidence(device):
    model,checkpoint_sha=load_model(device)
    players,ball,teams=tracking()
    # Published first-half start and public five-minute input extent only.
    first,last=11185,18685
    starts=np.unique(np.r_[np.arange(first,last-250+1,125),last-250])
    # Infer side from collective first-window geometry, independent of labels.
    real=np.isfinite(players[first:first+250]).all(2)
    mean=[np.nanmean(np.where(real[:,teams==k,None],players[first:first+250,teams==k,:2],np.nan)[:,:,0]) for k in (0,1)]
    left=int(np.argmin(mean))
    total=np.zeros((7500,10));weight=np.zeros(7500);accepted=[];diagnostics={}
    taper=np.maximum(.1,1-np.abs(np.linspace(-1,1,50)))
    with torch.inference_mode():
        for start in starts:
            ids=start+np.arange(50)*5
            result=window_input(players[ids],teams,left,diagnostics)
            if result is None:continue
            array,quality,keeper_missing=result
            macro,edges,embedded=model(torch.tensor(array[None],device=device))
            valid=model.valid_orig_edge_ids
            emissions=edges.index_select(2,valid)
            embedded=embedded.index_select(2,valid)
            path=model.crf.decode(emissions,embedded)[0]
            src=model.comp_src[path].cpu().numpy()
            probs=torch.softmax(edges[0],dim=1).reshape(50,26,26).sum(2).cpu().numpy()
            coarse=torch.softmax(macro[0],dim=1).cpu().numpy()
            pairs=[]
            for side,team in enumerate((left,1-left)):
                lo,hi=side*11,(side+1)*11
                pairs.append((team,np.column_stack([probs[:,lo:hi].sum(1),
                    coarse[:,lo:hi,0].sum(1),coarse[:,lo:hi,1].sum(1),((src>=lo)&(src<hi)).astype(float)])))
            f=np.zeros((50,10));f[:,8]=quality;f[:,9]=keeper_missing
            # Four team-paired channels; missing-window rows remain zero.
            for team,values in pairs:f[:,np.arange(4)*2+team]=values
            local=ids-first
            total[local]+=f*taper[:,None]*quality;weight[local]+=taper*quality
            accepted.append(dict(start=int(start),quality=quality,keeper_unanchored_fraction=keeper_missing))
    # Upsample only within supported windows; do not invent evidence over rejected windows.
    z=np.zeros_like(total);sampled=np.arange(0,7500,5)
    values=total[sampled]/np.maximum(weight[sampled,None],1e-8)
    for col in range(10):z[:,col]=np.interp(np.arange(7500),sampled,values[:,col])
    coverage=np.interp(np.arange(7500),sampled,(weight[sampled]>0).astype(float))
    z=np.column_stack([z,coverage])
    names=[f'pathcrf_{channel}_{team}' for channel in ('sender','coarse_sender','coarse_receiver','decoded_sender') for team in ('H','A')]+['pathcrf_observed_player_fraction','pathcrf_keeper_unanchored_fraction','pathcrf_coverage']
    provenance=dict(checkpoint_sha256=checkpoint_sha,upstream_commit='1b7c6f85a6b78c92e4543fde72ca2b86a933f997',
                    code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    attempted_windows=len(starts),accepted_windows=len(accepted),
                    covered_frames=int((coverage>.5).sum()),left_team=left,windows=accepted,
                    rejected_team_window_causes=diagnostics)
    np.savez_compressed(CACHE/'pathcrf_evidence.npz',features=z.astype(np.float32),names=names,provenance=json.dumps(provenance))
    print('pretrained evidence',provenance,flush=True)
    return z,names,provenance


def selfcheck():
    # Track continuity must survive arbitrary observation ordering and short loss.
    xy=np.stack([np.column_stack([np.arange(11)*5+t*.1,np.arange(11)*2]) for t in range(50)])
    observations=[frame[::-1].copy() for frame in xy]
    observations[20]=observations[20][1:];observations[21]=observations[21][1:]
    result=stitch_team(observations)
    assert result is not None
    recovered,_=result
    assert np.allclose(recovered,xy[:,::-1],atol=1e-6)
    assert stitch_team([frame[:10] for frame in xy]) is None


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',choices=['cpu','mps'],default='cpu');parser.add_argument('--cached',action='store_true');args=parser.parse_args()
    selfcheck()
    torch.set_num_threads(4);CACHE.mkdir(exist_ok=True)
    if args.cached:
        with np.load(CACHE/'pathcrf_evidence.npz') as saved:z,names,source=saved['features'],saved['names'].tolist(),json.loads(str(saved['provenance']))
    else:z,names,source=evidence(torch.device(args.device))
    x,base,_=build_features(broadcast=True);x=np.column_stack([x[11185:18685],z]);names=base+names
    truth=pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))[11185:18685]
    out=np.empty(7500,np.int8);contexts=[]
    for fold in range(5):
        a,b=fold*1500,(fold+1)*1500;train=np.ones(7500,bool);train[max(0,a-250):min(7500,b+250)]=False
        ids=np.flatnonzero(train)[::3];model=fit_model(x[ids],truth[ids],names)
        p=predict(model,x,names);out[a:b]=decode(p)[a:b];contexts.append(p)
    m=pm.evaluate(truth,out)
    report=dict(source=source,feature_count=len(names),metrics={k:m[k] for k in ['spell_f1_mean','macro_f1','n_pred_spells']},
                status='evidence_evaluated' if source['accepted_windows'] else 'No compatible windows: score is a zero-feature control, not pretrained performance',
                protocol='Fixed automatedHungarian stitching; no GT segmentation; pretrained perwindowCRF reset; soft extra evidence only; strict5fold60s+10s embargo, samefoldwholecontext HSMM. No target-based parameter search.',
                limitation='Checkpoint trained onJ03WOH, so no externalJ03WOHheldoutclaim. Ten outfieldtracks stitched perteam; gapfill/carry capped5s. Keepers use uncertain NaNspeedcoordinatesnearowngoal orspatialprior; missingnessfeaturesexplicit. No D prediction.')
    (ROOT/'experiments/pathcrf_results.json').write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(CACHE/'pathcrf_validation.npz',prediction=out,fold_probability=np.stack(contexts))
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
