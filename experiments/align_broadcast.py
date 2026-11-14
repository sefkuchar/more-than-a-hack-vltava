"""Label-free spatial alignment diagnostic; caches only under ignored data/."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]

def sampled(source, stride=25):
    path=ROOT/'data'/source/f'BAR-ATH_{source}_atd.txt'
    fs=[]; ps=[]; balls=[]
    for i,line in enumerate(path.open()):
        if i%stride: continue
        _,p,b=line.strip().split(':')
        p=np.fromstring(p.replace(';',','),sep=',').reshape(-1,3)
        # Exclude dummy goalkeepers and missing/implausible positions.
        p=p[np.isfinite(p).all(axis=1),:2]*[105.3,68]
        fs.append(i);ps.append(p);balls.append(np.fromstring(b,sep=',')[:2]*[105.3,68])
    return np.array(fs),ps,np.array(balls)

def signature(p):
    return np.quantile(p,np.linspace(.05,.95,9),axis=0).T.ravel()

def geometry(a,b):
    d=np.linalg.norm(a[:,None]-b[None],axis=2)
    da=d.min(axis=1);db=d.min(axis=0)
    return (np.median(da)+np.median(db))/2, (np.mean(da<2)+np.mean(db<2))/2

def main():
    tf,tp,tb=sampled('tactical');bf,bp,bb=sampled('broadcast')
    ti=np.array([i for i,p in enumerate(tp) if len(p)>=12]);bi=np.array([i for i,p in enumerate(bp) if len(p)>=12])
    tree=cKDTree(np.array([signature(tp[i]) for i in ti]))
    _,candidates=tree.query(np.array([signature(bp[i]) for i in bi]),k=30)
    anchors=[]
    for j,ii in enumerate(bi):
        best=None
        for kk in candidates[j]:
            k=ti[kk];cost,coverage=geometry(tp[k],bp[ii])
            if best is None or cost<best[0]:best=(cost,coverage,k)
        cost,cov,k=best
        if cost<1.2 and cov>.65: anchors.append((int(bf[ii]),int(tf[k]),float(cost),float(cov)))
    a=np.array(anchors)
    # A correct playback segment has near-constant time offset; require support locally.
    reliable=[]
    for row in a:
        near=np.abs(a[:,0]-row[0])<=500
        supported=np.abs((a[:,1]-a[:,0])-(row[1]-row[0]))<=25
        if np.sum(near & supported)>=5: reliable.append(row)
    r=np.array(reliable)
    out=ROOT/'data'/'broadcast'/'alignment_anchors.npz'
    np.savez_compressed(out,anchors=a,reliable=r)
    print('samples',len(tf),len(bf),'eligible',len(ti),len(bi),'anchors',len(a),'reliable',len(r),flush=True)
    if not len(r): return
    # Split reliable anchors when gaps exceed 5s or offsets jump by >1s.
    groups=[];start=0
    for i in range(1,len(r)+1):
        if i==len(r) or r[i,0]-r[i-1,0]>125 or abs((r[i,1]-r[i,0])-(r[i-1,1]-r[i-1,0]))>25:
            q=r[start:i]; start=i
            if len(q)>=3:groups.append(dict(b0=int(q[0,0]),b1=int(q[-1,0]),t0=int(q[0,1]),t1=int(q[-1,1]),offset=float(np.median(q[:,1]-q[:,0])),n=len(q),median_geometry_m=float(np.median(q[:,2]))))
    print('continuous anchor groups',len(groups),'saved under data/broadcast',flush=True)
    first=r[(r[:,1]>=11185)&(r[:,1]<=18684)]
    print('first5min anchors',len(first),'offset percentiles',np.quantile(first[:,1]-first[:,0],[0,.5,1]).tolist() if len(first) else [])
    (ROOT/'data'/'broadcast'/'alignment_segments.json').write_text(json.dumps(groups,indent=2))


def aligned_broadcast(force=False, data_dir=None):
    """Return tactical-indexed broadcast evidence; no possession labels are read.

    Keys include ball/nearest using broadcast observations, corrected_nearest
    using tactical ball/player positions and cross-camera team corrections,
    team_correction_fraction, confidence and broadcast_frame. corrected_nearest
    interpolates the tactical ball for features only; its contact evidence must
    be multiplied by the tactical observed-ball flag. No labels are read.
    """
    import sys
    sys.path.insert(0,str(ROOT))
    from possession.features import tracking, interpolate
    from games import half_video_blocks, require, source_frames
    raw=Path(require(['atd_raw'],data_dir=data_dir,source='broadcast')[0])
    tactical=require(['atd_meta','atd_raw'],data_dir=data_dir)
    cache=raw.parent/'aligned_evidence.npz'
    changed=max(Path(p).stat().st_mtime for p in [raw,*tactical,Path(__file__)])
    if cache.exists() and cache.stat().st_mtime>=changed and not force:
        with np.load(cache) as saved:
            if {'corrected_nearest','team_correction_fraction','team_correction_count'}.issubset(saved.files):
                return dict(saved)
    tp,tb,tt=tracking(data_dir)
    with raw.open() as handle:
        slots=len(handle.readline().split(':')[1].split(';'))
    bp=np.full((source_frames('broadcast'),slots,3),np.nan,dtype=np.float32)
    bb=np.full((len(bp),2),np.nan,dtype=np.float32)
    for i,line in enumerate(raw.open()):
        frame,p,b=line.strip().split(':')
        if int(frame)!=i+1 or i>=len(bp):
            raise ValueError('Non-contiguous broadcast tracking')
        bp[i]=np.fromstring(p.replace(';',','),sep=',').reshape(-1,3)
        bb[i]=np.fromstring(b,sep=',')[:2]
    if i+1!=len(bp):
        raise ValueError('Incomplete broadcast tracking')
    bp[:,:,:2]*=[105.3,68];bb*=[105.3,68]
    # Offsets identified by globally matched geometry and refined at single-frame resolution.
    # They are source timing properties, not possession-label parameters.
    offsets=(4014,26920)
    halves=half_video_blocks('BAR-ATH')
    match=np.full(len(bp),-1,dtype=np.int32)
    quality=np.zeros(len(bp),dtype=np.float32)
    votes=np.zeros((bp.shape[1],2),dtype=np.int64)
    for b in range(len(bp)):
        bv=np.isfinite(bp[b]).all(axis=1)
        if bv.sum()<8:continue
        best=None
        for off,(_,lo,hi) in zip(offsets,halves):
            t=b+off
            if not lo<=t<=hi:continue
            tv=np.isfinite(tp[t]).all(axis=1)
            if tv.sum()<8:continue
            x=tp[t,tv,:2];y=bp[b,bv,:2]
            d=np.linalg.norm(y[:,None]-x[None],axis=2)
            da=d.min(axis=1);db=d.min(axis=0)
            cost=(np.median(da)+np.median(db))/2
            cov=(np.mean(da<2)+np.mean(db<2))/2
            if cost<2 and cov>.55 and (best is None or cost<best[0]):
                best=(cost,cov,t,d,tv,bv)
        if best is None:continue
        cost,cov,t,d,tv,bv=best
        match[b]=t;quality[b]=cov*np.exp(-cost)
        if b%5==0:
            ids=np.flatnonzero(bv);near=d.argmin(axis=1);dist=d.min(axis=1)
            target=tt[tv][near]
            for s,k,dd in zip(ids,target,dist):
                if dd<1.0 and k in (0,1):votes[s,k]+=1
    counts=votes.sum(axis=1)
    teamconf=votes.max(axis=1)/np.maximum(counts,1)
    teams=np.where((counts>=20)&(teamconf>=.9),votes.argmax(axis=1),-1)
    n=len(tp);ball=np.full((n,2),np.nan,dtype=np.float32)
    nearest=np.full((n,2),np.nan,dtype=np.float32)
    confidence=np.zeros(n,dtype=np.float32);index=np.full(n,-1,dtype=np.int32)
    corrected_teams=np.broadcast_to(tt.astype(np.int8),(n,len(tt))).copy()
    real=np.isfinite(tp).all(axis=2)
    correction_count=np.zeros(n,dtype=np.int16)
    for b in np.flatnonzero(match>=0):
        t=match[b];index[t]=b;confidence[t]=quality[b]
        # Mutual nearest geometry prevents one broadcast person from relabelling
        # multiple overlapping tactical tracks. Unknown broadcast teams abstain.
        ti=np.flatnonzero(real[t])
        bi=np.flatnonzero(np.isfinite(bp[b]).all(axis=1)&(teams>=0))
        if len(ti) and len(bi):
            d=np.linalg.norm(tp[t,ti,:2][:,None,:]-bp[b,bi,:2][None,:,:],axis=2)
            near_b=d.argmin(axis=1);near_t=d.argmin(axis=0)
            for j,k in enumerate(near_b):
                if near_t[k]==j and d[j,k]<1.0:
                    corrected_teams[t,ti[j]]=teams[bi[k]]
            correction_count[t]=np.sum(corrected_teams[t,ti]!=tt[ti])
        if not np.isfinite(bb[b]).all():continue
        ball[t]=bb[b]
        distance=np.linalg.norm(bp[b,:,:2]-bb[b],axis=1)
        for k in (0,1):
            valid=(teams==k)&np.isfinite(bp[b]).all(axis=1)
            if valid.any():nearest[t,k]=distance[valid].min()
    distance=np.linalg.norm(tp[:,:,:2]-interpolate(tb)[:,None,:],axis=2)
    distance[~real]=np.inf
    corrected_nearest=np.stack([np.min(np.where(corrected_teams==k,distance,np.inf),axis=1)
                                for k in (0,1)],axis=1).astype(np.float32)
    corrected_nearest[~np.isfinite(corrected_nearest)]=100.
    correction_fraction=correction_count/np.maximum(real.sum(axis=1),1)
    result=dict(ball=ball,nearest=nearest,confidence=confidence,broadcast_frame=index,
                slot_team=teams,slot_team_confidence=teamconf,slot_team_votes=votes,
                offsets=np.array(offsets),corrected_nearest=corrected_nearest,
                team_correction_fraction=correction_fraction.astype(np.float32),
                team_correction_count=correction_count)
    np.savez_compressed(cache,**result)
    observed=np.isfinite(tb).all(axis=1);extra=np.isfinite(ball).all(axis=1)
    graded=np.zeros(n,dtype=bool)
    for _,lo,hi in halves:graded[lo:hi+1]=True
    for name,mask in [('graded',graded),('public_window',(np.arange(n)>=11185)&(np.arange(n)<=18684))]:
        print(name,json.dumps(dict(mapped=int(np.sum((index>=0)&mask)),frames=int(mask.sum()),
          tactical_ball=int(np.sum(observed&mask)),broadcast_ball=int(np.sum(extra&mask)),
          newly_visible=int(np.sum(extra&~observed&mask)),combined_visible=int(np.sum((extra|observed)&mask)),
          team_assignments_changed=int(correction_count[mask].sum()),
          frames_with_team_changes=int(np.sum((correction_count>0)&mask)))),flush=True)
    print('slot teams',teams.tolist(),'confidence',np.round(teamconf,3).tolist(),flush=True)
    both=observed&extra
    print('ball agreement metres',np.quantile(np.linalg.norm(tb[both]-ball[both],axis=1),[.25,.5,.75,.9]).tolist(),flush=True)
    # Geometric nearest-team agreement is diagnostic, not a possession accuracy score.
    ids=np.flatnonzero(both & np.isfinite(nearest).all(axis=1))
    agree=[]
    for t in ids:
        ds=np.linalg.norm(tp[t,:,:2]-tb[t],axis=1)
        ds[~np.isfinite(tp[t]).all(axis=1)]=np.inf
        near=[np.min(ds[tt==k]) for k in (0,1)]
        if np.isfinite(near).all():agree.append(int(np.argmin(near)==np.argmin(nearest[t])))
    print('nearest team agreement',float(np.mean(agree)),'n',len(agree),flush=True)
    return result


if __name__ == "__main__":
    import sys
    if "--fuse" in sys.argv:
        aligned_broadcast(force="--force" in sys.argv)
    else:
        main()
