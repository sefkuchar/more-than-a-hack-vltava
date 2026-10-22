"""Transfer from an unrelated DFL match, including simulated automatic tracking loss."""

import json
import hashlib
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'evaluation')]
import possession_metrics as pm
from possession.features import build_features, tracking
from possession.decode import viterbi
from experiments.compare import fit_model, predict, metrics
from experiments.load_idsse import load


def external_features(target_names):
    cache = ROOT/'experiments/cache/external_features.npz'
    digest = hashlib.sha256(b''.join((ROOT/p).read_bytes() for p in
        ('possession/features.py','experiments/load_idsse.py','experiments/transfer.py'))).hexdigest()
    if cache.exists():
        with np.load(cache) as z:
            if 'digest' in z and str(z['digest']) == digest and z['names'].tolist() == target_names:
                return z['x'], z['y']
    d = load(ROOT/'data/external/idsse')
    players, ball = d['players'].copy(), d['ball'].copy()
    scale = np.array([105.3, 68]) / d['pitch']
    players[:, :, :2] *= scale
    ball *= scale
    rng = np.random.RandomState(42)
    _, target_ball, _ = tracking()
    visibility = np.isfinite(target_ball[11185:90652]).all(axis=1)
    mask = np.tile(visibility, 3)[7000:7000+len(ball)]
    noise = gaussian_filter1d(rng.normal(size=ball.shape), 3, axis=0)
    noise *= 1.0 / noise.std()
    xs, ys = [], []
    for corrupt in (False, True):
        bc = ball.copy()
        if corrupt:
            bc += noise
            bc[~mask] = np.nan
        f, names, _ = build_features(arrays=(players, bc, d['teams']), use_tags=False)
        x = np.zeros((len(f), len(target_names)), dtype=np.float32)
        for j, name in enumerate(target_names):
            if name in names:
                x[:, j] = f[:, names.index(name)]
            elif name in ('broadcast_distance_H', 'broadcast_distance_A', 'broadcast_ball_error'):
                x[:, j] = 30
        selected = np.flatnonzero(d['labels'] != 'X')[::10]
        xs.append(x[selected])
        codes = np.array([{'H':0,'A':1,'D':2,'X':3}[s] for s in d['labels']], dtype=np.int8)
        ys.append(codes[selected])
        print('external', 'corrupt' if corrupt else 'clean', x.shape, flush=True)
    x, y = np.concatenate(xs), np.concatenate(ys)
    np.savez_compressed(cache, x=x, y=y, names=target_names, digest=digest)
    return x, y


if __name__ == '__main__':
    x, names, _ = build_features(broadcast=True)
    gt = pm.label_codes(pm.read_label_csv(ROOT/'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    xs, y = x[start:stop], gt[start:stop]
    ext_x, ext_y = external_features(names)
    results = []
    model = fit_model(ext_x,ext_y,names)
    p = predict(model,xs,names)
    np.savez_compressed(ROOT/'experiments/cache/external_probability.npz', probability=p)
    for c in (8,16,32):
        row=dict(method='external_only',penalty=c,**metrics(y,viterbi(np.log(p.clip(1e-5,1)),c)))
        results.append(row); print(row,flush=True)
    for weight in (3., 10.):
        p = np.empty((len(y),3))
        decoded = {c:np.empty(len(y),dtype=np.int8) for c in (8,16,32)}
        for fold in range(5):
            a,b=fold*1500,(fold+1)*1500
            train=np.ones(len(y),bool)
            train[max(0,a-250):min(len(y),b+250)]=False
            ids=np.flatnonzero(train)[::3]
            tx=np.concatenate([ext_x,xs[ids]])
            ty=np.concatenate([ext_y,y[ids]])
            weights=np.concatenate([np.ones(len(ext_y)),np.full(len(ids),weight)])
            model=fit_model(tx,ty,names,sample_weight=weights)
            probability=predict(model,xs,names)
            p[a:b]=probability[a:b]
            for c in decoded:
                decoded[c][a:b]=viterbi(np.log(probability.clip(1e-5,1)),c)[a:b]
            print('transfer fold',weight,fold,flush=True)
        np.savez_compressed(ROOT/('experiments/cache/transfer_{}_oof.npz'.format(int(weight))),probability=p)
        for c in (8,16,32):
            row=dict(method='transfer_oof',target_weight=weight,penalty=c,
                     **metrics(y,decoded[c]))
            results.append(row);print(row,flush=True)
    (ROOT/'experiments/transfer_results.json').write_text(json.dumps(results,indent=2)+'\n')
