"""Reproducible physical baseline and blocked validation; no random frame splits."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'evaluation')]
import possession_metrics as pm
from possession.features import build_features, rolling
from possession.decode import viterbi
from possession.model import fit_model, predict, swap_teams


def metrics(y, p):
    m = pm.evaluate(y, p)
    return {k: m[k] for k in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')}


if __name__ == '__main__':
    x, names, raw = build_features()
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    xs, y = x[start:stop], gt[start:stop]
    print('Features', x.shape, flush=True)
    print('Segment diagnostics: start stop label visibility player_speed ball_speed boundary nearH nearA', flush=True)
    for a, b, k in pm.segments(gt, [(start,stop)]):
        print(a,b,pm.LABELS[k],*[round(float(np.mean(raw[col][a:b])),2) for col in ('observed','mean_speed','ball_speed','boundary')],
              np.round(np.mean(raw['nearest'][a:b],axis=0),2), flush=True)
    results = []
    # This control-only method does not learn from public possession labels.
    for r in (5, 12, 25):
        contacts = rolling(raw['contact'], r)[start:stop]
        for penalty in (2, 5, 10):
            evidence = np.column_stack([contacts, np.full(len(y), -.05)])
            pred = viterbi(evidence, penalty)
            row = dict(method='contact', radius=r, penalty=penalty, **metrics(y,pred))
            results.append(row)
            print(row, flush=True)
    # Five contiguous 60-second folds, with ten seconds removed on each training side.
    # Every OOF emission is predicted by a model that did not see that block's labels.
    for kind in ('hist','extra'):
        oof = np.empty((len(y),3))
        decoded = {c: np.empty(len(y),dtype=np.int8) for c in (0,4,8,16,32)}
        for fold in range(5):
            a,b = fold*1500,(fold+1)*1500
            train = np.ones(len(y), dtype=bool)
            train[max(0,a-250):min(len(y),b+250)] = False
            inds = np.flatnonzero(train)[::3]
            model = fit_model(xs[inds], y[inds], names, kind)
            probability = predict(model,xs,names)
            oof[a:b] = probability[a:b]
            for c in decoded:
                decoded[c][a:b] = viterbi(np.log(probability.clip(1e-5,1)),c)[a:b]
            print('fold done',kind,fold,flush=True)
        out = ROOT / 'experiments/cache'
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / (kind+'_oof.npz'), probability=oof, names=names)
        for penalty in (0, 4, 8, 16, 32):
            pred = decoded[penalty]
            row = dict(method=kind, penalty=penalty, **metrics(y,pred))
            results.append(row)
            print(row,flush=True)
    (ROOT / 'experiments/initial_results.json').write_text(json.dumps(results,indent=2)+'\n')
