"""Pretrain ordered possession evidence on external tracking, then adapt per fold."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from torch import nn

from evaluation import possession_metrics as pm
from experiments.load_idsse import load
from experiments.temporal_classifier import TemporalClassifier, masked_loss, selfcheck
from possession.features import build_features, tracking
from possession.model import swap_teams
from possession.duration import decode


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'experiments/cache'


def external_sequences(target_names):
    d = load(ROOT / 'data/external/idsse')
    _, target_ball, _ = tracking()
    mask = np.isfinite(target_ball[11185:90652]).all(1)
    rng = np.random.default_rng(42)
    codes = np.array([{'H': 0, 'A': 1, 'D': 2, 'X': 3}[label] for label in d['labels']], np.int64)
    sequences = []
    for half in (1, 2):
        ids = np.flatnonzero(d['half'] == half)
        players, ball = d['players'][ids].copy(), d['ball'][ids].copy()
        scale = np.array([105.3, 68]) / d['pitch']
        players[:, :, :2] *= scale
        ball *= scale
        for corrupt in (False, True):
            observed_ball = ball.copy()
            if corrupt:
                noise = gaussian_filter1d(rng.normal(size=ball.shape), 3, axis=0)
                noise /= noise.std()
                observed_ball += noise
                seen = np.resize(np.roll(mask, half * 7000), len(ball))
                observed_ball[~seen] = np.nan
            f, names, _ = build_features(arrays=(players, observed_ball, d['teams']), use_tags=False)
            x = np.zeros((len(f), len(target_names)), np.float32)
            for j, name in enumerate(target_names):
                if name in names:
                    x[:, j] = f[:, names.index(name)]
            y = codes[ids][::5].copy()
            # Rolling features and finite differences wrap at array ends.
            y[:25] = 3
            y[-25:] = 3
            sequences.append((x[::5], y))
    return sequences


def pretrain(sequences, names, device):
    torch.manual_seed(23)
    rng = np.random.default_rng(23)
    all_x = np.concatenate([x for x, _ in sequences])
    symmetric = np.concatenate([all_x, swap_teams(all_x, names)])
    center, raw_scale = symmetric.mean(0), symmetric.std(0)
    constant = raw_scale < 1e-5
    scale = raw_scale.clip(.1)
    normalized = []
    for x, y in sequences:
        for values, labels in ((x, y), (swap_teams(x, names), np.where(y < 2, 1 - y, y))):
            normalized.append((np.clip((values - center) / scale, -6, 6), labels))
    net = TemporalClassifier(len(names)).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.01)
    for update in range(200):
        windows, labels = [], []
        for _ in range(16):
            x, y = normalized[int(rng.integers(len(normalized)))]
            start = int(rng.integers(len(x) - 256 + 1))
            windows.append(x[start:start + 256])
            labels.append(y[start:start + 256])
        values = torch.tensor(np.stack(windows).transpose(0, 2, 1), dtype=torch.float32, device=device)
        labels = torch.tensor(np.stack(labels), dtype=torch.long, device=device)
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(net(values), labels.clamp_max(2), reduction='none')[labels < 3].mean()
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite external temporal training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
        if update % 50 == 49:
            print('External temporal update', update + 1, 'CE', float(loss.detach().cpu()), flush=True)
    # Constant external columns have not learned a useful mapping.
    with torch.no_grad():
        net.input.weight[:, constant, :] = 0
    return net, center, scale, constant


def adapt(net, x, y, names, train, center, scale, constant, device):
    torch.manual_seed(23)
    net = copy.deepcopy(net)
    mirrored = swap_teams(x, names)
    selected = np.concatenate([x[train], mirrored[train]])
    center, scale = center.copy(), scale.copy()
    center[constant] = selected[:, constant].mean(0)
    scale[constant] = selected[:, constant].std(0).clip(.1)
    values = np.clip((np.stack([x, mirrored]) - center) / scale, -6, 6).transpose(0, 2, 1)
    values = torch.tensor(values, dtype=torch.float32, device=device)
    labels = torch.tensor(np.stack([y, np.where(y < 2, 1 - y, y)]), dtype=torch.long, device=device)
    mask = torch.tensor(train, device=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0005, weight_decay=.01)
    for _ in range(50):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(net(values), labels, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
    net.eval()
    with torch.no_grad():
        p = net(values).softmax(1).cpu().numpy().transpose(0, 2, 1)
    return (p[0] + p[1][:, [1, 0, 2]]) / 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(4)
    selfcheck()
    device = torch.device(args.device)
    x, names, _ = build_features(broadcast=True)
    sequences = external_sequences(names)
    net, center, scale, constant = pretrain(sequences, names, device)
    # Target labels enter only after all external representation training.
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    core, y = x[start:stop], gt[start:stop]
    sampled = np.arange(0, len(y), 5)
    with np.load(CACHE / 'booster_ensemble.npz') as z:
        hist = z['hist_fold_probability']
    previous = json.loads((ROOT / 'experiments/booster_ensemble_results.json').read_text())
    assert previous['feature_sha256'] == hashlib.sha256(core.tobytes()).hexdigest()
    contexts = []
    predictions = {name: np.empty(len(y), np.int8) for name in ('temporal_transfer', 'equal_hist_transfer')}
    for fold in range(5):
        a, b = fold * 1500, (fold + 1) * 1500
        train = np.ones(len(y), bool)
        train[max(0, a - 250):min(len(y), b + 250)] = False
        p = adapt(net, core[sampled], y[sampled], names, train[sampled], center, scale, constant, device)
        p = np.column_stack([np.interp(np.arange(len(y)), sampled, p[:, k]) for k in range(3)])
        contexts.append(p)
        for method, probability in [('temporal_transfer', p), ('equal_hist_transfer', (hist[fold] + p) / 2)]:
            predictions[method][a:b] = decode(probability)[a:b]
        print('Temporal transfer isolated fold', fold, flush=True)
    results = []
    for method, pred in predictions.items():
        score = pm.evaluate(y, pred)
        row = {'method': method, **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')}}
        results.append(row)
        print(row, flush=True)
    report = {'protocol': 'External J03WOH both halves, clean/corrupt ball, 200 pretraining updates; 50 fold-specific adaptation updates; 5 contiguous minutes/10s embargo/isolated duration decoding',
        'external_frames_5Hz': sum(len(y) for _, y in sequences),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'feature_sha256': hashlib.sha256(core.tobytes()).hexdigest(),
        'results': results}
    (ROOT / 'experiments/temporal_transfer_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(CACHE / 'temporal_transfer_validation.npz', fold_probability=np.array(contexts), **predictions)


if __name__ == '__main__':
    main()
