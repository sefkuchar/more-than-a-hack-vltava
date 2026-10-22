"""Test reconstructed-ball evidence without replacing observed tracking.

The ball checkpoint is trained on an unrelated match. Public possession labels
are opened only after the target reconstruction and its features are fixed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluation import possession_metrics as pm
from experiments.ball_reconstruction import BallNet, inputs, player_features, PITCH, LENGTH, STRIDE, CACHE, ROOT
from games import half_video_blocks
from possession.duration import decode
from possession.features import build_features, tracking, interpolate, history
from possession.model import fit_model, predict, swap_teams


def reconstruct(device, suffix=''):
    checkpoint = CACHE / f'ball_reconstruction{suffix}.pt'
    net = BallNet().to(device)
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['state_dict'])
    net.eval()
    players, ball, teams = tracking()
    features, valid = player_features(players[::STRIDE], teams)
    sampled_ball = ball[::STRIDE] / PITCH
    sampled_seen = np.isfinite(sampled_ball).all(1)
    n = len(sampled_ball)
    total = np.zeros((n, 2), np.float64)
    square = np.zeros((n, 2), np.float64)
    weight = np.zeros(n)
    anchors = np.zeros(n)
    ownership = np.zeros((n, 2))
    entropy = np.zeros(n)
    starts = []
    for _, first, last in half_video_blocks('BAR-ATH'):
        a, b = (first + STRIDE - 1) // STRIDE, last // STRIDE + 1
        starts.extend(np.unique(np.r_[np.arange(a, b - LENGTH + 1, 30), b - LENGTH]).tolist())
    taper = np.maximum(.1, 1 - np.abs(np.linspace(-1, 1, LENGTH)))
    team_tensor = torch.tensor(teams, device=device)
    with torch.no_grad():
        for first in range(0, len(starts), 32):
            batch = starts[first:first + 32]
            packed = []
            masks = []
            for start in batch:
                ids = start + np.arange(LENGTH)
                packed.append(inputs(features[ids], valid[ids], sampled_ball[ids], sampled_seen[ids]))
                masks.append(valid[ids])
            player = torch.tensor(np.stack([item[0] for item in packed]), device=device)
            anchor = torch.tensor(np.stack([item[1] for item in packed]), device=device)
            mask_tensor = torch.tensor(np.stack(masks), device=device)
            prediction, attention = net(player, mask_tensor, anchor, team_tensor, return_weights=True)
            prediction = prediction.cpu().numpy()
            mass = torch.stack([attention[:, :, team_tensor == k].sum(2) for k in (0, 1)], dim=2).cpu().numpy()
            uncertainty = (-(attention * attention.clamp_min(1e-8).log()).sum(2) / mask_tensor.sum(2).clamp_min(2).log()).cpu().numpy()
            for i, start in enumerate(batch):
                ids = start + np.arange(LENGTH)
                total[ids] += prediction[i] * taper[:, None]
                square[ids] += prediction[i] ** 2 * taper[:, None]
                weight[ids] += taper
                anchors[ids] += sampled_seen[ids].mean() * taper
                ownership[ids] += mass[i] * taper[:, None]
                entropy[ids] += uncertainty[i] * taper
    mean = total / weight.clip(1e-6)[:, None]
    std = np.sqrt(np.maximum(0, square / weight.clip(1e-6)[:, None] - mean ** 2))
    coordinates = np.column_stack([np.interp(np.arange(len(ball)), np.arange(n) * STRIDE, mean[:, k]) * PITCH[k] for k in (0, 1)])
    spread = np.interp(np.arange(len(ball)), np.arange(n) * STRIDE, np.linalg.norm(std * PITCH, axis=1))
    fraction = np.interp(np.arange(len(ball)), np.arange(n) * STRIDE, anchors / weight.clip(1e-6))
    seen = np.isfinite(ball).all(1)
    linear = interpolate(ball)
    edges = np.r_[0, np.flatnonzero(seen[1:] != seen[:-1]) + 1, len(seen)]
    gap_length = np.zeros(len(seen))
    for a, b in zip(edges[:-1], edges[1:]):
        if not seen[a]:
            gap_length[a:b] = b - a
    # The external benchmark favors interpolation for 1-second gaps.
    short = seen | (gap_length < 50)
    coordinates[short] = linear[short]
    distance = np.linalg.norm(players[:, :, :2] - coordinates[:, None], axis=2)
    distance[~np.isfinite(players[:, :, 2])] = 100
    distance = np.nan_to_num(distance, nan=100)
    nearest = np.column_stack([distance[:, teams == k].min(1) for k in (0, 1)])
    z = np.column_stack([np.minimum(nearest, 30), np.exp(-.5 * (nearest / 3) ** 2),
        np.minimum(np.linalg.norm(coordinates - linear, axis=1), 30), np.minimum(spread, 30), fraction])
    z[short] = 0
    names = ['reconstructed_distance_H', 'reconstructed_distance_A',
        'reconstructed_contact_H', 'reconstructed_contact_A',
        'reconstructed_disagreement', 'reconstructed_spread', 'reconstructed_anchor_fraction']
    if suffix.endswith('_contact'):
        mass = np.column_stack([np.interp(np.arange(len(ball)), np.arange(n) * STRIDE,
                ownership[:, k] / weight.clip(1e-6)) for k in (0, 1)])
        uncertainty = np.interp(np.arange(len(ball)), np.arange(n) * STRIDE, entropy / weight.clip(1e-6))
        z = np.column_stack([z, mass, history(mass, 25), history(mass, 75), uncertainty])
        names += ['neural_contact_H', 'neural_contact_A', 'neural_contact_past25_H',
            'neural_contact_past25_A', 'neural_contact_past75_H', 'neural_contact_past75_A', 'neural_contact_entropy']
    assert np.isfinite(z).all() and np.array_equal(coordinates[seen], ball[seen])
    source = {'checkpoint': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              'code': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    np.savez_compressed(CACHE / f'ball_target_features{suffix}.npz', features=z.astype(np.float32), names=names,
                        ball=coordinates.astype(np.float32), source=json.dumps(source))
    return z.astype(np.float32), names, source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--cached', action='store_true')
    parser.add_argument('--adapt-target', action='store_true')
    parser.add_argument('--contact-supervision', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.contact_supervision and not args.adapt_target:
        parser.error('--contact-supervision requires --adapt-target')
    suffix = '_target_contact' if args.contact_supervision else '_target' if args.adapt_target else ''
    if args.cached:
        with np.load(CACHE / f'ball_target_features{suffix}.npz') as data:
            z, names, source = data['features'], data['names'].tolist(), json.loads(str(data['source']))
    else:
        z, names, source = reconstruct(torch.device(args.device), suffix)
    x, base_names, _ = build_features(broadcast=True)
    x = np.column_stack([x, z])
    names = base_names + names
    # All feature fitting and reconstruction above are independent of these labels.
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    core, y = x[start:stop], gt[start:stop]
    assert np.array_equal(swap_teams(swap_teams(core, names), names), core)
    result = np.empty(len(y), np.int8)
    contexts = np.empty((5, len(y), 3))
    for fold in range(5):
        a, b = fold * 1500, (fold + 1) * 1500
        train = np.ones(len(y), bool)
        train[max(0, a - 250):min(len(y), b + 250)] = False
        ids = np.flatnonzero(train)[::3]
        model = fit_model(core[ids], y[ids], names)
        probability = predict(model, core, names)
        contexts[fold] = probability
        result[a:b] = decode(probability)[a:b]
        print('Reconstructed ball isolated fold', fold, flush=True)
    score = pm.evaluate(y, result, bootstrap_n=1000)
    report = {'source_sha256': source, 'feature_count': len(names),
        'protocol': 'Unrelated-match masked-ball model; 5 contiguous minute folds with 10s embargo and isolated full-context decoding',
        'short_gap_policy': 'Keep interpolation below 2 seconds; preserve every observed ball coordinate',
        'model': {key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_gt_spells', 'n_pred_spells', 'spell_bootstrap_low', 'spell_bootstrap_high')}}
    (ROOT / f'experiments/ball_possession{suffix}_report.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(CACHE / f'ball_possession_validation{suffix}.npz', prediction=result, fold_probability=contexts)
    print(json.dumps(report['model'], indent=2), flush=True)


if __name__ == '__main__':
    main()
