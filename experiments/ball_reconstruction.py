"""Small masked-ball benchmark; external first half trains, second half tests.

Run with .venv-ball/bin/python -m experiments.ball_reconstruction --device mps.
This is an original, smaller experiment inspired by BallRadar/TranSPORTmer, not
a reproduction of their architectures or published scores. Possession labels
are never inputs. Frozen external dead-ball coordinates are excluded targets.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from experiments.load_idsse import load


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'experiments/cache'
PITCH = np.array([105.3, 68], np.float32)
LENGTH, STRIDE = 61, 4
MODES = {'1s': 6, '5s': 31, '90percent': 55, 'hidden': 61}


def player_features(players, teams, pitch=PITCH, dt=4 / 25):
    """Identity is used only for short velocity differences, never model tokens."""
    valid = np.isfinite(players).all(axis=2)
    xy = np.nan_to_num(players[:, :, :2] / pitch).astype(np.float32)
    velocity = np.zeros_like(xy)
    velocity[1:] = (xy[1:] - xy[:-1]) * pitch / dt / 10
    velocity[1:][~(valid[1:] & valid[:-1])] = 0
    velocity[np.linalg.norm(velocity, axis=2) > 2] = 0
    center = np.zeros_like(xy)
    for team in (0, 1):
        mask = valid & (teams == team)[None, :]
        mean = (xy * mask[:, :, None]).sum(1) / mask.sum(1).clip(1)[:, None]
        center[:, teams == team] = mean[:, None]
    features = np.concatenate([xy, velocity,
        np.nan_to_num(players[:, :, 2:3] / 10), xy - center], axis=2)
    features[~valid] = 0
    return features.astype(np.float32), valid


def inputs(features, valid, ball, seen):
    """Every ball-derived input depends exclusively on explicitly visible xy."""
    t = np.arange(len(ball))
    base = np.empty_like(ball)
    if seen.any():
        for axis in (0, 1):
            base[:, axis] = np.interp(t, t[seen], ball[seen, axis])
    else:
        base[:] = (features[:, :, :2] * valid[:, :, None]).sum(1) / valid.sum(1).clip(1)[:, None]
    previous = np.maximum.accumulate(np.where(seen, t, -len(t)))
    following = np.minimum.accumulate(np.where(seen, t, 2 * len(t))[::-1])[::-1]
    age_before = np.clip((t - previous) / 60, 0, 1)
    age_after = np.clip((following - t) / 60, 0, 1)
    velocity = np.zeros_like(base)
    velocity[1:] = (base[1:] - base[:-1]) * PITCH / (4 / 25) / 10
    velocity = np.clip(velocity, -3, 3)
    anchor = np.column_stack([base, seen, age_before, age_after,
        np.full(len(t), seen.mean()), velocity, np.full(len(t), seen.any())]).astype(np.float32)
    relative = features[:, :, :2] - base[:, None]
    return np.concatenate([features, relative], axis=2), anchor


class Windows(Dataset):
    def __init__(self, features, valid, ball, starts, train=False, seed=0):
        self.features, self.valid, self.ball = features, valid, ball
        self.starts, self.train, self.seed = np.asarray(starts), train, seed
        self.epoch = 0

    def __len__(self):
        return 2560 if self.train else len(self.starts) * len(MODES)

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + index)
        if self.train:
            start = rng.choice(self.starts)
            mode = int(rng.integers(len(MODES)))
        else:
            start, mode = self.starts[index // len(MODES)], index % len(MODES)
        ids = start + np.arange(LENGTH)
        features = self.features[ids].copy()
        valid, ball = self.valid[ids], self.ball[ids].copy()
        available = np.isfinite(ball).all(1)
        if self.train:
            for axis in (0, 1):
                if rng.random() < .5:
                    features[:, :, axis] = 1 - features[:, :, axis]
                    features[:, :, axis + 2] *= -1
                    features[:, :, axis + 5] *= -1
                    ball[:, axis] = 1 - ball[:, axis]
        width = list(MODES.values())[mode]
        first = int(rng.integers(1, LENGTH - width)) if self.train and width < LENGTH - 1 else (LENGTH - width) // 2
        seen = np.ones(LENGTH, bool)
        seen[first:first + width] = False
        seen &= available
        player, anchor = inputs(features, valid, ball, seen)
        return (player, valid, anchor, np.nan_to_num(ball), available & ~seen, np.int64(mode))


class BallNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(9, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.temporal = nn.GRU(128 + 9, 48, batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))
        self.output = nn.Linear(96, 3)

    def forward(self, player, valid, anchor, teams, return_weights=False):
        embedding = self.phi(player)
        pools = []
        for team in (0, 1):
            mask = valid & (teams == team)[None, None, :]
            mean = (embedding * mask[..., None]).sum(2) / mask.sum(2).clamp_min(1)[..., None]
            maximum = embedding.masked_fill(~mask[..., None], -1e4).amax(2)
            maximum = torch.where(mask.any(2)[..., None], maximum, 0)
            pools.append(torch.cat([mean, maximum], dim=-1))
        symmetric = torch.cat([(pools[0] + pools[1]) / 2, (pools[0] - pools[1]).abs()], dim=-1)
        context, _ = self.temporal(torch.cat([symmetric, anchor], dim=-1))
        scores = self.attention(torch.cat([embedding, context[:, :, None].expand(-1, -1, player.shape[2], -1)], dim=-1)).squeeze(-1)
        weights = scores.masked_fill(~valid, -1e4).softmax(dim=2)
        candidate = (player[:, :, :, :2] * weights[..., None]).sum(2)
        head = self.output(context)
        gate = head[:, :, 2:].sigmoid() * anchor[:, :, 8:9]
        prediction = gate * anchor[:, :, :2] + (1 - gate) * candidate + .1 * head[:, :, :2].tanh()
        return (prediction, weights) if return_weights else prediction


def external_windows():
    data = load(ROOT / 'data/external/idsse')
    players = data['players'][::STRIDE].copy()
    players[:, :, :2] *= PITCH / data['pitch']
    ball = data['ball'][::STRIDE] / data['pitch']
    features, valid = player_features(players, data['teams'])
    # Require every original 25-Hz frame in a window to belong to the live half.
    starts = {}
    for half in (1, 2):
        good = (data['half'] == half) & (data['status'] == 1)
        bad_prefix = np.r_[0, np.cumsum(~good)]
        candidates = np.arange(0, len(data['ball']) - (LENGTH - 1) * STRIDE, STRIDE)
        end = candidates + (LENGTH - 1) * STRIDE + 1
        candidates = candidates[bad_prefix[end] == bad_prefix[candidates]] // STRIDE
        if half == 2:
            selected, next_start = [], -1
            for start in candidates:
                if start >= next_start:
                    selected.append(start)
                    next_start = start + LENGTH
            candidates = np.array(selected)
        starts[half] = candidates
    return features, valid, ball.astype(np.float32), data['teams'], starts


def target_windows():
    from games import half_video_blocks
    from possession.features import tracking
    players, ball, teams = tracking()
    features, valid = player_features(players[::STRIDE], teams)
    ball = ball[::STRIDE] / PITCH
    available = np.isfinite(ball).all(1)
    starts = {}
    for half, (_, first, last) in enumerate(half_video_blocks('BAR-ATH'), 1):
        a, b = (first + STRIDE - 1) // STRIDE, last // STRIDE + 1
        candidates = np.arange(a, b - LENGTH + 1, 5 if half == 1 else LENGTH)
        selected = []
        for start in candidates:
            # No public-window physical targets or ten-second neighboring context.
            if half == 1 and start * STRIDE <= 18684 + 250 and (start + LENGTH - 1) * STRIDE >= 11185 - 250:
                continue
            ids = start + np.arange(LENGTH)
            if available[ids].mean() >= .5 and valid[ids].sum(1).min() >= 14:
                selected.append(start)
        starts[half] = np.array(selected)
    assert len(starts[1]) and len(starts[2])
    return features, valid, ball.astype(np.float32), teams, starts


def selfcheck():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    features = rng.normal(size=(61, 6, 7)).astype(np.float32)
    valid = np.ones((61, 6), bool)
    ball = rng.uniform(size=(61, 2)).astype(np.float32)
    seen = np.ones(61, bool)
    seen[10:50] = False
    changed = ball.copy()
    changed[~seen] += 100
    x, a = inputs(features, valid, ball, seen)
    x2, a2 = inputs(features, valid, changed, seen)
    assert np.array_equal(x, x2) and np.array_equal(a, a2), 'Hidden truth leaked into input'
    net = BallNet().eval()
    teams = torch.tensor([0, 0, 0, 1, 1, 1])
    x, v, a = torch.tensor(x[None]), torch.tensor(valid[None]), torch.tensor(a[None])
    with torch.no_grad():
        prediction = net(x, v, a, teams)
        order = torch.tensor([3, 0, 4, 1, 5, 2])
        assert torch.allclose(prediction, net(x[:, :, order], v[:, :, order], a, teams[order]), atol=1e-6)
        assert torch.allclose(prediction, net(x, v, a, 1 - teams), atol=1e-6)
    assert prediction.shape == (1, 61, 2) and torch.isfinite(prediction).all()
    print('Hidden-target isolation, player permutation and team-swap checks passed', flush=True)


def constant_velocity(anchor):
    result = anchor[:, :2].copy()
    known = np.flatnonzero(anchor[:, 2])
    if not len(known):
        return result  # No anchors: the explicit baseline is the player centroid.
    for t in np.flatnonzero(~anchor[:, 2].astype(bool)):
        past = known[known < t]
        last = past[-1] if len(past) else known[0]
        velocity = (anchor[last, :2] - anchor[past[-2], :2]) / (last - past[-2]) if len(past) > 1 else 0
        result[t] = anchor[last, :2] + (t - last) * velocity
    return result


def evaluate(net, dataset, teams, device):
    rows = {mode: {'model': [], 'interpolation_or_centroid': [], 'constant_velocity_or_centroid': []} for mode in MODES}
    pitch = torch.tensor(PITCH, device=device)
    net.eval()
    with torch.no_grad():
        for player, valid, anchor, target, missing, mode in DataLoader(dataset, batch_size=32):
            velocity_baseline = torch.tensor(np.stack([constant_velocity(a) for a in anchor.numpy()]), device=device)
            player, valid, anchor, target, missing = [value.to(device) for value in (player, valid, anchor, target, missing)]
            pred = net(player, valid, anchor, teams)
            for name, value in [('model', pred), ('interpolation_or_centroid', anchor[:, :, :2]),
                                ('constant_velocity_or_centroid', velocity_baseline)]:
                errors = ((value - target) * pitch).square().sum(-1).sqrt().cpu().numpy()
                mask = missing.cpu().numpy()
                for i, key in enumerate(mode.tolist()):
                    selected = errors[i, mask[i]]
                    if len(selected):
                        rows[list(MODES)[key]][name].append([float(selected.mean()), float(selected.max())])
    return {mode: {name: {'windows': len(values), 'ADE_m': float(np.mean(values, axis=0)[0]),
        'mean_window_max_error_m': float(np.mean(values, axis=0)[1])} for name, values in methods.items()}
        for mode, methods in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--selfcheck', action='store_true')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--adapt-target', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--contact-supervision', action='store_true')
    args = parser.parse_args()
    if args.contact_supervision and not args.adapt_target:
        parser.error('--contact-supervision requires --adapt-target')
    torch.set_num_threads(4)
    selfcheck()
    if args.selfcheck:
        return
    torch.manual_seed(7)
    np.random.seed(7)
    device = torch.device(args.device)
    if device.type == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable in this process; use --device cpu or permit GPU access')
    features, valid, ball, teams, starts = target_windows() if args.adapt_target else external_windows()
    train = Windows(features, valid, ball, starts[1], train=True, seed=7)
    heldout = Windows(features, valid, ball, starts[2])
    net = BallNet().to(device)
    source_checkpoint = None
    if args.adapt_target:
        source_checkpoint = CACHE / ('ball_reconstruction_target.pt' if args.contact_supervision else 'ball_reconstruction.pt')
        net.load_state_dict(torch.load(source_checkpoint, map_location=device, weights_only=True)['state_dict'])
    suffix = '_target_contact' if args.contact_supervision else '_target' if args.adapt_target else ''
    checkpoint = CACHE / f'ball_reconstruction{suffix}.pt'
    if args.evaluate:
        net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['state_dict'])
    team_tensor = torch.tensor(teams, device=device)
    pitch = torch.tensor(PITCH, device=device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0003 if args.adapt_target else .001, weight_decay=.0001)
    loader = DataLoader(train, batch_size=32, shuffle=False)
    print('Device', device, 'training starts', len(starts[1]), 'held-out windows', len(starts[2]), flush=True)
    began, losses = time.monotonic(), []
    for epoch in range(0 if args.evaluate else args.epochs):
        net.train()
        train.epoch = epoch
        total = 0
        for player, valid_batch, anchor, target, missing, _ in loader:
            player, valid_batch, anchor, target, missing = [value.to(device) for value in (player, valid_batch, anchor, target, missing)]
            optimizer.zero_grad(set_to_none=True)
            prediction, attention = net(player, valid_batch, anchor, team_tensor, return_weights=True)
            error = ((prediction - target) * pitch).square().sum(-1).add(.01).sqrt()
            loss = error[missing].mean()
            if args.contact_supervision:
                distance = ((player[:, :, :, :2] - target[:, :, None]) * pitch).square().sum(-1)
                distance = distance.masked_fill(~valid_batch, float('inf'))
                nearest_distance, nearest_player = distance.min(2)
                contact = (nearest_distance < 2.5 ** 2) & missing
                if contact.any():
                    mass = attention.gather(2, nearest_player[:, :, None]).squeeze(2)
                    loss = loss - mass[contact].clamp_min(1e-6).log().mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step()
            total += float(loss.detach().cpu())
        losses.append(total / len(loader))
        print('epoch', epoch + 1, 'training loss', round(losses[-1], 3), 'seconds', round(time.monotonic() - began, 1), flush=True)
    # Fixed epoch count; the second half has not selected any checkpoint.
    scores = evaluate(net, heldout, team_tensor, device)
    report_path = ROOT / f'experiments/ball_reconstruction{suffix}_report.json'
    if args.evaluate:
        report = json.loads(report_path.read_text())
        assert report['checkpoint_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report['scores'] = scores
        report['caveats'][0] = 'One match per split; neither match-disjoint generalization nor target possession spellF1.'
        report['evaluation_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(json.dumps(scores, indent=2), flush=True)
        return
    CACHE.mkdir(exist_ok=True)
    source = 'Target first half excluding public window and 10s context; raw observed ball only' if args.adapt_target else 'IDSSE J03WOH first half; continuous live windows only'
    torch.save({'state_dict': net.cpu().state_dict(), 'epochs': args.epochs, 'seed': 7,
                'source': source}, checkpoint)
    report = {'protocol': source + '; second-half test; nonoverlapping test windows; no possession inputs',
        'epochs': args.epochs, 'seed': 7, 'torch': torch.__version__, 'device': str(device),
        'length': LENGTH, 'stride': STRIDE, 'training_starts': len(starts[1]),
        'training_loss': losses, 'scores': scores,
        'training_objective': 'Masked ADE metres' + (' + unit-weight nearest-player NLL within 2.5m (geometric proxy, not control labels)' if args.contact_supervision else ''),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'source_checkpoint_sha256': hashlib.sha256(source_checkpoint.read_bytes()).hexdigest() if source_checkpoint else None,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'caveats': ['One match per split; neither match-disjoint generalization nor target possession spellF1.',
                    '90percent denotes one centered contiguous gap, not random frame masking.',
                    'Fully hidden baseline is player centroid; interpolation is undefined without anchors.']}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(scores, indent=2), flush=True)


if __name__ == '__main__':
    main()
