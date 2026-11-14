"""Learn ordered possession context with a small dilated temporal classifier.

Fixed experiment: 5 Hz, 16 channels, 10.4 seconds receptive field, 200 updates.
The held-out minute and its embargo never contribute to training or scaling.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from evaluation import possession_metrics as pm
from possession.features import build_features
from possession.model import swap_teams
from possession.duration import decode


ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'experiments/cache'
STEP = 5


class TemporalClassifier(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.input = nn.Conv1d(features, 16, 1)
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(16, 16, 5, padding=2 * d, dilation=d),
                          nn.SiLU(), nn.Dropout(.2)) for d in (1, 3, 9)])
        self.head = nn.Conv1d(16, 3, 1)

    def forward(self, x):
        x = self.input(x).tanh()
        for block in self.blocks:
            x = x + block(x)
        return self.head(x)


def masked_loss(logits, labels, train):
    values = nn.functional.cross_entropy(logits, labels, reduction='none')
    return values[:, train].mean()


def selfcheck():
    torch.manual_seed(0)
    net = TemporalClassifier(4).eval()
    values = torch.randn(2, 4, 80)
    logits = net(values)
    labels = torch.randint(3, (2, 80))
    train = torch.ones(80, dtype=torch.bool)
    train[20:40] = False
    other = labels.clone()
    other[:, ~train] = (other[:, ~train] + 1) % 3
    assert torch.equal(masked_loss(logits, labels, train), masked_loss(logits, other, train))
    assert logits.shape == (2, 3, 80) and torch.isfinite(logits).all()
    print('Temporal classifier shapes and label-mask isolation passed', flush=True)


def train_fold(x, y, names, mask, device):
    torch.manual_seed(19)
    mirrored = swap_teams(x, names)
    # Fit normalization only on this fold's allowed rows, including team symmetry.
    selected = np.concatenate([x[mask], mirrored[mask]])
    center, scale = selected.mean(0), selected.std(0).clip(.1)
    inputs = np.stack([x, mirrored])
    inputs = np.clip((inputs - center) / scale, -6, 6).transpose(0, 2, 1)
    labels = np.stack([y, np.where(y < 2, 1 - y, y)])
    values = torch.tensor(inputs, dtype=torch.float32, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    train = torch.tensor(mask, device=device)
    net = TemporalClassifier(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=.0015, weight_decay=.01)
    for _ in range(200):
        net.train()
        optimizer.zero_grad(set_to_none=True)
        loss = masked_loss(net(values), labels, train)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite temporal training loss')
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()
    net.eval()
    with torch.no_grad():
        p = net(values).softmax(1).cpu().numpy().transpose(0, 2, 1)
    return (p[0] + p[1][:, [1, 0, 2]]) / 2, float(loss.detach().cpu())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    parser.add_argument('--selfcheck', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    selfcheck()
    if args.selfcheck:
        return
    x, names, _ = build_features(broadcast=True)
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    core, y = x[start:stop], gt[start:stop]
    with np.load(CACHE / 'booster_ensemble.npz') as z:
        hist = z['hist_fold_probability']
    previous = json.loads((ROOT / 'experiments/booster_ensemble_results.json').read_text())
    assert previous['feature_sha256'] == hashlib.sha256(core.tobytes()).hexdigest()
    assert hist.shape == (5, len(y), 3)
    predictions = {name: np.empty(len(y), np.int8) for name in ('temporal', 'equal_hist_temporal')}
    contexts, losses = [], []
    sampled = np.arange(0, len(y), STEP)
    began = time.monotonic()
    for fold in range(5):
        a, b = fold * 1500, (fold + 1) * 1500
        train = np.ones(len(y), bool)
        train[max(0, a - 250):min(len(y), b + 250)] = False
        p, loss = train_fold(core[sampled], y[sampled], names, train[sampled], torch.device(args.device))
        full = np.column_stack([np.interp(np.arange(len(y)), sampled, p[:, k]) for k in range(3)])
        contexts.append(full)
        losses.append(loss)
        for method, probability in [('temporal', full), ('equal_hist_temporal', (hist[fold] + full) / 2)]:
            predictions[method][a:b] = decode(probability)[a:b]
        print('Temporal isolated fold', fold, 'training CE', round(loss, 4), 'elapsed', round(time.monotonic() - began, 1), flush=True)
    results = []
    for name, prediction in predictions.items():
        score = pm.evaluate(y, prediction)
        row = {'method': name, **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')}}
        results.append(row)
        print(row, flush=True)
    report = {'protocol': 'Five contiguous minute folds, 10s embargo, training normalization and labels excluded; same-fold isolated full-context decoding',
        'model': '16-channel residual temporal convolution; k5 dilations1/3/9; 5Hz; 200 AdamW updates; team symmetric',
        'features': len(names), 'torch': torch.__version__, 'device': args.device,
        'training_losses': losses, 'results': results,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'feature_sha256': hashlib.sha256(core.tobytes()).hexdigest(),
        'label_sha256': hashlib.sha256(y.tobytes()).hexdigest()}
    (ROOT / 'experiments/temporal_classifier_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(CACHE / 'temporal_classifier_validation.npz', fold_probability=np.array(contexts), **predictions)


if __name__ == '__main__':
    main()
