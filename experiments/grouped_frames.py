"""Classify fixed short frame groups with soft labels, then recover frame labels."""

import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluation import possession_metrics as pm
from experiments.physical_boundary_snap import control_onsets, snap
from possession.duration import decode
from possession.features import build_features
from possession.model import swap_teams, predict


ROOT = Path(__file__).resolve().parents[1]


def group_frames(x, size):
    edges = np.r_[np.arange(0, len(x), size), len(x)]
    values = np.array([x[a:b].mean(0) for a, b in zip(edges[:-1], edges[1:])])
    return values.astype(np.float32), edges


def training_groups(values, edges, y, train):
    xs, ys, weights = [], [], []
    for group, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if not train[a:b].all():
            continue
        counts = np.bincount(y[a:b], minlength=3)
        for label in range(3):
            if counts[label]:
                xs.append(values[group])
                ys.append(label)
                # Match the total supervision weight of the existing stride-3 fit.
                weights.append(counts[label] / 3)
    return np.array(xs), np.array(ys), np.array(weights)


def fit_groups(x, y, weight, names, size):
    if set(np.unique(y)) != {0, 1, 2}:
        raise ValueError('Grouped training requires H/A/D examples')
    model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=7, max_depth=3,
        learning_rate=.07, l2_regularization=10, min_samples_leaf=max(2, round(105 / size)), random_state=0)
    model.fit(np.concatenate([x, swap_teams(x, names)]),
        np.concatenate([y, np.where(y < 2, 1 - y, y)]), sample_weight=np.tile(weight, 2))
    return model


def selfcheck():
    x = np.arange(60).reshape(20, 3)
    values, edges = group_frames(x, 4)
    y = np.tile([0, 1, 2, 0], 5)
    train = np.ones(20, bool)
    train[7:14] = False
    other = y.copy()
    other[~train] = (other[~train] + 1) % 3
    first = training_groups(values, edges, y, train)
    second = training_groups(values, edges, other, train)
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    assert np.isclose(first[2].sum(), 8 / 3)
    print('Whole-group embargo and soft-target weight checks passed', flush=True)


def main():
    selfcheck()
    x, names, raw = build_features(broadcast=True)
    stable = x[:, [names.index('stable_contact_H'), names.index('stable_contact_A')]]
    onsets = control_onsets(raw['nearest'], stable, raw['observed'])
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    core, y = x[start:stop], gt[start:stop]
    results, saved = [], {}
    for size in (12, 25):
        grouped, edges = group_frames(core, size)
        out = {name: np.empty(len(y), np.int8) for name in ('duration', 'contact_snap')}
        contexts, samples = [], []
        for fold in range(5):
            a, b = fold * 1500, (fold + 1) * 1500
            train = np.ones(len(y), bool)
            train[max(0, a - 250):min(len(y), b + 250)] = False
            tx, ty, weights = training_groups(grouped, edges, y, train)
            model = fit_groups(tx, ty, weights, names, size)
            probability = predict(model, grouped, names)
            probability = np.repeat(probability, np.diff(edges), axis=0)
            contexts.append(probability)
            base = decode(probability)
            refined, _ = snap(base, onsets, start)
            out['duration'][a:b] = base[a:b]
            out['contact_snap'][a:b] = refined[a:b]
            samples.append({'fold': fold, 'weighted_training_frames': float(weights.sum() * 3),
                            'nonzero_group_class_pairs': len(ty)})
            print('Grouped frames', size, 'isolated fold', fold, flush=True)
        for name, prediction in out.items():
            score = pm.evaluate(y, prediction)
            row = {'group_frames': size, 'method': name,
                **{key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')}}
            results.append(row)
            print(row, flush=True)
            saved[f'group{size}_{name}'] = prediction
        saved[f'group{size}_fold_probability'] = np.array(contexts)
    report = {'protocol': 'Five contiguous held-out minutes, 10s embargo; train only groups entirely inside allowed training frames; soft class counts; same-fold full-context decoding at25Hz',
        'grouping': 'Fixed nonoverlapping groups of12/25 frames; mean of77 features; no labels in group construction',
        'group_targets': 'Each class repeated with weight count/3; mixed-label groups remain soft; minleaf rescales105rawframes/groupwidth',
        'decoder': 'External duration temperature5; optional unchanged contact-onset snap',
        'results': results, 'feature_sha256': hashlib.sha256(core.tobytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'caveat': 'Repeated exploration on17publicpossessions; group edges constrain the initial boundary resolution.'}
    (ROOT / 'experiments/grouped_frames_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(ROOT / 'experiments/cache/grouped_frames_validation.npz', **saved)


if __name__ == '__main__':
    main()
