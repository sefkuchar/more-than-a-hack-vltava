"""Fixed paper-feature ablations with the production nested blocked protocol."""

import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from experiments.align_broadcast import aligned_broadcast
from experiments.object_context import context_features
from games import GAME, half_video_blocks
from possession.calibration import blocked_calibration, calibrated
from possession.control import control_onsets, snap_control_boundaries
from possession.duration import decode, PRIOR_PATH
from possession.features import build_features, rolling, tracking
from possession.model import fit_model, predict, swap_teams
from possession.physics import paper_features, selfcheck

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'experiments/cache'


def half_features(builder, players, ball, teams):
    out = None
    for _, start, end in half_video_blocks(GAME):
        values, names = builder(players[start:end + 1], ball[start:end + 1], teams)
        if out is None:
            out = np.zeros((len(ball), len(names)), dtype=np.float32)
        out[start:end + 1] = values
    return out, names


def camera_features(raw):
    """Soft cross-camera team corrections; never override a predicted label."""
    br = aligned_broadcast()
    seen = (br['broadcast_frame'] >= 0) & raw['observed']
    contact = np.exp(-.5 * (br['corrected_nearest'] / 1.5) ** 2) * seen[:, None]
    columns = [seen[:, None], br['team_correction_fraction'][:, None], contact]
    names = ['camera_mapped_ball', 'camera_team_correction', 'camera_contact_H', 'camera_contact_A']
    for radius in (12, 50):
        columns.append(rolling(contact, radius))
        names.extend([f'camera_contact_mean_{radius}_H', f'camera_contact_mean_{radius}_A'])
    return np.concatenate(columns, axis=1).astype(np.float32), names


def main():
    selfcheck()
    base, names, raw = build_features(broadcast=True)
    players, ball, teams = tracking()
    physics, physics_names = half_features(paper_features, players, ball, teams)
    context, context_names = half_features(context_features, players, ball, teams)
    camera, camera_names = camera_features(raw)
    variants = {
        'baseline': (base, names),
        'kick': (np.column_stack([base, physics[:, :9]]), names + physics_names[:9]),
        'control': (np.column_stack([base, physics[:, 9:]]), names + physics_names[9:]),
        'physics': (np.column_stack([base, physics]), names + physics_names),
        'camera': (np.column_stack([base, camera]), names + camera_names),
        'physics_camera': (np.column_stack([base, physics, camera]), names + physics_names + camera_names),
        'object_context': (np.column_stack([base, context]), names + context_names),
        'physics_context': (np.column_stack([base, physics, context]), names + physics_names + context_names),
        'combined': (np.column_stack([base, physics, camera, context]), names + physics_names + camera_names + context_names),
    }
    assert all(np.isfinite(x).all() for x, _ in variants.values())
    # All physical feature construction precedes access to possession labels.
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    runs = pm.valid_runs(gt)
    if len(runs) != 1:
        raise ValueError('Expected one contiguous public label window')
    start, stop = runs[0]
    y = gt[start:stop]
    stable = base[:, [names.index('stable_contact_H'), names.index('stable_contact_A')]]
    onsets = control_onsets(raw['nearest'], stable, raw['observed'])
    CACHE.mkdir(parents=True, exist_ok=True)
    results, arrays = {}, {}
    for kind, (features, feature_names) in variants.items():
        core = features[start:stop]
        assert np.array_equal(swap_teams(swap_teams(core, feature_names), feature_names), core)
        prediction = np.empty(len(y), np.int8)
        probabilities, folds = [], []
        for fold, ids in enumerate(np.array_split(np.arange(len(y)), 5)):
            a, b = int(ids[0]), int(ids[-1]) + 1
            allowed = np.ones(len(y), bool)
            allowed[max(0, a - 250):min(len(y), b + 250)] = False
            sample = np.flatnonzero(allowed)[::3]
            model = fit_model(core[sample], y[sample], feature_names)
            parameters, inner = blocked_calibration(core, y, feature_names, allowed)
            probability = calibrated(predict(model, core, feature_names), parameters)
            decoded, changes = snap_control_boundaries(decode(probability), onsets, start)
            prediction[a:b] = decoded[a:b]
            probabilities.append(probability)
            folds.append({'fold': fold, 'training_frames': len(sample),
                          'calibration_parameters': parameters.tolist(), 'inner_calibration': inner,
                          'refined_boundaries_full_context': len(changes)})
            print(kind, 'fold', fold, 'complete', flush=True)
        if kind == 'baseline':
            assert hashlib.sha256(prediction.tobytes()).hexdigest() == '2bd646213ae430a1160e25ba0c0d9bc4fbf3ba12cb445a9badad15eb6dd15a5a', 'Production baseline did not reproduce'
        score = pm.evaluate(y, prediction)
        fields = ('spell_f1_mean', 'macro_f1', 'n_pred_spells', 'per_class', 'spell_f1', 'confusion')
        results[kind] = {'metrics': {key: score[key] for key in fields},
                         'feature_count': len(feature_names), 'added_features': feature_names[len(names):],
                         'feature_sha256': hashlib.sha256(core.tobytes()).hexdigest(), 'folds': folds}
        arrays[kind + '_prediction'] = prediction
        arrays[kind + '_probability'] = np.array(probabilities)
        print(kind, {key: score[key] for key in fields[:3]}, flush=True)
        np.savez_compressed(CACHE / 'paper_evidence.npz', **arrays)
        report = {'protocol': 'Production: five contiguous minute folds; 250-frame embargo; inner blocked probability calibration; same-fold full-context duration decoding and unchanged contact refinement.',
                  'variants': list(variants), 'results': results,
                  'selection': 'Initial baseline/physics/camera/physics_camera/object_context/combined comparison, followed by kick/control/physics_context ablations and fixed equal probability blends on the same public sample.',
                  'labels_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
                  'prior_sha256': hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),
                  'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                      'experiments/paper_evidence.py', 'possession/physics.py', 'possession/features.py',
                      'possession/model.py', 'possession/calibration.py', 'possession/control.py',
                      'possession/duration.py', 'experiments/align_broadcast.py')},
                  'caveat': 'Same 17 public spells used for repeated method selection; no private full-match score.'}
        (ROOT / 'experiments/paper_evidence_results.json').write_text(json.dumps(report, indent=2) + '\n')
    blends = {}
    for kind in variants:
        if kind == 'baseline':
            continue
        prediction = np.empty(len(y), np.int8)
        probability = (arrays['baseline_probability'] + arrays[kind + '_probability']) / 2
        for fold, ids in enumerate(np.array_split(np.arange(len(y)), 5)):
            decoded, _ = snap_control_boundaries(decode(probability[fold]), onsets, start)
            prediction[ids] = decoded[ids]
        score = pm.evaluate(y, prediction)
        blends['baseline_' + kind] = {key: score[key] for key in fields}
        arrays['baseline_' + kind + '_blend_prediction'] = prediction
        print('baseline_' + kind + '_blend', {key: score[key] for key in fields[:3]}, flush=True)
    report['equal_calibrated_probability_blends'] = blends
    (ROOT / 'experiments/paper_evidence_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(CACHE / 'paper_evidence.npz', **arrays)


if __name__ == '__main__':
    main()
