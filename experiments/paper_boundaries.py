"""Test paper kick impulses as extra candidates for existing H/A boundaries."""

import hashlib
import json
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from experiments.paper_evidence import ROOT, CACHE, half_features
from possession.calibration import blend_teams
from possession.control import control_onsets, snap_control_boundaries
from possession.duration import decode
from possession.features import build_features, tracking
from possession.physics import paper_features


def team_blend(base, other):
    return np.stack([blend_teams(p, q) for p, q in zip(base, other)])


def main():
    x, names, raw = build_features(broadcast=True)
    physics, physics_names = half_features(paper_features, *tracking())
    stable = x[:, [names.index('stable_contact_H'), names.index('stable_contact_A')]]
    sustained = control_onsets(raw['nearest'], stable, raw['observed'])
    onsets = control_onsets(raw['nearest'], stable, raw['observed'], physics[:, physics_names.index('kick_accel')])
    source = ROOT / 'experiments/paper_evidence_results.json'
    previous = json.loads(source.read_text())
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    y = gt[start:stop]
    assert hashlib.sha256(y.tobytes()).hexdigest() == previous['labels_sha256']
    assert hashlib.sha256(x[start:stop].tobytes()).hexdigest() == previous['results']['baseline']['feature_sha256']
    with np.load(CACHE / 'paper_evidence.npz') as z:
        base = z['baseline_probability']
        variants = {'baseline': base, 'camera_blend': (base + z['camera_probability']) / 2,
                    'physics_camera_blend': (base + z['physics_camera_probability']) / 2,
                    'camera_team_blend': team_blend(base, z['camera_probability']),
                    'physics_camera_team_blend': team_blend(base, z['physics_camera_probability'])}
    results, predictions = {}, {}
    for kind, probabilities in variants.items():
        prediction = np.empty(len(y), np.int8)
        for fold, ids in enumerate(np.array_split(np.arange(len(y)), 5)):
            decoded, _ = snap_control_boundaries(decode(probabilities[fold]), onsets, start)
            prediction[ids] = decoded[ids]
        score = pm.evaluate(y, prediction)
        results[kind] = {k: score[k] for k in ('macro_f1', 'spell_f1_mean', 'n_pred_spells')}
        predictions[kind] = prediction
        print(kind, results[kind], flush=True)
    report = {'results': results, 'additional_kick_onsets': [len(np.setdiff1d(a, b)) for a, b in zip(onsets, sustained)],
              'method': 'Observed exclusive proximity interval contains a supported SG7 acceleration >=4m/s²; union with sustained onsets; unchanged +/-50 frame H/A-only refinement.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'feature_report_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'cache_sha256': hashlib.sha256((CACHE / 'paper_evidence.npz').read_bytes()).hexdigest(),
              'caveat': 'Follow-up selection on the same 17 public spells; impulse contact is not proof of controlled recovery.'}
    (ROOT / 'experiments/paper_boundaries_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(CACHE / 'paper_boundaries.npz', **predictions)


if __name__ == '__main__':
    main()
