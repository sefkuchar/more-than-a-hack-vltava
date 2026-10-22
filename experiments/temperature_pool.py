"""Fixed duration-temperature candidates and MBR-inspired spell consensus."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from evaluation import possession_metrics as pm
from possession.calibration import blend_teams, calibrated
from possession.control import control_onsets, snap_control_boundaries
from possession.duration import duration_decode, prior_tables, PRIOR_PATH
from possession.features import build_features
from possession.model import predict
from possession.train import MODEL_PATH

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'experiments/cache'
TEMPERATURES = (5., 2., 1., 10.)


def consensus(candidates):
    """Finite-support span MBR adaptation; no labels enter candidate selection."""
    unique = []
    for candidate in candidates:
        if not any(np.array_equal(candidate, old) for old in unique):
            unique.append(candidate)
    utility = [np.mean([pm.evaluate(reference, candidate)['spell_f1_mean']
                       for reference in unique]) for candidate in unique]
    return unique[int(np.argmax(utility))]


def main():
    candidate = np.r_[np.zeros(20, np.int8), np.ones(20, np.int8)]
    assert np.array_equal(consensus([candidate, candidate]), candidate)
    x, names, raw = build_features(broadcast=True, paper_evidence=True)
    bundle = joblib.load(MODEL_PATH)
    base = x[:, [names.index(name) for name in bundle['features']]]
    p = calibrated(predict(bundle['classifier'], base, bundle['features']), bundle['calibration'])
    other = bundle['paper_blend']
    q = calibrated(predict(other['classifier'], x, names), other['calibration'])
    training = blend_teams(p, q)
    stable = x[:, [names.index('stable_contact_H'), names.index('stable_contact_A')]]
    onsets = control_onsets(raw['nearest'], stable, raw['observed'], raw['kick_accel'])
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(gt)[0]
    y = gt[start:stop]
    with np.load(CACHE / 'paper_evidence.npz') as z:
        folds = np.stack([blend_teams(p, q) for p, q in zip(z['baseline_probability'], z['camera_probability'])])
    density, survival = prior_tables(json.loads(PRIOR_PATH.read_text()), len(y))
    times = np.arange(len(y) + 1)
    outputs = {str(t): np.empty(len(y), np.int8) for t in TEMPERATURES}
    outputs['consensus'] = np.empty(len(y), np.int8)
    train_predictions = {}
    for fold, probability in enumerate([training[start:stop], *folds]):
        pool = []
        for temperature in TEMPERATURES:
            decoded = duration_decode(np.log(probability.clip(1e-5, 1)) / temperature, density, survival, times)
            decoded, _ = snap_control_boundaries(decoded, onsets, start)
            pool.append(decoded)
            if fold == 0:
                train_predictions[str(temperature)] = decoded
            else:
                ids = np.array_split(np.arange(len(y)), 5)[fold - 1]
                outputs[str(temperature)][ids] = decoded[ids]
        chosen = consensus(pool)
        if fold == 0:
            train_predictions['consensus'] = chosen
        else:
            outputs['consensus'][ids] = chosen[ids]
    results = {}
    for kind, predictions in (('public_training_context', train_predictions), ('held_out', outputs)):
        results[kind] = {}
        for name, prediction in predictions.items():
            score = pm.evaluate(y, prediction)
            results[kind][name] = {key: score[key] for key in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')}
        print(kind, json.dumps(results[kind], indent=2), flush=True)
    report = {'results': results, 'temperatures': TEMPERATURES,
              'method': 'Fixed temperature pool; unique-path medoid using mean official spellF1 as an MBR-inspired span utility. No labels in consensus.',
              'selection_target': 'Public training spellF1 >=.75; held-out scores separately reported, not an untouched test set.',
              'caveat': 'Training diagnostics decode the full public7500context, not the whole half; final artifact requires official full submission score.',
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()}
    (ROOT / 'experiments/temperature_pool_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(CACHE / 'temperature_pool.npz', **outputs, **{'training_' + k: v for k, v in train_predictions.items()})


if __name__ == '__main__':
    main()
