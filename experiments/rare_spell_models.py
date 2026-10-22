"""Three fixed capacity/spell-weight ablations with nested matching learners."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from evaluation import possession_metrics as pm
from experiments.segment_weighting import spell_weights
from possession.calibration import calibrated, fit_calibration, blend_teams
from possession.control import control_onsets, snap_control_boundaries
from possession.duration import decode, PRIOR_PATH
from possession.features import build_features
from possession.model import predict, swap_teams

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'experiments/cache/rare_spell_models.npz'
REPORT = ROOT / 'experiments/rare_spell_models_results.json'
VARIANTS = {
    'larger': {'weighted': False, 'max_depth': 4, 'max_leaf_nodes': 15, 'min_samples_leaf': 15},
    'sqrt_spells': {'weighted': True, 'max_depth': 3, 'max_leaf_nodes': 7, 'min_samples_leaf': 35},
    'larger_sqrt_spells': {'weighted': True, 'max_depth': 4, 'max_leaf_nodes': 15, 'min_samples_leaf': 15},
}


def training_rows(y, allowed, weighted):
    allowed = np.asarray(allowed, dtype=bool)
    if allowed.shape != y.shape or not np.isin(y[allowed], (0, 1, 2)).all():
        raise ValueError('Expected one allowed-row mask and H/A/D training labels')
    sample = np.flatnonzero(allowed)[::3]
    weights = spell_weights(y, allowed)[sample] if weighted else np.ones(len(sample))
    if len(sample):
        weights /= weights.mean()
    return sample, weights


def fit_learner(x, y, names, allowed, variant):
    sample, weights = training_rows(y, allowed, variant['weighted'])
    if set(np.unique(y[sample])) != {0, 1, 2}:
        raise ValueError('Training requires all three classes')
    model = HistGradientBoostingClassifier(
        max_iter=120, learning_rate=.07, l2_regularization=10, random_state=0,
        **{key: value for key, value in variant.items() if key != 'weighted'})
    train = x[sample]
    model.fit(np.concatenate([train, swap_teams(train, names)]),
              np.concatenate([y[sample], np.where(y[sample] < 2, 1-y[sample], y[sample])]),
              sample_weight=np.tile(weights, 2))
    return model, {'training_frames': len(sample), 'weight_min': float(weights.min()),
                   'weight_max': float(weights.max()),
                   'training_rows_sha256': hashlib.sha256(sample.tobytes()).hexdigest()}


def nested_calibration(x, y, names, allowed, variant):
    probabilities, labels, folds = [], [], []
    for inner, ids in enumerate(np.array_split(np.arange(len(y)), 5)):
        a, b = int(ids[0]), int(ids[-1]) + 1
        valid = ids[allowed[ids]]
        if not len(valid):
            continue
        train = allowed.copy()
        train[max(0, a-250):min(len(y), b+250)] = False
        sample, _ = training_rows(y, train, variant['weighted'])
        assert not np.any(np.isin(sample, valid)) and np.all(allowed[sample])
        if set(np.unique(y[sample])) != {0, 1, 2}:
            continue
        model, details = fit_learner(x, y, names, train, variant)
        probabilities.append(predict(model, x[valid], names))
        labels.append(y[valid])
        folds.append({'inner_fold': inner, 'calibration_frames': len(valid), **details})
    if not probabilities:
        raise ValueError('No valid nested calibration examples')
    target = np.concatenate(labels)
    parameters, objective = fit_calibration(np.concatenate(probabilities), target)
    return parameters, {'calibration_frames': len(target), 'objective': objective, 'folds': folds}


def selfcheck():
    y = np.repeat(np.arange(3), 60).astype(np.int8)
    allowed = np.ones(len(y), bool)
    allowed[55:78] = False
    alternate = y.copy()
    alternate[~allowed] = (alternate[~allowed] + 1) % 3
    for weighted in (False, True):
        sample, weights = training_rows(y, allowed, weighted)
        other_sample, other_weights = training_rows(alternate, allowed, weighted)
        assert np.all(allowed[sample]) and np.array_equal(sample, other_sample)
        assert np.array_equal(weights, other_weights) and np.isclose(weights.mean(), 1)
    rng = np.random.RandomState(7)
    x = rng.normal(size=(len(y), 3))
    names = ['contact_H', 'contact_A', 'motion']
    for variant in VARIANTS.values():
        model, _ = fit_learner(x, y, names, allowed, variant)
        p = predict(model, x, names)
        assert np.allclose(p, predict(model, swap_teams(x, names), names)[:, [1, 0, 2]])
        assert np.isfinite(p).all() and np.allclose(p.sum(axis=1), 1)


def main():
    selfcheck()
    paper, paper_names, raw = build_features(broadcast=True, paper_evidence=True)
    assert len(paper_names) == 85
    base_names = paper_names[:77]
    stable = paper[:, [paper_names.index('stable_contact_H'), paper_names.index('stable_contact_A')]]
    onsets = control_onsets(raw['nearest'], stable, raw['observed'], raw['kick_accel'])
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    runs = pm.valid_runs(gt)
    if len(runs) != 1:
        raise ValueError('Expected the single contiguous public window')
    start, stop = runs[0]
    y = gt[start:stop]
    cores = [(paper[start:stop, :77], base_names), (paper[start:stop], paper_names)]
    fields = ('spell_f1_mean', 'macro_f1', 'n_pred_spells', 'per_class', 'spell_f1')
    results, arrays = {}, {}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    report = {'protocol': 'Five contiguous folds, 250-frame outer/inner embargo, stride3, matching learner in every inner calibration; each fold predicts/decodes full public7500 context before heldout rows retained.',
              'variants_predeclared': VARIANTS,
              'weighting': 'Existing sqrt(250/allowed contiguous segment length), clipped.3..5, sampled weights normalized to mean1; includes H/A/D, masks censor segments.',
              'full_fit_warning': 'Public-context resubstitution diagnostic only; not held-out and not the full-match artifact/official run.',
              'results': results,
              'labels_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
              'base_features_sha256': hashlib.sha256(cores[0][0].tobytes()).hexdigest(),
              'paper_features_sha256': hashlib.sha256(cores[1][0].tobytes()).hexdigest(),
              'prior_sha256': hashlib.sha256(PRIOR_PATH.read_bytes()).hexdigest(),
              'source_sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in (
                  'experiments/rare_spell_models.py', 'experiments/segment_weighting.py',
                  'possession/features.py', 'possession/calibration.py', 'possession/control.py',
                  'possession/physics.py', 'possession/model.py', 'possession/duration.py')}}

    def fit_pair(allowed, variant):
        pair, details, learners = [], [], []
        for x, names in cores:
            model, fit_details = fit_learner(x, y, names, allowed, variant)
            parameters, calibration = nested_calibration(x, y, names, allowed, variant)
            pair.append(calibrated(predict(model, x, names), parameters))
            details.append({'feature_count': len(names), 'parameters': parameters.tolist(),
                            'inner_calibration': calibration, **fit_details})
            learners.append({'classifier': model, 'features': names, 'calibration': parameters})
        return pair, details, learners

    for kind, variant in VARIANTS.items():
        pair, details, learners = fit_pair(np.ones(len(y), bool), variant)
        probability = blend_teams(*pair)
        decoded, _ = snap_control_boundaries(decode(probability), onsets, start)
        score = pm.evaluate(y, decoded)
        model_path = CACHE.with_name('rare_spell_models_' + kind + '.joblib')
        joblib.dump(learners, model_path, compress=3)
        results[kind] = {'full_public_fit_context': {key: score[key] for key in fields},
                         'full_public_fit_learners': details,
                         'model_cache': str(model_path.relative_to(ROOT)),
                         'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest()}
        arrays[kind + '_full_fit_probability'] = probability
        arrays[kind + '_full_fit_prediction'] = decoded
        np.savez_compressed(CACHE, **arrays)
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print(kind, 'PUBLIC_CONTEXT_RESUB', {key: score[key] for key in fields[:3]}, flush=True)
        prediction = np.empty(len(y), np.int8)
        probabilities, components, folds = [], [], []
        for fold, ids in enumerate(np.array_split(np.arange(len(y)), 5)):
            a, b = int(ids[0]), int(ids[-1]) + 1
            allowed = np.ones(len(y), bool)
            allowed[max(0, a-250):min(len(y), b+250)] = False
            pair, details, _ = fit_pair(allowed, variant)
            probability = blend_teams(*pair)
            decoded, changes = snap_control_boundaries(decode(probability), onsets, start)
            prediction[ids] = decoded[ids]
            probabilities.append(probability)
            components.append(pair)
            folds.append({'fold': fold, 'learners': details, 'refined_boundaries_full_context': len(changes)})
            print(kind, 'fold', fold, 'complete', flush=True)
        score = pm.evaluate(y, prediction)
        results[kind].update({'held_out': {key: score[key] for key in fields}, 'folds': folds})
        arrays[kind + '_prediction'] = prediction
        arrays[kind + '_probability'] = np.array(probabilities)
        arrays[kind + '_component_probability'] = np.array(components)
        np.savez_compressed(CACHE, **arrays)
        REPORT.write_text(json.dumps(report, indent=2) + '\n')
        print(kind, 'HELD_OUT', {key: score[key] for key in fields[:3]}, flush=True)


if __name__ == '__main__':
    main()
