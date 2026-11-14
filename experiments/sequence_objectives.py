"""Fixed live-only MAP, contact-reward and posterior spell-risk decoders.

No target labels enter these objectives. Public training scores select among the
fixed methods; cached five-fold excluded predictions provide an honesty check.
"""

import hashlib
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
from numba import njit

from evaluation import possession_metrics as pm
from possession.calibration import blend_teams, calibrated
from possession.control import control_onsets, snap_control_boundaries
from possession.duration import decode, prior_tables
from possession.features import build_features
from possession.model import predict

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / 'experiments/cache'
SAMPLES = 32


@njit(cache=True)
def path_tables(evidence, density, survival, bonus, left_censored, right_censored):
    """Exact two-team semi-Markov max/sum-product, alternating only at H/A edges."""
    n = len(evidence)
    prefix = np.zeros((n + 1, 2))
    for t in range(n):
        prefix[t + 1] = prefix[t] + evidence[t]
    best = np.full((n + 1, 2), -np.inf)
    total = np.full((n + 1, 2), -np.inf)
    back = np.zeros((n + 1, 2), np.int32)
    for stop in range(1, n + 1):
        for k in range(2):
            for start in range(stop):
                censored = (start == 0 and left_censored) or (stop == n and right_censored)
                prior = survival[k, stop - start] if censored else density[k, stop - start]
                local = prefix[stop, k] - prefix[start, k] + prior
                if start:
                    local += bonus[start, k] - np.log(2.)
                    candidate = best[start, 1-k] + local
                    summed = total[start, 1-k] + local
                else:
                    candidate = summed = local
                if candidate > best[stop, k]:
                    best[stop, k] = candidate
                    back[stop, k] = start
                total[stop, k] = np.logaddexp(total[stop, k], summed)
    labels = np.empty(n, np.int8)
    if n:
        k, stop = np.argmax(best[n]), n
        while stop:
            start = back[stop, k]
            labels[start:stop] = k
            k, stop = 1-k, start
    return labels, prefix, total, best


def sample_paths(prefix, total, density, survival, bonus, left_censored, right_censored, rng):
    n = len(prefix) - 1
    result = np.empty((SAMPLES, n), np.int8)

    def draw(logweights):
        weights = np.exp(logweights - np.max(logweights))
        return int(rng.choice(len(weights), p=weights / weights.sum()))

    for row in result:
        k, stop = draw(total[n]), n
        while stop:
            starts = np.arange(stop)
            duration = stop - starts
            censored = ((starts == 0) & left_censored) | ((stop == n) & right_censored)
            prior = np.where(censored, survival[k, duration], density[k, duration])
            score = total[starts, 1-k] + prefix[stop, k] - prefix[starts, k] + prior
            score[1:] += bonus[starts[1:], k] - np.log(2.)
            score[0] = prefix[stop, k] + prior[0]
            start = draw(score)
            row[start:stop] = k
            k, stop = 1-k, start
    return result


def fixed_dead_paths(probability, reference, density, survival, onsets, bonus, rng=None):
    """Keep every D frame; use complete duration priors at known stoppage edges."""
    n = len(reference)
    prediction = reference.copy()
    samples = np.tile(reference, (SAMPLES, 1)) if rng is not None else None
    edges = np.flatnonzero(np.diff(np.r_[False, reference < 2, False].astype(np.int8)))
    for a, b in zip(edges[::2], edges[1::2]):
        evidence = np.log(probability[a:b, :2].clip(1e-5, 1)) / 5
        labels, prefix, total, _ = path_tables(evidence, density[:2], survival[:2], bonus[a:b], a == 0, b == n)
        prediction[a:b] = labels
        if rng is not None:
            samples[:, a:b] = sample_paths(prefix, total, density[:2], survival[:2], bonus[a:b], a == 0, b == n, rng)
    prediction, _ = snap_control_boundaries(prediction, onsets, 0)
    if samples is not None:
        for j in range(len(samples)):
            samples[j], _ = snap_control_boundaries(samples[j], onsets, 0)
    assert np.array_equal(prediction == pm.D, reference == pm.D)
    return prediction, samples


def spell_medoid(candidates):
    """Choose the candidate with highest posterior expected official spellF1."""
    spells = [pm.possession_spells(c, [(0, len(c))]) for c in candidates]
    expected = np.zeros(len(candidates))
    # The first candidate is MAP, not a posterior draw; it is not a pseudo truth.
    for i, prediction in enumerate(spells):
        for target in spells[1:]:
            overlaps = pm.spell_overlaps(target, prediction)
            denominator = len(target) + len(prediction)
            expected[i] += np.mean([2 * pm.match_spells(overlaps, len(target), len(prediction), t)[0]
                                   / denominator if denominator else 0 for t in pm.SPELL_IOUS])
    return candidates[int(np.argmax(expected))], expected / (len(candidates) - 1)


def selfcheck():
    rng = np.random.default_rng(4)
    for n in (1, 4):
        evidence = rng.normal(size=(n, 2))
        density = -rng.uniform(size=(2, n + 1))
        survival = -rng.uniform(size=(2, n + 1))
        bonus = rng.normal(size=(n, 2))
        for left, right in ((False, False), (True, False), (False, True), (True, True)):
            scores = []
            for seq in itertools.product(range(2), repeat=n):
                bounds = np.r_[0, np.flatnonzero(np.diff(seq)) + 1, n]
                score = sum(evidence[t, seq[t]] for t in range(n))
                for a, b in zip(bounds[:-1], bounds[1:]):
                    prior = survival if (a == 0 and left) or (b == n and right) else density
                    score += prior[seq[a], b-a]
                    if a:
                        score += bonus[a, seq[a]] - np.log(2.)
                scores.append(score)
            labels, prefix, total, best = path_tables(evidence, density, survival, bonus, left, right)
            assert np.isclose(best[n].max(), max(scores))
            assert np.isclose(np.logaddexp(*total[n]), np.logaddexp.reduce(scores))
            sampled = sample_paths(prefix, total, density, survival, bonus, left, right, rng)
            assert sampled.shape == (SAMPLES, n) and np.isin(sampled, (0, 1)).all()
            assert np.isin(labels, (0, 1)).all()
    candidates = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]], np.int8)
    chosen, _ = spell_medoid(candidates)
    assert np.array_equal(chosen, candidates[0])


def main():
    selfcheck()
    features, names, raw = build_features(broadcast=True, paper_evidence=True)
    stable = features[:, [names.index('stable_contact_H'), names.index('stable_contact_A')]]
    global_onsets = control_onsets(raw['nearest'], stable, raw['observed'], raw['kick_accel'])
    truth = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    start, stop = pm.valid_runs(truth)[0]
    y = truth[start:stop]
    onsets = [v[(v >= start) & (v < stop)] - start for v in global_onsets]
    model_path = ROOT / 'models/possession_model.joblib'
    bundle = joblib.load(model_path)
    base = features[start:stop, [names.index(k) for k in bundle['features']]]
    probability = calibrated(predict(bundle['classifier'], base, bundle['features']), bundle['calibration'])
    if bundle.get('paper_blend'):
        paper = bundle['paper_blend']
        core = features[start:stop, [names.index(k) for k in paper['features']]]
        probability = blend_teams(probability, calibrated(predict(paper['classifier'], core, paper['features']), paper['calibration']))
    with np.load(CACHE / 'paper_evidence.npz') as cache:
        heldout = np.stack([blend_teams(p, q) for p, q in zip(cache['baseline_probability'], cache['camera_probability'])])
    density, survival = prior_tables(bundle['duration_prior'], len(y))
    bonus = np.zeros((len(y), 2))
    for k in range(2):
        bonus[onsets[k], k] = np.log(2.)
    zeros = np.zeros_like(bonus)
    methods = ('reference', 'fixed_dead_map', 'directed_contact', 'posterior_spell_risk')
    outputs = {scope + '_' + method: np.empty(len(y), np.int8)
               for scope in ('training', 'heldout') for method in methods}
    details = []
    for fold, p in [(-1, probability), *enumerate(heldout)]:
        reference, _ = snap_control_boundaries(decode(p, bundle['duration_prior']), onsets, 0)
        map_path, sampled = fixed_dead_paths(p, reference, density, survival, onsets, zeros, np.random.default_rng(100 + fold))
        contact, _ = fixed_dead_paths(p, reference, density, survival, onsets, bonus)
        risk, expected = spell_medoid(np.vstack([map_path, sampled]))
        core = np.arange(len(y)) if fold < 0 else np.array_split(np.arange(len(y)), 5)[fold]
        scope = 'training' if fold < 0 else 'heldout'
        for method, prediction in zip(methods, (reference, map_path, contact, risk)):
            assert np.array_equal(prediction == pm.D, reference == pm.D)
            outputs[scope + '_' + method][core] = prediction[core]
        details.append({'fold': fold, 'posterior_samples': SAMPLES,
                        'risk_candidate': int(np.argmax(expected)), 'expected_spell_f1': float(expected.max()),
                        'changed_contact_frames_full_context': int(np.sum(contact != reference))})
        print(scope, fold, 'complete', flush=True)
    results = {}
    for key, prediction in outputs.items():
        score = pm.evaluate(y, prediction)
        results[key] = {k: score[k] for k in ('spell_f1_mean', 'macro_f1', 'n_pred_spells')}
        print(key, results[key], flush=True)
    report = {'results': results, 'details': details,
              'protocol': 'Training uses the saved full-public-fit classifier. Heldout uses cached 5 minute folds with 250-frame embargo and nested calibration; each fold decodes all 7500 frames before retaining its own minute. No labels enter inference objectives.',
              'methods': {'fixed_dead_map': 'Exact H/A semi-Markov MAP constrained to reference D; complete duration density at D boundaries, survival only at public window edges.',
                          'directed_contact': 'Same fixed-D MAP with log(2) new-team transition reward at existing sustained-or-SG7 kick onsets; no hard insertion or threshold sweep.',
                          'posterior_spell_risk': '32 exact semi-Markov H/A posterior samples, then select MAP or sampled sequence maximizing posterior expected official spellF1; all candidates receive unchanged physical refinement.'},
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'cache_sha256': hashlib.sha256((CACHE / 'paper_evidence.npz').read_bytes()).hexdigest(),
              'label_sha256': hashlib.sha256(y.tobytes()).hexdigest(),
              'limits': ['Public training scores are resubstitution, not generalization.',
                         'Method selection reuses the same 17 public spells.',
                         'Posterior risk is a 32-sample approximation with fixed D, so it cannot recover an incorrectly dead possession.',
                         'No manual labels or target segment timestamps enter decoding.']}
    (ROOT / 'experiments/sequence_objectives_results.json').write_text(json.dumps(report, indent=2) + '\n')
    np.savez_compressed(CACHE / 'sequence_objectives.npz', **outputs)


if __name__ == '__main__':
    main()
