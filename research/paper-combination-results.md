# Added-paper experiments, 2026-10-22

The initial paper combination improves nested held-out public **spellF1 from 0.6429 to
0.6800**, with **macroF1 changing from 0.8919 to 0.8912**. Selection prioritizes
spellF1, the competition ranking metric. The default artifact and submission were
rebuilt; the previous model and its outputs are preserved separately.

The subsequent [training-target iteration](training-target-results.md) selects
temperature 2: **0.8267 public training spellF1 / 0.9885 macroF1**, with held-out
**0.6933 / 0.8869**. The comparisons below record the earlier temperature-5 phase.

## Papers inspected

| Paper | Transfer tested | Important limit |
|---|---|---|
| Link & Hoernig (2017), [Individual ball possession in soccer](../stuff/file.pdf), pp. 5–7 | Observed-ball acceleration, proximity-weighted kick evidence, motion variance, opponent pressure and one-second predicted separation | The method receives live/dead status; its player-level F1 is not challenge spellF1. A kick does not itself prove controlled recovery. |
| Vidal-Codina et al. (2022), [Automatic event detection in football using tracking data](../stuff/s12283-022-00381-6.pdf), pp. 4–5 | Seven-frame, degree-two Savitzky-Golay derivatives; contact changes can confirm short contact intervals and locate a gain at interval onset | Speed OR direction change confirms a contact in the described implementation. The supplied main PDF does not provide all supplement thresholds; this is an adaptation, not a claimed reproduction. |
| Borghesi et al. (2023), [Using Temporal Convolutional Networks to estimate ball possession in soccer games](../stuff/1-s2.0-S0957417423002816-main.pdf), pp. 4–9, 11–12 | Separate live/dead and conditional team reasoning; signed velocity, missing/visible objects, outside-pitch flags and nearest-five team context | Its per-object temporal network trains on multiple complete labelled matches. The local aggregate-feature TCN had already underperformed; no new network or weights were downloaded. |
| Casal et al. (2017), [Possession Zone as a Performance Indicator in Football](../stuff/fpsyg-08-01176.pdf), p. 4 | Control may be confirmed by repeated touches or teammate continuation; supports keeping contact evidence soft | This is manually coded possession-zone/success analysis, not an automated detector. Its win-probability regression is not an ownership prior. No zone or team-strength prior was added. |

The first two detector papers overlap earlier research notes. The new work is
implementing and testing their missing motion/control details, rather than adding
duplicate versions of existing proximity and duration features.

## Protocol

Use the existing five contiguous one-minute folds, 250-frame training embargo,
stride-three training, team mirroring and inner blocked probability calibration.
Each outer model predicts and decodes the same complete public context before its
held-out minute is retained. Decoder temperature, duration prior and search radius
stay fixed. Feature construction and broadcast alignment do not access labels.

The original model's 77-feature matrix and held-out predictions reproduce exactly.
The initial comparison used baseline, physics, camera, physics+camera, object
context and all features. Follow-up ablations split kick/control features and
combined physics with object context; equal blends and boundary refinement were
then tested. These choices therefore share the same small selection sample.

## Feature results

All rows below use separately nested calibration, the original duration decoder
and the original sustained-contact refinement.

| Classifier input | Features | spellF1 | macroF1 |
|---|---:|---:|---:|
| Original baseline | 77 | 0.6429 | 0.8919 |
| + kick dynamics | 86 | 0.6267 | 0.8848 |
| + predicted controllability / opponent pressure | 85 | 0.6429 | 0.8916 |
| + all physical additions | 94 | 0.6267 | 0.8842 |
| + soft cross-camera contact corrections | 85 | 0.6267 | 0.8726 |
| + physical and camera additions | 102 | 0.6286 | 0.8828 |
| + object context | 86 | 0.5931 | 0.8690 |
| + physical and object context | 103 | 0.5793 | 0.8644 |
| + all additions | 111 | 0.5037 | 0.8499 |

Appending every paper idea reduced performance. The only useful combination was
retaining the original classifier and combining its posterior with a separately
trained camera-evidence classifier.

| Combination | spellF1 | macroF1 |
|---|---:|---:|
| Equal full-probability base/camera blend, original refinement | 0.6800 | 0.8821 |
| Same blend, sustained + kick refinement | 0.6800 | 0.8888 |
| **Conditional team base/camera blend, sustained + kick refinement** | **0.6800** | **0.8912** |
| Conditional base/physics+camera blend, same refinement | 0.6800 | 0.8912 |
| Original classifier only, sustained + kick refinement | 0.6429 | 0.8995 |

The smaller camera model wins the tie with physics+camera. It uses eight new
features rather than 25. The 17 experimental physical classifier features and
nine object-context features remain available for reproducible ablations, but are
not appended to the selected classifier.

## Selected combination

Let `p` be the calibrated original model and `q` the calibrated camera model.
Keep `r_D = p_D`, average their conditional H/A distributions, then scale the result
by the original live probability:

```
r_(H,A) = (p_H + p_A) * 0.5 * (
    p_(H,A) / (p_H + p_A) + q_(H,A) / (q_H + q_A))
```

A zero-live-mass distribution uses a uniform conditional team fallback. No blend
weight was optimized: the two team distributions have equal weight. D stays with
the original model, whose dead-ball predictions were already stronger.

For boundary candidates, retain the existing eight-frame sustained-contact rule.
Also confirm an observed exclusive proximity interval if it contains supported
ball acceleration of at least 4 m/s². Calculate SG7/degree-two derivatives within
observed runs of at least seven frames and within each half; reject speed above
60 m/s or acceleration above 150 m/s². These gates are engineering choices for
this feed, not paper-reported optimal thresholds. Missing/unsupported acceleration
is zero. Only existing H/A boundaries may move, within 50 frames and without
crossing a neighbor. D boundaries, spell order and spell count remain unchanged by
refinement; additional spells come from the posterior ensemble and decoder.

## Reproduce and inspect

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
.venv-ball/bin/python -m possession.selfcheck
.venv-ball/bin/python -m experiments.object_context
.venv-ball/bin/python -m experiments.paper_evidence
.venv-ball/bin/python -m experiments.paper_boundaries
.venv-ball/bin/python -m possession.train
.venv-ball/bin/python -m possession.validate
.venv-ball/bin/python -m possession.run
.venv-ball/bin/python -m possession.compare
```

Reports:

- [All feature ablations and fixed blends](../experiments/paper_evidence_results.json)
- [Boundary and conditional-team combinations](../experiments/paper_boundaries_results.json)
- [Selected production validation](../experiments/validation_report.json)
- [Previous production validation](../experiments/baseline_validation_report.json)
- [Current saved artifact metadata](../models/possession_model.json)
- [Original 0.7429 public training-score report](../evaluation/results/possession_baseline.json)

Checks cover observed-run derivatives, gaps and outliers, empty/short sequences,
dummy tracks, true team-feature equivariance, normalized conditional fusion,
unchanged D mass, zero live probabilities, exhaustive tiny decoder optima and
boundary sequence/D preservation. Production features and selected held-out
predictions are byte-identical to the corresponding experiment. The original
classifier/calibrator in the new bundle also reproduces the preserved baseline.

## Interpretation

The selected output predicts 13 spells, compared with 11 previously and 17 in the
public reference. Its spell bootstrap interval is 0.4800–0.8667. With only 17 spells
and repeated method selection, this does not establish a statistically reliable
full-match gain. Public ball visibility also exceeds full-match visibility.

The **0.7429** previously displayed by the full submission was its score on public
labels used in training; that same model scored **0.6429 held out**. New held-out
results must be compared against 0.6429. Neither public training score measures
private full-match performance. No private labels or manual corrections were used.
