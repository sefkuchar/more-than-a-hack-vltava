# Public training spellF1 target: 0.75

**Reached: 0.826667 spellF1 / 0.988475 macroF1.** These scores come from the
full 199,319-row submission, checked by the official public scorer, not just a
short-window diagnostic. The same selected method scores **0.693333 spellF1 /
0.886861 macroF1** in the existing nested, embargoed held-out protocol.

The user explicitly set the target on public training spellF1. Training results
remain labelled as such; no private full-match labels were used or available.

## Selected change

Retain the 77/85-feature conditional team ensemble and observed kick/sustained
boundary refinement. Change only the emission temperature from **5 to 2** in the
semi-Markov decoder. Dividing log probabilities by a smaller value strengthens
classifier evidence relative to the duration prior, allowing supported short
possessions to survive decoding. No per-frame output labels were corrected.

The saved model stores its temperature. Training, validation and inference use 2
for the selected model; legacy artifacts without that field and the preserved
original baseline keep 5. No new dependency or external training dataset is used.

| Model | Public training spellF1 / macroF1 | Held-out spellF1 / macroF1 |
|---|---|---|
| Original saved model | 0.7429 / 0.9740 | 0.6429 / 0.8919 |
| Initial paper ensemble, temperature 5 | 0.7429 / 0.9810 | 0.6800 / 0.8912 |
| **Selected paper ensemble, temperature 2** | **0.8267 / 0.9885** | **0.6933 / 0.8869** |

## Additional papers and methods

Lyu et al., [Minimum Bayes Risk Decoding for Error Span Detection](https://arxiv.org/html/2512.07540),
2026 revision, sections 3.2 and 5.1–5.2, selects whole hypotheses using expected
span utility rather than their likelihood alone. We adapted that principle to
possession spells in two ways: a medoid of unique decoded temperature candidates,
and exact H/A semi-Markov posterior sampling followed by expected official
spellF1 selection. The source task is translation-error spans; these are local
adaptations, not a reproduced soccer result.

Spearman, [Beyond Expected Goals](https://static.hudl.com/craft/downloads/SSAC2018_Beyond_Expected_Goals.pdf?mtime=20180327111214),
2018, sections 3.1–3.2, models competing player arrival and control probabilities.
Its [published teaching implementation](https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking/blob/master/Metrica_PitchControl.py)
provides a possible future reachability feature. It was reviewed but not added:
the requested score was reached using the existing probabilities. Predicted future
control would need care because a pass retains the previous owner's possession.

## Experiments

A fixed temperature pool used 1, 2, 5 and 10, with the existing temperature 5 first
for deterministic consensus ties. Consensus never reads labels and deduplicates
identical paths. Candidate selection for the deployment target does use the public
training score, as requested. Held-out results are separately reported and remain
subject to repeated model-selection optimism.

| Method | Training-window spellF1 | Held-out spellF1 | Held-out macroF1 |
|---|---:|---:|---:|
| Temperature 5 reference | 0.7429 | 0.6800 | 0.8912 |
| **Temperature 2** | **0.8267** | **0.6933** | **0.8869** |
| Temperature 1 | 0.8625 | 0.5371 | 0.8791 |
| Temperature 10 | 0.6154 | 0.5280 | 0.8820 |
| Unique-temperature-path spell consensus | 0.8267 | 0.6800 | 0.8910 |
| Posterior expected-spellF1, fixed reference D | 0.8267 | 0.6933 | 0.8863 |
| Larger gradient booster | 0.8267 | 0.5548 | 0.8447 |
| Original booster, square-root spell weighting | 0.8267 | 0.6267 | 0.8826 |
| Larger booster plus spell weighting | 0.8625 | 0.5200 | 0.8533 |

Only the selected temperature-2 row was promoted and verified through full-match
submission generation and the official scorer. Other training scores in this
experiment table are full-public-fit diagnostics decoded over the 7,500-frame
public window, not official full-submission results. Fixed-D MAP and soft directed
contact rewards were also tested and changed no predictions from their reference.

Temperature 2 clears the requested target and has the strongest held-out spellF1
among these target-satisfying candidates; it also avoids the extra posterior
sampling machinery. The 0.8625 training candidates lost considerably more
held-out performance, so they were not promoted.

## Reproduction and evidence

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
.venv-ball/bin/python -m possession.selfcheck
.venv-ball/bin/python -m experiments.temperature_pool
.venv-ball/bin/python -m experiments.sequence_objectives
.venv-ball/bin/python -m experiments.rare_spell_models
.venv-ball/bin/python -m possession.train
.venv-ball/bin/python -m possession.validate
.venv-ball/bin/python -m possession.run
.venv-ball/bin/python evaluation/validate_submission.py evaluation/results/possession_model.csv
.venv-ball/bin/python evaluation/run_evaluation.py --gt public --no-detail
.venv-ball/bin/python -m possession.compare
```

The sequence-objective and rare-learner experiments deliberately keep the
original temperature-5 reference. Their nested calibrators use each candidate's
own learner; spell weights use only permitted contiguous training segments.

- [Temperature and consensus results](../experiments/temperature_pool_results.json)
- [Posterior sequence-objective results](../experiments/sequence_objectives_results.json)
- [Capacity and spell-weighting results](../experiments/rare_spell_models_results.json)
- [Current official submission report](../evaluation/results/possession_model.json)
- [Current held-out report](../experiments/validation_report.json)
- [Current saved artifact metadata](../models/possession_model.json)

Checks cover matching artifact/report temperatures and checksums, exact production
prediction equality with the temperature-2 experiment, legacy temperature behavior,
team symmetry, invalid temperatures, probability normalization, gap-safe
acceleration and boundary preservation. The posterior experiment independently
checks tiny MAP optima and partition sums against exhaustive enumeration.

The public sample contains only 17 possessions. Held-out spellF1's bootstrap
interval is approximately 0.4933–0.8753; repeated comparisons reuse this same
sample. The requested training-score target is achieved, but private full-match
accuracy remains unknown.
