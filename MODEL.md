# Best possession model

The saved model is **[models/possession_model.joblib](models/possession_model.joblib)**.
It contains the trained classifier, probability calibrator, feature schema and
fitted duration prior. No external dataset or downloaded checkpoint is needed.

**Held-out public spellF1: 0.6429. MacroF1: 0.8919.** The strongest bundled baseline
scores 0.4242 spellF1 on the same public sample. The requested 0.65 milestone was
not reached; the original iteration goal is paused. Private full-match performance
is unknown. The organiser's 0.2433 baseline refers to that private full match.

## Run from this directory

The existing environment is local to this repository:

```bash
source .venv-ball/bin/activate
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python -m possession.run
```

This loads the saved model and writes:

- `evaluation/results/possession_model.csv`: all 199,319 frame labels.
- `evaluation/results/possession_model.json`: checksums and public training scores.

It needs the supplied tactical ATD, tactical metadata, Smart Tagging and broadcast
ATD under `data/`. Video is unnecessary. The fitted model is small and runs on CPU;
PyTorch and GPU access are unnecessary for the selected pipeline.

## Run the comparator

```bash
python -m possession.compare
```

This uses the official metric library to compare the saved **held-out** predictions
in `experiments/validation_predictions.csv` against all four bundled baselines.
Expected scores:

| Entry | Public spellF1 | MacroF1 |
|---|---:|---:|
| Held-out model | **0.6429** | **0.8919** |
| baseline_smooth125 | 0.4242 | 0.5812 |
| baseline_smooth50 | 0.3256 | 0.6207 |
| baseline_smooth25 | 0.2868 | 0.6200 |
| baseline_raw | 0.0378 | 0.5972 |

The original submission validator and scoreboard also work:

```bash
python evaluation/validate_submission.py evaluation/results/possession_model.csv
python evaluation/run_evaluation.py --gt public --no-detail
```

The original scoreboard scores the full submission on the public sample. That
submission's classifier was trained on the public labels, so its displayed public
score is a **training score**. Use `possession.compare` for the held-out comparison.

## Fresh environment and reproduction

All installation paths below remain inside the current directory:

```bash
python3 -m venv .venv-ball
.venv-ball/bin/python -m pip install -r requirements-model.txt
source .venv-ball/bin/activate
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
```

Python 3.11+ is required; development used Python 3.14. Dependencies are pinned.
The saved classifier requires scikit-learn 1.8.0.

```bash
python -m possession.train       # rebuild models/possession_model.joblib
python -m possession.validate    # recompute nested held-out predictions and report
python -m possession.run         # generate the full submission from the saved model
python -m possession.compare
```

Training and validation use only the allowed public possession labels and the
provided challenge inputs. Raw external data is unnecessary for these commands.
The model metadata is in `models/possession_model.json`; validation details and
bootstrap intervals are in `experiments/validation_report.json`.

## Method

1. Build 77 physical, tracking-quality, temporal and automatic-tagging features.
   Distances exclude dummy goalkeeper observations with missing speed. Ball
   interpolation carries explicit missingness flags.
2. Align broadcast tracking to tactical tracking using player geometry. Reliability
   gates reject replays and mismatched views. No possession labels enter alignment.
3. Fit shallow histogram gradient boosting with H/A team swapping during training
   and symmetric probability averaging at prediction time.
4. Calibrate probabilities with two parameters: a shared log-probability scale and
   a dead-ball intercept. Fit them on inner blocked held-out training predictions,
   minimizing regularized log loss. No outer held-out labels fit the calibrator.
5. Decode each half independently with an exact 25-Hz semi-Markov duration model,
   using emission temperature 5 and censored-edge survival probabilities. The small
   duration parameters were learned from an unrelated public IDSSE match and are
   retained inside the model artifact. Source/license attribution remains in
   [external-training-data.md](research/external-training-data.md).
6. Refine existing H/A boundaries to nearby sustained observed-contact onsets:
   new-team distance below 1.5 m, opponent beyond 3 m, motion stability above 0.2,
   at least eight frames, search within 50 frames. This preserves spell count,
   label sequence and every dead-ball boundary. It never inserts a possession.

The system uses offline future tracking context. No clock, frame number, event ID,
private labels or manually corrected output labels enter classification.

## Validation and limits

The five-minute public sample is divided into five contiguous one-minute folds,
with a ten-second training embargo on each side. Each outer fold fits its own
classifier. Its calibrator uses inner held-out predictions from the remaining
training data, with both embargoes enforced. That same outer model predicts,
decodes and refines the full context before its held-out minute is retained.

There are only **17 public possessions**. The possession bootstrap interval is
approximately **0.4286–0.8276**. Many configurations were compared on this sample;
selection optimism remains. Ball visibility is 88.2% in this sample versus about
59.3% over the full graded match, so the public result does not establish full-match
accuracy.

External datasets, downloaded research checkpoints, experiment caches and known
task-created temporary downloads outside this directory were removed. Historical
experiment scripts/reports remain for inspection; experiments that used those
external assets require downloading them again. The selected model does not.

Research notes:

- [Verified possession methods](research/verified-possession-methods.md)
- [Ball Radar review](research/ballradar-review.md)
- [Ball reconstruction papers](research/ball-imputation-review.md)
- [Local ball reconstruction benchmark](research/local-ball-benchmark.md)
- [Grouped-frame experiments](research/grouped-frame-results.md)
