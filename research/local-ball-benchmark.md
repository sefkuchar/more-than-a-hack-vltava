# Local ball reconstruction benchmark

The local experiment improves coordinate reconstruction for long artificial gaps,
but **has not improved held-out possession spellF1**. Production remains the
77-feature classifier with nested probability calibration, duration decoding and
observed-contact refinement: **0.6429 spellF1 / 0.8919 macroF1**. Neural ball reconstruction is
experimental. The 0.65 held-out public milestone remains unmet.

The external data and experiment checkpoints described below were removed during
cleanup. These are historical measurements; the packaged model requires none of
those assets. See [MODEL.md](../MODEL.md) for the runnable model and comparator.

## Setup and scope

The optional environment is `.venv-ball`, using **PyTorch 2.14.1** and
[requirements-ball.txt](../requirements-ball.txt). CPU execution works inside the
sandbox. MPS was checked with access outside the sandbox; the recorded training
runs used MPS. Device availability depends on the executing process, so CPU is
the portable fallback.

[ball_reconstruction.py](../experiments/ball_reconstruction.py) implements a small
masked-ball regressor using player geometry, pooled team context, bidirectional
sequence context and visible ball anchors. Player slots are not persistent model
tokens; slot continuity is used only to estimate short motion differences. Inputs
carry visibility and anchor-age information. Hidden ball coordinates are excluded
from every ball-derived input. The model has no possession-label inputs.

This is an original engineering experiment inspired by
[Ball Radar](https://arxiv.org/html/2306.08206) and the context/sequence modelling in
[PathCRF](https://arxiv.org/html/2602.12080v1), **not a reproduction** of either
architecture, training data or published score. See
[ballradar-review.md](ballradar-review.md) and
[ball-imputation-review.md](ball-imputation-review.md) for the primary-source review.

## External trajectory benchmark

Source: public IDSSE match **J03WOH**, Fortuna Düsseldorf–SSV Jahn Regensburg,
2022-08-26. The data are attributed to Bassek et al./DFL/Sportec, CC-BY-4.0:
[dataset paper](https://doi.org/10.1038/s41597-025-04505-y),
[public mirror](https://huggingface.co/datasets/pysport/idsse-data), and
[local ingestion/attribution notes](external-training-data.md).

Training uses first-half windows whose entire original 25-Hz extent is live.
Testing uses **119 nonoverlapping second-half windows**. Coordinates are sampled
every four frames, giving 6.25 Hz; each window contains 61 observations, spanning
9.6 seconds. There are 6,576 candidate training starts. Training ran for a fixed
12 epochs with seed 7; second-half scores did not select the checkpoint.

The masks are centered contiguous gaps at evaluation time, labelled approximately
1 second, 5 seconds, 90% of the window, and fully hidden. Training randomizes gap
locations. ADE below is the mean of each window's mean Euclidean error in metres,
on hidden coordinates only. It is not a possession metric.

| Mask | Neural ADE | Interpolation ADE | Constant-velocity ADE |
|---|---:|---:|---:|
| ~1 s | 1.816 m | **0.874 m** | 1.743 m |
| ~5 s | **4.552 m** | 6.609 m | 17.571 m |
| 90% contiguous | **5.762 m** | 11.809 m | 41.022 m |
| Fully hidden | **7.015 m** | 20.314 m* | 20.314 m* |

\* With no anchors, interpolation and constant velocity are undefined. Both
baseline columns explicitly fall back to the player centroid in that condition.
Their equal scores are one fallback baseline, not two independent competitors.

The neural model helps on longer gaps in this one-match split; interpolation
remains stronger on one-second gaps. This does not establish transfer to other
matches or the target's actual missing ball intervals.

## Target adaptation and trajectory check

The external checkpoint was adapted for another fixed 12 epochs, seed 7, using
**raw observed target ball coordinates from the first half**, excluding the public
five-minute window and ten seconds of neighboring context. No target possession
labels train the imputer. The target has 1,968 candidate training starts.

Testing uses **177 nonoverlapping second-half windows**, requiring at least 50%
raw ball observations and at least 14 valid tracked players at every sampled
step. These target windows are not independently established live-only windows.
The artificial masks hide available raw observations; unavailable observations
are excluded from scoring.

| Mask | Adapted neural ADE | Interpolation ADE | Constant-velocity ADE |
|---|---:|---:|---:|
| ~1 s | 1.213 m | **0.844 m** | 1.722 m |
| ~5 s | **5.042 m** | 6.486 m | 16.801 m |
| 90% contiguous | **6.908 m** | 13.025 m | 27.444 m |
| Fully hidden | **8.444 m** | 19.775 m* | 19.775 m* |

\* Player-centroid fallback again applies to both baseline columns.

These errors measure reconstruction of **masked automatic observations**, which
can themselves be inaccurate. We have no coordinate ground truth for genuinely
unseen target ball positions. The visibility filter also selects easier windows
than the most severe real occlusions. An apparent gain here cannot be interpreted
as verified accuracy on the target's full missing 41%.

## Downstream possession result

[ball_features.py](../experiments/ball_features.py) preserves every observed ball
coordinate and keeps interpolation for gaps shorter than two seconds. It adds
seven reconstruction features to the 77 production features: per-team distance
and contact evidence, disagreement with interpolation, overlap spread and anchor
fraction. This produces 84 features.

The same five contiguous one-minute folds, ten-second training embargo, training
stride three and isolated full-context decoder are used. Each fold's own
label-excluded classifier predicts and decodes the context before its held-out
minute is retained. The imputer's training and feature construction are independent
of those public possession labels.

| Possession configuration | Held-out spellF1 | MacroF1 |
|---|---:|---:|
| Historical 77-feature baseline, before boundary refinement | **0.626667** | 0.878004 |
| External imputer features | 0.613333 | 0.875972 |
| Target-adapted imputer features | **0.626667** | 0.878746 |
| Contact-supervised variant, 91 features | 0.476923 | 0.865470 |

These comparisons use the historical decoder without the later physical boundary
refinement. Current production is **0.6400 / 0.880400** after that refinement;
the improvement comes from physical timing, not neural reconstruction. Applying
the identical refinement to the adapted imputer also scores 0.6400 spellF1,
so it offers no primary-metric gain over the simpler production model.

The adapted version ties the historical primary metric. Its approximately 0.00074 macroF1
increase does not establish a useful improvement on 17 public possessions, so it
has **not been adopted**. The held-out public 0.65 milestone remains unmet.
Repeated method comparisons on this sample still introduce selection optimism.

The last variant adds a nearest-player auxiliary loss where a hidden raw ball
observation is within 2.5 m of a player, then exposes team attention and its short
history as features. Those geometric targets are a contact proxy, not controlled
possession labels. Its second-half target ball ADE improves to 4.722 m on five-second
gaps, but its downstream possession score deteriorates, so it is rejected too.

## Reproduce and inspect

The measured reports are
[external reconstruction](../experiments/ball_reconstruction_report.json),
[target-adapted reconstruction](../experiments/ball_reconstruction_target_report.json),
[external possession features](../experiments/ball_possession_report.json), and
[adapted possession features](../experiments/ball_possession_target_report.json).
They record checkpoint and code hashes. Their measured revisions should be
distinguished from subsequent experimental code. The completed contact variant is
recorded separately in [its ball report](../experiments/ball_reconstruction_target_contact_report.json)
and [possession report](../experiments/ball_possession_target_contact_report.json).

```bash
python3 -m venv .venv-ball
.venv-ball/bin/python -m pip install -r requirements-ball.txt
.venv-ball/bin/python -m experiments.ball_reconstruction --selfcheck
.venv-ball/bin/python -m experiments.ball_reconstruction --device cpu --epochs 12
.venv-ball/bin/python -m experiments.ball_reconstruction --device cpu --adapt-target --epochs 12
.venv-ball/bin/python -m experiments.ball_reconstruction --device cpu --evaluate
.venv-ball/bin/python -m experiments.ball_reconstruction --device cpu --adapt-target --evaluate
.venv-ball/bin/python -m experiments.ball_features --device cpu
.venv-ball/bin/python -m experiments.ball_features --device cpu --adapt-target
.venv-ball/bin/python -m experiments.ball_reconstruction --device cpu --adapt-target --contact-supervision
.venv-ball/bin/python -m experiments.ball_features --device cpu --adapt-target --contact-supervision
```

Use `--device mps` instead only in a process with working MPS access. The external
IDSSE data and target tracking must be present to train; downstream possession
validation also uses the supplied Smart Tagging/broadcast inputs and public labels.
Checkpoints and reconstructed-feature caches live under ignored
`experiments/cache/`; `.venv-ball` and raw data are also ignored. They are local
artifacts, not bundled pretrained models. Re-running a changed architecture can
produce a different checkpoint and does not reproduce an older report by name
alone; compare the recorded hashes and retain the corresponding source revision.
