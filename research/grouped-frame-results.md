# Grouping frames and predicting temporal labels

The user's proposed frame grouping and temporal prediction were evaluated with
five contiguous held-out minutes, a 10-second training embargo and the same
external-duration decoder. Each fold's own classifier predicts and decodes the
entire context before its held-out minute is retained. No decoder runs across
stitched out-of-fold emissions. Future unlabeled evidence is allowed offline.

The packaged model uses the **77-feature frame classifier, nested probability
calibration and physical refinement: 0.6429 spellF1 / 0.8919 macroF1**. The 0.65
milestone is unmet. Before calibration, physical refinement scored 0.6400 / 0.8804;
without either addition, the classifier scored 0.6267 / 0.8780.

| Fixed experiment | SpellF1 | MacroF1 |
|---|---:|---:|
| Groups of 12 frames, soft targets | 0.5333 | 0.8669 |
| Same, physical refinement | 0.5733 | 0.8729 |
| Groups of 25 frames, soft targets | 0.5067 | 0.8668 |
| Same, physical refinement | 0.5429 | 0.8649 |
| Adaptive physical groups | 0.4750 | 0.8082 |
| Same, physical refinement | 0.4500 | 0.8118 |
| Physical-feature graph label spreading | 0.4343 | 0.7398 |
| Equal graph/frame-classifier mixture | 0.5935 | 0.8483 |
| Temporal convolution label predictor | 0.5333 | 0.8650 |
| Equal temporal/frame-classifier mixture | 0.5467 | 0.8655 |
| External-pretrained temporal predictor | 0.4889 | 0.8412 |
| Equal transferred-temporal/frame mixture | 0.5481 | 0.8825 |

Fixed groups average the 77 input features over nonoverlapping 12-frame (0.48 s)
or 25-frame (1 s) intervals. Group boundaries use no labels. Training retains only
groups wholly inside the allowed training mask and preserves mixed-label groups
as class-count-weighted soft targets. Group emissions expand to 25 Hz for decoding.
This reduces temporal resolution and the number of independent training groups.

Adaptive grouping uses changes in standardized physical evidence sampled at 5 Hz:
a fixed RMS change threshold 0.75, minimum 0.4-second group and maximum 2 seconds.
It creates 204 groups in the public context. Grouping is label-free, but supervised
group labels still come only from allowed training intervals.

Graph label spreading connects similar physical feature vectors, including unlabeled
whole-match samples and team-swapped copies. It has no frame/time edges or truth
changepoints. Disconnected queries require a uniform fallback; it fails to recover
enough useful labels. This transductive method and its fixed equal-weight mixture
both underperform the frame classifier.

The temporal predictor is a 16-channel residual convolution at 5 Hz, with kernel 5
and dilations 1/3/9, trained for 200 updates with fold-specific normalization and
team symmetry. A separate variant pretrains on external J03WOH data, including
simulated ball corruption, then adapts for 50 updates within each training fold.
These predict labels from ordered feature sequences rather than group averages;
neither improves possession spells. Explicit ±0.4/1/2-second physical snapshots
were also rejected (0.4828 spellF1).

The chosen physical refinement shifts only existing direct H/A edges to nearby
observed, sustained new-team control onsets. It preserves spell count/order and
dead-ball frames. Its entire public gain comes from changing **20 held-out frames**.
It is implemented in [control.py](../possession/control.py), independently of the
rejected grouping and neural predictors.

Measured reports: [fixed groups](../experiments/grouped_frames_results.json),
[adaptive groups](../experiments/adaptive_groups_results.json),
[feature graph](../experiments/feature_graph_results.json),
[temporal predictor](../experiments/temporal_classifier_results.json),
[temporal transfer](../experiments/temporal_transfer_results.json), and
[physical refinement](../experiments/physical_boundary_snap_results.json).

Only 17 public possessions are available. Repeated configuration comparisons
create selection optimism despite label-excluded folds; bootstrap uncertainty is
wide and full-match accuracy remains unknown. These results reject these fixed
implementations, not grouping or temporal prediction in general.
