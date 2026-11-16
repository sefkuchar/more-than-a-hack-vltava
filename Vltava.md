# Team Vltava: ball possession, technical document

**More than a Hack 2027, Phase 1** · Repository: <https://github.com/sefkuchar/more-than-a-hack-vltava>
· Submission: `Vltava.csv` (SHA-256 `0586b4350d28ce1420374d162a9fbbdbe5f87e5846caa2c46fb6e26451eb53a6`)

## 1. Approach

We predict per-frame H/A/D evidence with a classifier trained on **other games**. We then decode
whole possession sequences instead of smoothing frame labels. `spellF1` rewards recovering
complete possessions, and frame classifiers flicker. A duration-aware decoder trades the
evidence against how long possessions and stoppages really last.

The pipeline (`python -m possession.run`) has four steps.

1. **Features (label-free, 62 per frame).** They are computed from tactical tracking and pitch
   geometry:
   - distance from the ball to the nearest home and nearest away player, and the margin between them;
   - Gaussian "contact" scores and sustained co-motion contact;
   - ball visibility and the age of the last observation;
   - ball speed and trajectory turn, distance to the pitch boundary, and player movement;
   - rolling means over 5, 12, 25, 50 and 125 frames.

   The same code computes the same features on the external matches.
2. **Classifier.** A shallow `HistGradientBoostingClassifier` (`random_state=0`). Home and away
   are swapped during training and averaged at prediction, so the model has no team bias.
3. **Calibration.** A 2-parameter calibrator (probability scale and dead-ball intercept), fitted
   on a held-out external match.
4. **Semi-Markov decoding.** An exact 25 Hz explicit-duration decoder runs separately for each
   half. Its log-normal-mixture priors for possession and dead-ball durations were fitted on the
   converted IDSSE training labels. The emission temperature was chosen on the held-out external
   match.

## 2. Training data

We use no labels of the provided match (BAR-ATH): not for training, calibration, tuning or model
selection. The only labelled data is:

| Dataset | Source | Who labelled it | How much | Used for |
|---|---|---|---|---|
| IDSSE J03WOH (Düsseldorf–Regensburg, 2. Bundesliga 2022/23) | [IDSSE / DFL-Sportec open data](https://doi.org/10.1038/s41597-025-04505-y), CC-BY-4.0, [mirror](https://huggingface.co/datasets/pysport/idsse-data) | Provider (Sportec Solutions for the DFL): per-frame `BallStatus` and `BallPossession` in the official position data | 137,214 frames | Classifier training; duration prior |
| IDSSE J03WQQ (Düsseldorf–St. Pauli) | same | same | 142,345 frames | Classifier training; duration prior |
| IDSSE J03WN1 (Bundesliga 2022/23) | same | same | 141,561 frames | Classifier training; duration prior |
| IDSSE J03WPY (2. Bundesliga 2022/23) | same | same | 146,211 frames | Classifier training; duration prior |
| IDSSE J03WR9 (2. Bundesliga 2022/23) | same | same | 146,810 frames | Classifier training; duration prior |
| IDSSE J03WOY (Düsseldorf–Rostock) | same | same | 142,536 frames | Calibration and decoder settings only (held out from the classifier) |

Labels map as follows: `BallStatus = 0` → D; `BallStatus = 1` with `BallPossession` 1 or 2 → H or A.

**Conversion to controlled possession (a rule, not a fit).** The provider's `BallPossession`
flags every touch: about 400 live spells per match, mean 7 s, 27% of them under 2 s. The
challenge defines possession as *controlled* ("a player keeps the ball for a few seconds"), and
says touches, deflections and duels without control don't count. So we give any live spell
shorter than **2 s** back to the team in control before it, or after it when it follows a
stoppage (`possession/external.py`, `controlled_labels`). That changes about 2% of frames and
leaves about 280 spells per match with a mean of 11 s. The 2 s value comes from the written
definition. We adopted the conversion before running the pipeline check below and kept it
regardless of the result. The spell-duration prior is refitted on the converted training labels.

Training uses every 4th frame of each training match, twice: once with the clean ball track and
once with a corrupted copy. That gives 357,076 rows in total. The seventh IDSSE match, J03WMX, wasn't used: our file-transfer tooling couldn't move its 418 MB tracking file.

**Domain adaptation, no labels involved.** The provided match's automatic tracking sees the ball
on only 59% of live frames, while IDSSE sees it on almost every frame. So the corrupted copy
overlays BAR-ATH's *unlabelled* ball-visibility pattern on the IDSSE ball track, at a random
offset, and adds smooth 1 m positional noise. The only thing read from BAR-ATH here is
`isfinite(ball)`.

**Public 5 minutes.** These were used once, at the end, as the pipeline check the rules allow:
0.5200 spellF1 / 0.7893 macroF1. They played no part in any choice. Before the rules update of
23 Oct 2026 we had trained on that window. That model and its training code were removed. The
`experiments/` and `research/` folders are kept for history only, and the pipeline uses neither.

## 3. Assets used and dropped

| Asset | Used? | How |
|---|---|---|
| Tactical tracking (ATD `.xml` + `.txt`) | Yes, the only provided input | Ball and player positions (normalised coordinates scaled to the 105.3 × 68 m pitch), team slots, visibility, boundary distance |
| Tactical homography | **No** | Not needed: the tracking is already in pitch coordinates |
| Smart Tagging | **No** (dropped) | IDSSE has no equivalent, so a model trained on other games can't learn from it |
| Broadcast tracking, broadcast homography | **No** (dropped) | Same reason. Our earlier cross-camera features also relied on thresholds chosen with public labels |
| Videos | No | Not needed |
| IDSSE (external) | Yes | See §2 |

Earlier experiments that we dropped (see `research/`): neural ball reconstruction, frame
grouping, temporal convolution label predictors, graph label spreading and paper-derived feature
stacks. All of them were evaluated under the old rules on the public window, so we don't rely on
any of them now.

## 4. Possession rules implemented

- **Live vs dead.** `D` comes from the classifier's dead-ball probability, which draws on ball
  speed and visibility, stationary players, player movement and distance to the pitch boundary.
  The decoder imposes realistic stoppage durations, learned from IDSSE. A dead ball never gets a
  team.
- **Loose balls, contested phases, single touches, deflections.** Possession stays with the last
  team in control. The semi-Markov decoder keeps the current possession through short,
  low-evidence spells unless the evidence for the other team outweighs the duration prior. A
  touch without control, a deflection or an aerial duel produces only a few frames of
  other-team evidence, and the prior absorbs it.
- **Change of possession.** A new spell starts only where the classifier's evidence stays
  consistent for long enough to beat the duration prior. In practice that means repeated
  proximity, contact and co-motion with the ball by the new team.
- **Restarts.** After a stoppage the decoder starts a new spell. Its team comes from the evidence
  (which team is near the ball), not from a hard-coded restart rule.
- **Contact refinement** (snapping H/A edges to sustained-contact onsets) was evaluated on the
  held-out IDSSE match. It did not help there (best 0.4066 with it, 0.4133 without), so it is
  off.
- **Abstention.** `X` is written only outside the two halves. Every graded frame is H, A or D.

No frame number, clock, event ID, label of the provided game or manual correction enters the
prediction. Every label in `Vltava.csv` comes out of the pipeline.

## 5. Results, limitations and reproduction

| Evaluation | spellF1 | macroF1 |
|---|---:|---:|
| Held-out IDSSE J03WOY, converted labels (selection set; see note) | 0.4133 | 0.7764 |
| Public 5 min, pipeline check only | 0.5200 | 0.7893 |
| Organisers' baseline on the public 5 min | 0.3256 | 0.6207 |

Decoder grid on J03WOY (spellF1): temperatures 1, 2, 3, 5 and 8 gave 0.400, 0.403, 0.413,
0.397 and 0.385 without refinement, and 0.402, 0.400, 0.407, 0.395 and 0.383 with it. We picked
temperature 3 without refinement.

**Version history, every version selected on J03WOY only.**

| Version | J03WOY spellF1 / macroF1 | Public check spellF1 / macroF1 |
|---|---|---|
| v1: raw touch-level labels, 2 training matches | 0.3070 / 0.7513 (raw labels) | 0.4848 / 0.7849 |
| v2: controlled-possession labels, 2 training matches | 0.4107 / 0.7600 | 0.5161 / 0.7599 |
| **v3: controlled-possession labels, 5 training matches (submitted)** | **0.4133 / 0.7764** | 0.5200 / 0.7893 |

We decided in advance to adopt v3 only if it beat v2 on J03WOY, which it did on both metrics.

**How to read these scores.** J03WOY was used to fit the calibration, pick the decoder
temperature and choose between versions. So its score is a *validation/selection* score, and it
is optimistically biased, not an unbiased test result. The public five minutes are a pipeline
check only. They drove no model change. We have no unbiased estimate for the full match, which
the organisers score.



Limitations:

- **Label convention mismatch.** Even after conversion, IDSSE spells (mean about 11 s) are
  shorter than the challenge's (about 14 s on the public window). Our 2 s control threshold is a
  reading of the written definition, not a measured property of the ground truth. The full
  submission predicts 224 live spells with a mean of 16.2 s.
- **Domain shift.** The external matches are German second division with near-complete optical
  tracking. The provided match has fragmented automatic tracking and identity switches. Ball
  corruption covers only part of that gap.
- **Small external sample.** Five training matches and one validation match, all from one
  provider and two German leagues.
- **With more time** we would add a ball-imputation feature trained on IDSSE with block masking;
  learn D boundaries from restart geometry; and test pitch-control or reachability features. All
  of these would be selected on external validation only.

Reproduction. You need Python 3.11+ (verified under 3.13 and 3.14), the challenge bundle in `./data`
(only `tactical/BAR-ATH_tactical_atd.xml` and `.txt` are read), and
for retraining only, the IDSSE files. Put these in `./data/external/`, unrenamed, from
<https://huggingface.co/datasets/pysport/idsse-data>:

- `DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-{J03WOH,J03WQQ,J03WOY,J03WPY,J03WR9}.xml`
- `DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-{J03WOH,J03WQQ,J03WOY,J03WPY,J03WR9}.xml`
- `DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml` and
  `DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml`

```bash
python -m venv .venv-ball
source .venv-ball/bin/activate          # Windows: .venv-ball\Scripts\activate
pip install -r requirements-model.txt   # pinned; scikit-learn must be 1.8.0
python -m possession.train              # optional: rebuilds the model and duration prior (~10 min)
python -m possession.run                # writes Vltava.csv (~1 min, 4 CPU cores)
python evaluation/validate_submission.py Vltava.csv
```

`possession.run` writes `Vltava.json` next to the CSV, which records the hashes of the model,
the prior and the submission. The expected submission hash is the one at the top of this
document. No GPU, video or PyTorch is needed.
