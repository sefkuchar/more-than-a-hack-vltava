# Team Vltava: ball possession, technical document

**More than a Hack 2027, Phase 1** · Repository: <https://github.com/sefkuchar/more-than-a-hack-vltava>
· Submission: `Vltava.csv` (SHA-256 `2fcb6f10fb885bc7041d62506fc014c4d119226165b7193d9fb2ad2ff4edebe2`)

## 1. Approach

We predict per-frame evidence for H/A/D with learned classifiers. We then **decode whole
possession sequences** instead of smoothing frame labels. `spellF1` rewards recovering complete
possessions, and frame-level classifiers flicker. A plain majority-vote filter either keeps the
flicker or erases real short possessions. A duration-aware decoder trades the evidence against
how long possessions and stoppages really last.

Pipeline (`python -m possession.run`):

1. **Features (label-free).** 77 base features per frame, computed from tactical tracking, Smart
   Tagging and pitch geometry:
   - distance from the ball to the nearest home and nearest away player, and the margin between them;
   - Gaussian "contact" scores and sustained-contact flags;
   - ball visibility and age of the last observation, ball speed and trajectory turn;
   - distance to the pitch boundary, and player movement (share of moving players, mean speed);
   - Smart Tagging restart and team-phase windows;
   - rolling means over 5, 12, 25, 50 and 125 frames.

   A second view adds 8 features (85 in total) from the broadcast camera:
   - a label-free spatial/temporal alignment of broadcast tracking to the tactical timeline;
   - cross-camera team-assignment correction and the corrected contact evidence;
   - ball kick acceleration from 7-frame Savitzky–Golay derivatives.
2. **Two classifiers.** Shallow `HistGradientBoostingClassifier` models (`random_state=0`), one
   per feature view, trained on the public 5 minutes (every 3rd frame). Home and away are swapped
   during training and averaged at prediction, so the model has no team bias.
3. **Calibration and blend.** Each model gets a 2-parameter calibrator (scale and dead-ball
   intercept), fitted on inner blocked held-out predictions. The blend keeps the base model's D
   probability and averages the two models' conditional H/A probabilities.
4. **Semi-Markov decoding.** An exact 25 Hz semi-Markov (explicit-duration) decoder runs per half,
   with emission temperature 2. The duration priors for possessions and dead-ball spells were
   fitted on an unrelated public match (IDSSE / DFL-Sportec, CC-BY-4.0). Only the fitted
   parameters are shipped, in `possession/duration_prior.json`.
5. **Contact boundary refinement.** Existing H/A transitions snap to the nearest onset of new-team
   control within ±50 frames. Spell count, label order and dead-ball edges are unchanged.
6. **Camera short-spell recovery.** Short possessions that the duration prior suppressed are
   restored, but only where broadcast-corrected geometry strongly supports them (see §3).

## 2. Assets used and dropped

| Asset | Used? | How |
|---|---|---|
| Tactical tracking (ATD) | Yes, core | Ball and player positions, team slots, visibility |
| Tactical homography | Yes | Pitch coordinates and boundary distance |
| Smart Tagging | Yes | `SET PIECES` → restart feature; tactical phases → per-team phase features |
| Broadcast tracking + homography | Yes | Second view: team-assignment correction, corrected contact, short-spell recovery |
| Videos (`.mp4`) | **No** | Not needed by the final pipeline |
| Public 5-min ground truth | Yes | The only supervised labels: training, calibration, blocked CV |
| IDSSE public match (external) | Duration prior only | Fitted spell-duration distributions; the raw data is not shipped |

Tried and **dropped** (all scored on the same nested blocked cross-validation; details in `research/`):

- **Neural ball reconstruction** (a small model inspired by Ball Radar). It filled long ball gaps
  better than interpolation, but did not improve possession spellF1.
- **Frame grouping** (fixed or adaptive), graph label spreading, temporal-convolution label
  predictors and external-pretrained temporal transfer. All scored between 0.43 and 0.59 spellF1,
  below the frame classifier with decoding.
- **Appending every feature idea from the papers to one classifier** (kick dynamics, opponent
  pressure, object context): 0.50 to 0.63 spellF1. Only the separate two-view blend helped.
- **Decoder emission temperature 5** (0.6800 spellF1) and a fixed temperature pool; we chose
  temperature 2.

## 3. Possession rules implemented

The challenge definitions and the public ground truth are the reference. The system encodes them
as follows:

- **Live vs dead.** Dead-ball frames (`D`) come from the classifier's D probability. That
  probability draws on ball visibility, ball speed, stationary players, distance to the pitch
  boundary and Smart Tagging `SET PIECES` windows. The decoder then enforces realistic stoppage
  durations. We never assign a team to a dead ball, as the challenge requires.
- **Loose balls and contested phases.** Possession stays with the last controlling team. The
  semi-Markov decoder keeps the current possession through brief, low-evidence spells (a loose
  ball, an aerial duel, a missed tracking frame) unless the evidence for the other team outweighs
  the duration prior.
- **What counts as a change of possession** (controlled gain). A new H/A edge snaps only to a
  control onset. There are two kinds:
  - *Sustained contact:* the ball is observed, the gaining team is within 1.5 m, the other team
    is more than 3 m away, and this co-motion lasts **at least 8 frames (0.32 s)**.
  - *Kick:* an exclusive-proximity interval that contains an observed ball acceleration of at
    least 4 m/s². Accelerations above 60 m/s or 150 m/s² are rejected as noise.

  Single touches, deflections and duels without follow-up control produce neither onset. They are
  also too short to survive the duration prior.
- **Short possessions.** A run of at least 12 frames (0.48 s) can flip H↔A in recovery, but only
  where broadcast correction is active, and only if:
  - the corrected camera puts the alternative team within 2.5 m of the ball;
  - the current team is more than 3 m away;
  - the classifier gives the alternative team at least 0.4 mean probability.

  Recovery never touches `D` frames. The full submission recovers 8 such runs.
- **Restarts.** After a stoppage, the decoder starts a new spell. Its team comes from the
  classifier's evidence (proximity, team phase tags), not from a rule about who restarts.
- **Abstention.** `X` is written only outside the two halves (before kickoff, halftime and after
  the final whistle). Every graded frame gets H, A or D.

No frame number, clock, event ID, private label or manual correction enters the prediction. Every
label in `Vltava.csv` comes out of the pipeline.

## 4. Results and limitations

| Evaluation | spellF1 | macroF1 |
|---|---:|---:|
| Public 5 min, training score (labels were used to fit) | 0.9882 | 0.9919 |
| **Nested 5-fold blocked CV, pooled out-of-fold** | **0.7750** | **0.8880** |
| Provided baseline (`baseline_smooth50`) | 0.3256 | 0.6207 |

The cross-validation uses five contiguous 1-minute folds, with a 10-second embargo on both sides
and nested calibration. A seeded paired bootstrap (10,000 draws) gives a 90% interval of
0.59–0.93 for spellF1.

Limitations:

- **Tiny supervision.** There are only 17 public possessions. Repeated model selection on them is
  optimistic, and the CV interval is wide.
- **Ball visibility shift.** The ball is visible on 88% of public frames, but only about 59%
  across the whole match. Accuracy in long occlusions is untested and probably lower.
- **Dead-ball timing.** D boundaries rely on indirect cues (speed, boundary, Smart Tagging). Quick
  throw-ins and free kicks that are taken fast can merge with live play.
- **Team-assignment errors** in the automatic tracking are only partly corrected, and only where
  broadcast alignment exists.
- **With more time we would:**
  - train on more labelled matches (external providers) to fit the classifiers, not just the
    duration prior;
  - add a ball-imputation feature trained on block-masked real occlusions;
  - add pitch-control / reachability features;
  - learn D boundaries directly from Smart Tagging restart events.

## 5. Reproduction

Needs Python 3.11+ (output verified identical under 3.13 and 3.14) and the challenge data bundle
in `./data` (or `MTAH_DATA_DIR`). Required files:

- `tactical/BAR-ATH_tactical_atd.xml` and `.txt`, `_homography.csv`, `_smart_tagging.xml`;
- `broadcast/BAR-ATH_broadcast_atd.txt`, `_homography.csv`.

```bash
python -m venv .venv-ball
source .venv-ball/bin/activate          # Windows: .venv-ball\Scripts\activate
pip install -r requirements-model.txt   # pinned; scikit-learn must be 1.8.0
python -m possession.run --out Vltava.csv
python evaluation/validate_submission.py Vltava.csv
```

`possession.run` loads the saved model `models/possession_model.joblib`
(SHA-256 `fa0acdef…c6fa`) and writes `Vltava.csv` plus `Vltava.json`, a report that includes the
submission hash. The expected hash is the one at the top of this document. Retraining from scratch
is `python -m possession.train`; cross-validation is `python -m possession.validate`. Runtime is a
few minutes on 4 CPU cores. No GPU, video, PyTorch or external download is needed.
