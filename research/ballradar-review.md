# Ball Radar: reconstruction and imputation feasibility

Reviewed 2026-10-21. Primary paper and actual repository files inspected; no target corrected tracking or private labels used. Local environment check: ARM64, Python 3.14.2, PyTorch not installed. This is research and an implementation proposal, not a reproduced model result.

## Paper evidence

[Kim et al., KDD 2023, primary paper](https://arxiv.org/html/2306.08206), sections 3–5 and Tables 1–3:

- **Task:** reconstruct ball positions throughout a player-trajectory window using bidirectional context; this is offline inference, not future-only forecasting. A separate experiment adds partial ball observations for imputation.
- **Inputs:** 10-second, 10-Hz windows; 22 players, six motion features each. Only known in-play episodes are used. Missing GPS goalkeepers are estimated separately.
- **Supervision:** ball coordinates and individual possession. During flight, possession belongs to the **next receiver**, unlike this challenge's last-controller convention. GPS ball targets are reconstructed using events.
- **Split:** training 10 Fitogether matches plus Metrica games 1–2; validation 2 Fitogether matches plus game 3 first half; test 3 Fitogether matches plus game 3 second half.
- **Main results:** H-LSTM-RL mean position error **3.6561 m**, player accuracy **64.70%**, team accuracy **85.85%**. Postprocessing increases position error to **4.0719 m**.
- **Imputation:** before postprocessing, 90% random masking yields **2.0939 m** position error and **95.04%** team accuracy; 80% masking yields **1.3052 m** and **97.41%**. These are not spellF1, and random missingness differs from prolonged real occlusion.

## Repository reality

Inspected main tree commit `2486896cfb933ecd304489883241a00e3645ca8c` via the [complete Git tree](https://api.github.com/repos/hyunsungkim-ds/ballradar/git/trees/2486896cfb933ecd304489883241a00e3645ca8c?recursive=1). No model checkpoints or root project licence were present. The [releases page](https://github.com/hyunsungkim-ds/ballradar/releases) is empty. The embedded [Set Transformer licence](https://github.com/hyunsungkim-ds/ballradar/blob/main/set_transformer/LICENSE) is MIT; it does not establish the licence of the entire project. Do not describe the whole repository as MIT-licensed.

The [official README](https://github.com/hyunsungkim-ds/ballradar#readme) links public Metrica preprocessing and a tracking-data Drive folder, but Fitogether training data remain private. It instructs users to train models and save resulting weights locally; no ready pretrained model download was identified. Thus this is a source implementation to adapt/train, not a verified downloadable predictor.

### Concrete code issues

1. **Imputation branch appears unreachable.** [train.py](https://github.com/hyunsungkim-ds/ballradar/blob/main/train.py) checks `model.module.model_type == "player_ball"` before supplying masked targets. But [PlayerBall](https://github.com/hyunsungkim-ds/ballradar/blob/main/models/player_ball.py) sets that attribute to `"macro_regressor"` for ball prediction. Static reading therefore indicates the published launcher bypasses its masked-conditioning branch. This was not executed locally.
2. **Conditional inputs include possession hints.** In that intended branch, training passes both `macro_target` and `micro_target` with one random mask. The forward method one-hot encodes masked **true individual-possession labels**, as well as masked true ball coordinates. For our use, those true player labels are unavailable. Passing `macro_target=None` is supported by the forward method, but a model must be trained in that same regime; treating conditional results as automatically attainable from ball-only anchors is unjustified.
3. **CUDA launcher.** The trainer sets `default_device="cuda:0"` and wraps the model in `DataParallel`. CPU/MPS require launcher changes. Even Metrica-only training enumerates the GPS directory unconditionally, another setup issue.
4. **Pitch assumptions.** The model's final scaling hardcodes `[108,72]`; this must agree with preprocessing. Our 105.3×68 m coordinates need consistent rescaling, including derivatives and inverse output conversion.

The [dataset loader](https://github.com/hyunsungkim-ds/ballradar/blob/main/dataset.py) expects persistent player columns, groups 11 players per team, adds four outside classes, drops columns with missing values within phases, and skips phases without enough players. It backfills individual possession and builds positive in-play episodes before sliding windows. Consequently, simply giving it our 46 fragmented slots would either fail shape/coverage assumptions or select unreliable fragments. Its set encoders handle player ordering at a time step; they do not automatically repair identity swaps along the recurrent player paths. The loader also materializes overlapping windows, so a lazy window dataset would be preferable for local memory use.

## Proposed adaptation for the current challenge

**Priority:** test whether ball imputation improves missing-ball evidence, while retaining the existing H/A/D decoder. Ball imputation cannot by itself supply referee live/dead status. We should not promise that it raises the current public spellF1 of approximately .6267 to .65.

1. **Establish a separate trajectory benchmark first.** On the unrelated public IDSSE match already available locally, hide complete contiguous ball spans, with lengths drawn from the target's unlabeled missingness. Compare masked-frame position error and nearest-team correctness against linear interpolation and constant velocity. Split by whole half or long disjoint blocks before creating windows; forbid overlapping train/validation contexts. Random single-frame masking alone would make the comparison too easy.
2. **Start with a smaller imputer we can train and audit.** Pool team player sets per frame with explicit missingness masks, then use a short bidirectional sequence model receiving visible ball coordinates, observation masks, and time since/until visible anchors. Predict coordinates or residuals around anchor interpolation. A separate team head is possible using external provider labels; do not manufacture individual possession ground truth from our 17 public spells. This is a proposed simplification, not Ball Radar reproduction.
3. **If reproducing Ball Radar specifically, first build stable local tracklets.** Deduplicate same-team objects and stitch plausible continuations within each 10-second window. Require sufficiently stable 11-per-team inputs, include actual goalkeeper estimates with uncertainty, and abstain where reconstruction fails. Alternatively, modifying the architecture to accept masked variable player sets is a substantive new model, requiring training with synthetic player dropout, slot switches and positional noise.
4. **Train with observed ball anchors only.** Correct the launcher branch; keep macro hints absent during both training and inference. Preserve a separate observation mask instead of conflating a missing anchor with coordinates `(0,0)`. Add block masking and unreliable-anchor corruption. Use public external coordinates as targets; optionally add target automatic observed coordinates only as noisy self-supervision with explicit quality weighting, never call them corrected truth.
5. **Integrate conservatively.** Keep observed and inferred ball features separate. Add inferred player distances, uncertainty and anchor age to the possession classifier rather than replacing reliable observed ball positions. Recompute inferred features independently for each evaluation training regime if any target possession supervision enters the imputer. Continue strict outer label isolation and match the actual spellF1 metric.

## Compute plan

PyTorch must be installed in an isolated compatible environment; the original Python 3.10/PyTorch 1.13 recipe should not be assumed compatible with the current Python 3.14 interpreter. The upstream [requirements file](https://github.com/hyunsungkim-ds/ballradar/blob/main/requirements.txt) does not itself list torch.

For a modern implementation, select MPS only after `torch.backends.mps.is_available()` succeeds, otherwise CPU; [official PyTorch MPS documentation](https://docs.pytorch.org/docs/2.14/notes/mps.html) explains this device check. Remove CUDA-only wrapping and use one model on the selected device. Actual MPS availability, operation coverage, runtime and memory have not been measured here.

Begin with float32, 100-step windows, small batches and lazy sampling. Benchmark one forward/backward batch and one external epoch before committing to full training; log device, throughput and validation error. CPU is sufficient for a small prototype in principle, but no honest training-time estimate can be given without that measurement. The original launcher uses batch 224 and 256-unit recurrent layers; that is not a justified default for this machine or one external match.

A successful external block-imputation experiment would justify scaling to more public matches and testing downstream possession. A failed comparison against interpolation should stop this branch before expensive neural training and public-window model selection.
