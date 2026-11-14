# Ball reconstruction papers and a feasible next experiment

Reviewed 2026-10-21. No production changes. The current local possession result is 0.6267 public blocked-validation spell F1; none of the papers establishes transfer to this scorer.

## TranSPORTmer, ACCV 2024

[Primary paper, HTML including supplement](https://arxiv.org/html/2410.17785v2), [conference PDF](https://openaccess.thecvf.com/content/ACCV2024/papers/Capellera_TranSPORTmer_A_Holistic_Approach_to_Trajectory_Understanding_in_Multi-Agent_Sports_ACCV_2024_paper.pdf).

**Imputation** retains some ball observations: 80%/90% masked ball ADE is **0.84/1.09 m**. **Inference** hides the entire ball trajectory across **9.6 seconds**, conditioned on player trajectories throughout that interval: ADE **2.71 m**, maximum error **7.39 m**, four-state accuracy **80.84%**. These are offline reconstructions. Future forecasting is a different mask/task; its soccer experiment predicts players for 6.4 seconds after a 3.2-second prefix and can condition on other agents' future positions.

Inputs are pitch xy, team/agent type, visibility/prediction masks; RGB and player velocity are unnecessary. Temporal and social attention permit past/future context. A NaN mask excludes unavailable observations. Agent permutation equivariance does not itself repair identity switches within a trajectory.

Training uses 283 LaLiga matches, 60-frame windows at 6.25 Hz, 22 players plus ball; match-disjoint train/validation/test windows: 82,954/7,500/6,258. State targets are pass, possession, uncontrolled, out-of-play; they do not directly provide H/A team ownership. Metrics are displacement and frame accuracy, not spell F1. See main paper sections 4–5, especially Table 4.

[Primary supplement](https://openaccess.thecvf.com/content/ACCV2024/supplemental/Capellera_TranSPORTmer_A_Holistic_ACCV_2024_supplemental.pdf), also reproduced in the HTML, specifies separate training for every experiment: RTX A6000, 100 epochs, batch 64, AdamW learning rate .001/epsilon .0001, halving LR every 20 epochs, gradient clipping 5, embedding 128, 16 heads, SAB hidden 512, state CE weight 4. Thus one universal pretrained checkpoint is not demonstrated.

The main comparison imputes missing goalkeepers before training. On original data with missing goalkeepers, supplementary Table 3 reports **2.57 m ADE, 7.22 m maximum error, 81.64% state accuracy**. This supports explicit missingness masks, not substituting stationary dummy goalkeepers.

## Peral et al., WACV 2026

[Primary HTML](https://arxiv.org/html/2512.19528v1), [conference record](https://openaccess.thecvf.com/content/WACV2026/html/Peral_Multi-Modal_Soccer_Scene_Analysis_with_Masked_Pre-Training_WACV_2026_paper.html).

This performs full hidden-ball inference using player xy, team type, and player image crops. It uses MixSort bounding boxes and an action-trained ResNet crop encoder; CropDrop masks entire players' crop sequences during pretraining. Full input modalities are restored for fine-tuning. No past/future ball positions are inputs; this does not mean future player context is excluded.

Dataset: proprietary 283 professional matches; 60 frames/9.6 seconds at 6.25 Hz; match-disjoint 43,397/3,968/3,347 train/validation/test sequences. Ball target positions are interpolated from possessor annotations. **All out-of-play sequences are removed**, so the evaluated state classifier has pass/possession/uncontrolled only and does not validate D detection.

Best reported results: **1.17 m ADE, 4.77 m maximum error, 85.85% state accuracy, 89.56% possessor accuracy**. Trajectories-only ablation: 1.97 m, 7.12 m, 77.79%, 82.12%. This shows useful structure-only signal; it does not show that the full visual model remains equally effective without video. Training/checkpoint instructions were not found in the inspected main paper.

## Public implementation and checkpoint availability

No verified official TranSPORTmer or WACV2026 code/weights download was found in the inspected papers, conference pages, author publication pages, or author public repositories. This is a search result, not proof that no release exists. The [author GitHub inventory](https://api.github.com/users/guillem-cf/repos?per_page=100) currently lists the personal website and [JointDiff](https://github.com/guillem-cf/JointDiff). JointDiff exposes a sports benchmark/data interface, not a usable soccer-ball TranSPORTmer checkpoint. Its Bundesliga data derives from IDSSE and could simplify constructing training windows.

[BallRadar's official repository](https://github.com/hyunsungkim-ds/ballradar) is an available implementation alternative. Its README documents training and evaluation, with data/preprocessing instructions; it does not establish a downloadable pretrained checkpoint. The repository requires stable player trajectories and velocities and predicts possession before ball regression.

## Proposed implementation next step

These are engineering inferences from the papers and local inputs, not reproduced paper results:

1. Build 6.25-Hz, roughly 10-second track windows from the already downloaded external IDSSE match. Keep stable PersonId identities within each window and represent substitutions/missing agents with masks. Remove unused roster entries; distinguish 22 simultaneous people from 39 roster identities or 46 unstable target slots.
2. Train a small player-set temporal ball regressor on external ball xy with structured masks: short gaps, long gaps, and fully hidden ball. For target adaptation, mask only trusted observed ball positions and reconstruct them without possession labels. Split windows by contiguous time before generating overlapping samples.
3. Evaluate masked-ball ADE by gap duration against linear interpolation and constant velocity. Preserve observed versus reconstructed flags and confidence. Do not silently replace high-quality observations or train on fabricated dead-ball tail coordinates.
4. Only after reconstruction helps held-out ball positions, append reconstructed nearest-team contacts and disagreement/uncertainty features to the possession model. Run the same strict blocked spell validation once with the fixed external duration decoder.

With no local video, the WACV visual branch is unavailable. With identity switches, per-frame set invariance is insufficient: temporal tokens can jump between players. First stitch short tracklets by geometry/team with explicit uncertainty or limit temporal windows to stable identity stretches. A successful reconstruction experiment would be meaningful new evidence; paper ADE alone cannot guarantee 0.65 spell F1.
