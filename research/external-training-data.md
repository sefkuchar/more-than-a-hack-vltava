# External possession training data

Researched 2026-10-21. No target Barcelona–Athletic labels or corrected possession files were accessed. After initial 4 KB schema probes, the unrelated J03WOH full tracking and metadata were downloaded into ignored `data/external/idsse/` and extracted by `experiments/load_idsse.py`.

Extraction produced 137,214 frames and 39 roster slots. BallStatus counts: live 71,945, dead 65,269. BallPossession has only 1/2 values. Combined labels: H 39,126, A 32,819, D 65,269. First/second halves have 69,424 / 67,790 frames. Metadata PlayingTime totals 2,877,760 ms, equivalent to 71,944 frames at 25 Hz; the one-frame difference corroborates the live-status interpretation. Ball coordinates remain finite in dead frames, including repeated tail values, so ball visibility patterns may differ substantially from target tracking.

## Recommended source: IDSSE / Sportec Open DFL

[PySport's dataset mirror](https://huggingface.co/datasets/pysport/idsse-data) provides seven complete 2022/23 German Bundesliga / second-division matches, tracked at 25 Hz with all players and ball. It identifies the license as CC-BY-4.0 and links the [original Figshare DOI](https://doi.org/10.6084/m9.figshare.28196177) and [dataset paper](https://doi.org/10.1038/s41597-025-04505-y). Attribute Bassek et al. and DFL/Sportec in downstream artifacts.

[Kloppy's official documentation](https://kloppy.pysport.org/user-guide/loading-data/sportec/) demonstrates dense `ball_state`, `ball_owning_team_id`, ball x/y/z/speed and player x/y/speed. This is immediately useful for supervised H/A/D training on external matches. These provider labels have not been shown to follow the hackathon's precise spell convention.

## Smallest complete match

Choose `J03WOH`: Fortuna Düsseldorf versus SSV Jahn Regensburg, 2022-08-26. Verified against the [mirror file inventory API](https://huggingface.co/api/datasets/pysport/idsse-data/tree/main):

| File | Bytes | Direct download |
|---|---:|---|
| Tracking | 348,945,715 | [positions XML](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml) |
| Metadata | 11,745 | [matchinformation XML](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml) |
| Events, optional | 635,747 | [events XML](https://huggingface.co/datasets/pysport/idsse-data/resolve/main/DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml) |

All seven tracking XMLs total approximately 2.63 GB. Start with one (~333 MiB), then add a second held-out match if transfer looks useful. `J03WQQ` (Düsseldorf–St. Pauli) tracking is 356,219,423 bytes and `J03WOY` (Düsseldorf–Rostock) is 362,567,859 bytes. Their URL patterns differ only in match ID. `J03WMX` (Köln–Bayern) is a first-division alternative, 418,524,833 bytes, with competition ID `DFL-COM-000001`.

## Raw schema and ingestion caveats

Two HTTP byte-range requests verified raw schema without downloading the match. The beginning includes pitch `105.00 × 68.00`, then `FrameSet GameSection="firstHalf" TeamId="..." PersonId="..."`. Player frames have `N,T,X,Y,D,S,A,M`. The tail contains ball frames with `X,Y,Z,D,S,A,M,BallPossession,BallStatus`; actual dead-ball examples have `BallPossession="2" BallStatus="0"`.

The XML is grouped by object and period, rather than one complete multi-object frame per row. Fetching a prefix will only obtain one player's trajectory. A complete file scan is necessary to align every player and ball by `(GameSection,N)`; use `iterparse` and clear consumed elements to limit memory.

The [official Kloppy deserializer](https://raw.githubusercontent.com/PySport/kloppy/master/kloppy/infra/serializers/tracking/sportec/deserializer.py) interprets `BallStatus==1` as alive and `BallPossession==1` as home. Critically, its fallback maps every other possession value, including absent values, to away. Parse and count raw values before converting; use home for 1, away for 2, and unknown otherwise. Dead state overrides the possession owner. Keep `only_alive=False` to retain the D training class. Never include either provider label as a model input.

## Suggested practical experiment

1. Download one tracking XML plus metadata, keeping them external to the target fixture's inputs. Parse 25 Hz raw labels and tracks; aggregate into the same time step and geometry used by the target model.
2. Build only shared features: ball/player coordinates, relative nearest-team distances, movement and short time-window aggregates. Derive velocity from position differences so mismatched provider speed units cannot silently poison transfer. Check coordinate origin, orientation and units using pitch metadata.
3. Train an external H/A/D classifier with team identity swapping / geometric mirroring. External provider status and possession fields are targets only. Do not fabricate opponent labels for unknown raw values.
4. Use a separate external match for validation if a second download is made. Use the allowed five-minute target annotation for target-domain calibration and temporal decoding selection with strict contiguous holdout; its short duration cannot establish full-match spellF1.
5. Compare transfer against the existing target-only baseline before choosing a final model. Provider-label training may improve control and dead-ball cues but cannot guarantee spellF1 >= 0.65 under the target annotation convention.

## Alternatives

[Metrica sample data](https://github.com/metrica-sports/sample-data) contains synchronized anonymized event/tracking examples but its repository does not document equivalent dense provider possession/live-state targets. [SkillCorner open data](https://github.com/SkillCorner/opendata) is worth future inspection for broader geometry coverage, but it should not displace IDSSE for this immediate supervised-label need. Event-only sources and video tracking benchmarks require substantial label reconstruction and offer less immediate value.

One source inconsistency: the Hugging Face dataset card lists two match IDs (`J03WPF`, `J03WQF`) absent from its actual file tree. Use actual inventory and Kloppy's documented seven matches, including `J03WQQ` and `J03WR9`.
