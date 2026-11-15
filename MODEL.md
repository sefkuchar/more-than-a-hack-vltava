# Model card

The saved model is [`models/possession_model.joblib`](models/possession_model.joblib), with its
training report in [`models/possession_model.json`](models/possession_model.json). The full
write-up is the technical document, [Vltava.md](Vltava.md).

| | |
|---|---|
| Classifier | `HistGradientBoostingClassifier` (`random_state=0`), team-symmetric, 62 label-free features |
| Training labels | IDSSE matches J03WOH and J03WQQ: DFL/Sportec `BallStatus` and `BallPossession` (other games, CC-BY-4.0), converted to controlled possession (spells < 2 s returned to the team in control) |
| Calibration and tuning | IDSSE match J03WOY, held out from the classifier |
| Provided-game labels | **none**: not for training, calibration or selection |
| Decoder | exact 25 Hz semi-Markov; spell-duration prior refitted on the converted IDSSE training labels; temperature 3 |
| Held-out IDSSE J03WOY (converted labels) | 0.4107 spellF1 / 0.7600 macroF1 |
| Pipeline check, public 5 min | 0.5161 spellF1 / 0.7599 macroF1 (baseline 0.3256 / 0.6207) |

IDSSE's raw `BallPossession` flags every touch (about 400 live spells per match, mean 7 s).
Following CHALLENGE.md's definition of *controlled* possession, we return spells shorter than
2 s to the team in control. That leaves about 280 spells per match with a mean of 11 s. The public-window number is a pipeline check, not a selection criterion. Nothing
was chosen on it.

```bash
python -m possession.selfcheck     # decoder and team-symmetry unit checks
python -m possession.train         # needs data/external IDSSE files (see Vltava.md)
python -m possession.run           # writes Vltava.csv
```

Python 3.11+ and the pinned `requirements-model.txt` (scikit-learn 1.8.0) are required.
`experiments/` and `research/` are historical, from before the 23 Oct 2026 rules update; see
[experiments/README.md](experiments/README.md).
