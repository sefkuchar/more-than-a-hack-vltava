# Model card

The saved model is [`models/possession_model.joblib`](models/possession_model.joblib), with its
training report in [`models/possession_model.json`](models/possession_model.json). The full
write-up is the technical document, [Vltava.md](Vltava.md).

| | |
|---|---|
| Classifier | `HistGradientBoostingClassifier` (`random_state=0`), team-symmetric, 62 label-free features |
| Training labels | IDSSE matches J03WOH and J03WQQ: DFL/Sportec `BallStatus` and `BallPossession`, other games, CC-BY-4.0 |
| Calibration and tuning | IDSSE match J03WOY, held out from the classifier |
| Provided-game labels | **none**: not for training, calibration or selection |
| Decoder | exact 25 Hz semi-Markov; spell-duration prior fitted on IDSSE; temperature 1 |
| Held-out IDSSE J03WOY | 0.3070 spellF1 / 0.7513 macroF1 |
| Pipeline check, public 5 min | 0.4848 spellF1 / 0.7849 macroF1 (baseline 0.3256 / 0.6207) |

The J03WOY score is measured against IDSSE's own possession convention. That convention has
about 400 live spells per match, with a mean of 7 s, so it is much harder than the challenge's
(mean 14 s). The public-window number is a pipeline check, not a selection criterion. Nothing
was chosen on it.

```bash
python -m possession.selfcheck     # decoder and team-symmetry unit checks
python -m possession.train         # needs data/external IDSSE files (see Vltava.md)
python -m possession.run           # writes Vltava.csv
```

Python 3.11+ and the pinned `requirements-model.txt` (scikit-learn 1.8.0) are required.
`experiments/` and `research/` are historical, from before the 23 Oct 2026 rules update; see
[experiments/README.md](experiments/README.md).
