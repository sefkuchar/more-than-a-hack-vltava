# Historical experiments (not part of the submission)

These scripts were written before the rules update of 23 October 2026, when training on
the 5 public minutes was still allowed. Many of them fit or select on
`evaluation/gt/BAR-ATH_possession_gt_public.csv`. **None of them is used by the submission
pipeline**, and none of their outputs feeds the saved model.

The submission is produced only by `possession/train.py` (IDSSE external matches) and
`possession/run.py`. The only modules imported from this folder are
`experiments/load_idsse.py` (a parser for the public IDSSE XML) and, optionally,
`experiments/align_broadcast.py` (label-free, not used by the current model).

The historical prediction files (`experiments/*validation_predictions.csv`) and old models were
removed from the repository; scripts that read them will not run as-is.
