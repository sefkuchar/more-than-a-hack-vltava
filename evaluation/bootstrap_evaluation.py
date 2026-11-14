"""Seeded bootstrap evaluation of held-out possession predictions.

Scores the current out-of-fold predictions and the previous baseline against the public
ground truth, then resamples whole possessions for spellF1 and whole ground-truth segments
for macro-F1. The same seed is used for both models, so every draw is paired and the
reported difference is a paired bootstrap rather than the difference of two independent
intervals.

The outer cross-validation predictions are the honest comparison: both models are scored
out of fold on the same frames. Per-fold intervals are included, but each fold contains
only four or five possessions, so they are coarse and mostly useful as a warning about
sample size.

Usage:
    python evaluation/bootstrap_evaluation.py
    python evaluation/bootstrap_evaluation.py --draws 20000 --seed 7
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from evaluation import possession_metrics as pm  # noqa: E402


DEFAULT_GT = ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv"
DEFAULT_CURRENT = ROOT / "experiments/validation_predictions.csv"
DEFAULT_BASELINE = ROOT / "experiments/baseline_validation_predictions.csv"
DEFAULT_FOLDS = ROOT / "experiments/validation_report.json"
DEFAULT_OUTPUT = ROOT / "experiments/bootstrap_report.json"
PERCENTILES = (5, 95)


def display_path(path):
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(scores):
    """Percentile interval plus the draw spread and a discreteness hint."""
    if not len(scores):
        return {"low": None, "high": None, "mean": None, "std": None,
                "n_unique": 0, "n_draws": 0}
    low, high = np.percentile(scores, PERCENTILES)
    return {
        "low": float(low),
        "high": float(high),
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=0)),
        "n_unique": int(len(np.unique(scores))),
        "n_draws": int(len(scores)),
    }


def point_metrics(gt, prediction):
    score = pm.evaluate(gt, prediction, bootstrap_n=0)
    return {key: score[key] for key in
            ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells")}


def fold_scores(gt, prediction, folds, draws, seed):
    """Per-fold point scores and resampled intervals.

    A fold is a contiguous slice of the public window, so it is safe to score in isolation.
    Empty folds are skipped rather than assigned a meaningless interval.
    """
    out = []
    for fold in folds:
        first = int(fold["held_out_first_frame"])
        last = int(fold["held_out_last_frame"])
        fold_gt = gt[first:last + 1]
        fold_pred = prediction[first:last + 1]
        runs = pm.valid_runs(fold_gt)
        if not runs:
            continue
        filled = pm.fill_abstentions(fold_pred, runs)
        spell_draws = pm.bootstrap_spell_f1_scores(
            fold_gt, filled, runs, draws, seed)
        macro_draws = pm.bootstrap_macro_f1_scores(
            fold_gt, fold_pred, runs, draws, seed)
        out.append({
            "fold": int(fold["fold"]),
            "held_out_first_frame": first,
            "held_out_last_frame": last,
            "point": point_metrics(fold_gt, fold_pred),
            "bootstrap": {
                "spell_f1": interval(spell_draws),
                "macro_f1": interval(macro_draws),
            },
        })
    return out


def model_report(path, gt, runs, folds, draws, seed):
    prediction = pm.label_codes(pm.read_label_csv(path))
    if len(prediction) != len(gt):
        raise ValueError("{}: {} rows, expected {}".format(path, len(prediction), len(gt)))
    filled = pm.fill_abstentions(prediction, runs)
    spell_draws = pm.bootstrap_spell_f1_scores(gt, filled, runs, draws, seed)
    macro_draws = pm.bootstrap_macro_f1_scores(gt, prediction, runs, draws, seed)
    return {
        "predictions": display_path(path),
        "prediction_sha256": sha256(path),
        "point": point_metrics(gt, prediction),
        "bootstrap": {
            "spell_f1": interval(spell_draws),
            "macro_f1": interval(macro_draws),
        },
        "per_fold": fold_scores(gt, prediction, folds, draws, seed),
        "_draws": {"spell_f1_mean": spell_draws, "macro_f1": macro_draws},
    }


def paired(current, baseline, key):
    """Paired difference current - baseline from the same seeded draws."""
    current_draws = current["_draws"][key]
    baseline_draws = baseline["_draws"][key]
    if len(current_draws) != len(baseline_draws):
        raise ValueError("paired draws have different lengths")
    difference = current_draws - baseline_draws
    low, high = np.percentile(difference, PERCENTILES) if len(difference) else (None, None)
    return {
        "point_difference": (current["point"][key] - baseline["point"][key]),
        "low": None if low is None else float(low),
        "high": None if high is None else float(high),
        "mean": float(np.mean(difference)) if len(difference) else None,
        "share_current_better": (float(np.mean(difference > 0))
                                 if len(difference) else None),
        "share_tied": float(np.mean(difference == 0)) if len(difference) else None,
        "n_draws": int(len(difference)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, default=DEFAULT_GT)
    parser.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--current-label", default="current")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.draws <= 0:
        raise ValueError("--draws must be positive")

    gt = pm.label_codes(pm.read_label_csv(args.gt))
    runs = pm.valid_runs(gt)
    if not runs:
        raise ValueError("ground truth has no graded frames")
    with open(args.folds) as handle:
        folds = json.load(handle)["folds"]

    current = model_report(args.current, gt, runs, folds, args.draws, args.seed)
    baseline = model_report(args.baseline, gt, runs, folds, args.draws, args.seed)

    report = {
        "validation_type": "paired_bootstrap_evaluation",
        "protocol": "Possessions are resampled with replacement for spellF1; ground-truth "
                    "segments are resampled with replacement for macro-F1. Both models use "
                    "the same seed and therefore the same resamples.",
        "ground_truth": display_path(args.gt),
        "ground_truth_sha256": sha256(args.gt),
        "public_window": [list(run) for run in runs],
        "draws": args.draws,
        "seed": args.seed,
        "percentiles": list(PERCENTILES),
        "models": {args.current_label: current, args.baseline_label: baseline},
        "paired": {
            "{}_minus_{}".format(args.current_label, args.baseline_label): {
                "spell_f1": paired(current, baseline, "spell_f1_mean"),
                "macro_f1": paired(current, baseline, "macro_f1"),
            }
        },
        "notes": [
            "The comparison uses held-out out-of-fold predictions, not the public submission.",
            "Each fold contains only four or five possessions; per-fold intervals are coarse.",
            "The ground truth has 17 possessions, so all intervals are wide by construction.",
            "The seed fixes the resamples; changing --draws changes the interval endpoints.",
        ],
    }
    for model in report["models"].values():
        del model["_draws"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    summary = {
        "models": {
            name: {"point": model["point"], "bootstrap": model["bootstrap"]}
            for name, model in report["models"].items()
        },
        "paired": report["paired"],
        "output": display_path(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
