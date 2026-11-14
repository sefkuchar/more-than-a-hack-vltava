"""Compare pooled cross-validation predictions using the official metric."""

from pathlib import Path

from evaluation import possession_metrics as pm

ROOT = Path(__file__).resolve().parents[1]


def main():
    gt = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    prediction = ROOT / 'experiments/validation_predictions.csv'
    if not prediction.is_file():
        raise FileNotFoundError('Cross-validation predictions missing. Run: python -m possession.validate')
    paths = [('model_cross_validated', prediction)] + [
        (path.stem, path) for path in sorted((ROOT / 'evaluation/results').glob('baseline*.csv'))]
    previous = ROOT / 'experiments/baseline_validation_predictions.csv'
    if previous.is_file():
        paths.append(('previous_model_cv', previous))
    rows = []
    for name, path in paths:
        table = pm.read_label_csv(path)
        reason = pm.validate_submission(table, len(gt))
        if reason:
            raise ValueError(f'{path.name}: {reason}')
        score = pm.evaluate(gt, pm.label_codes(table))
        rows.append((name, score['spell_f1_mean'], score['macro_f1']))
    print('Five-fold blocked cross-validation; pooled out-of-fold scores on the public five minutes.\n')
    print(f'{"model":26s} {"spellF1":>9s} {"macroF1":>9s}')
    for name, spell, macro in sorted(rows, key=lambda row: (-row[1], -row[2])):
        print(f'{name:26s} {spell:9.4f} {macro:9.4f}')
    print('\n17 public possessions; repeated model selection makes this exploratory.')


if __name__ == '__main__':
    main()
