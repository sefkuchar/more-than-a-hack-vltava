"""Regenerate the public validation figure from the current report and CSVs."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME', str(ROOT / '.cache'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

sys.path.insert(0, str(ROOT / 'evaluation'))
import possession_metrics as pm


if __name__ == '__main__':
    report = json.loads((ROOT / 'experiments/validation_report.json').read_text())
    truth = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/gt/BAR-ATH_possession_gt_public.csv'))
    model = pm.label_codes(pm.read_label_csv(ROOT / 'experiments/validation_predictions.csv'))
    baseline = pm.label_codes(pm.read_label_csv(ROOT / 'evaluation/results/baseline_smooth125.csv'))
    start, stop = pm.valid_runs(truth)[0]
    scores = [report['model']['spell_f1_mean'], report['baselines']['baseline_smooth125']['spell_f1_mean']]
    for labels, score in zip((model, baseline), scores):
        actual = pm.evaluate(truth, labels)['spell_f1_mean']
        assert np.isclose(actual, score), 'CSV score differs from current report'
    colors = ['#2563eb', '#e59b20', '#a8afb9', '#ffffff']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12})
    fig, (timeline, bars) = plt.subplots(2, 1, figsize=(13, 6), gridspec_kw={'height_ratios': [1, 1.1]})
    fig.subplots_adjust(left=.21, right=.97, top=.80, bottom=.18, hspace=.55)
    fig.suptitle('Public possession validation', x=.21, ha='left', y=.96, fontsize=19, fontweight='bold')
    fig.text(.21, .875, f'Held-out spell F1 {scores[0]:.4f}  |  0.65 milestone unmet', fontsize=14)
    fig.legend(handles=[Patch(color=colors[k], label=label) for k, label in enumerate(
        ('Barcelona (H)', 'Athletic (A)', 'Dead ball (D)'))], loc='upper right',
        bbox_to_anchor=(.98, .975), ncol=3, frameon=False, fontsize=11)
    timeline.imshow(np.stack([truth[start:stop], baseline[start:stop], model[start:stop]]),
        aspect='auto', interpolation='nearest', cmap=ListedColormap(colors), vmin=0, vmax=3,
        extent=(0, (stop - start) / 25, 2.5, -.5))
    timeline.set_yticks([0, 1, 2], ['Public ground truth', 'Baseline smooth125', 'Held-out model'])
    timeline.set_xlabel('Seconds after the public window starts')
    for second in (60, 120, 180, 240):
        timeline.axvline(second, color='white', alpha=.5, lw=.8)
    bars.barh([1, 0], scores, color=[colors[0], '#9da7b5'], height=.47)
    bars.set_yticks([1, 0], ['Held-out model', 'Baseline smooth125'])
    bars.set_xlim(0, 1)
    bars.set_ylim(-.55, 1.6)
    bars.axvline(.65, color='#596477', linestyle='--', linewidth=1.2)
    bars.text(.66, 1.47, '0.65 target', va='top', fontsize=11, color='#596477')
    for position, score in zip((1, 0), scores):
        bars.text(score - .014, position, f'{score:.4f}', va='center', ha='right', color='white', fontweight='bold')
    bars.set_xlabel('Public spell F1')
    bars.spines[['top', 'right']].set_visible(False)
    fig.text(.025, .075, 'Five contiguous 60 s folds; 10 s training embargo; each fold decoded with its own label-excluded model.',
             fontsize=10, color='#596477')
    fig.text(.025, .035, f"Only {report['model']['n_gt_spells']} public possessions. Configuration selection optimism remains; full-match performance is unknown.",
             fontsize=10, color='#596477')
    fig.savefig(ROOT / 'experiments/public_validation.png', dpi=160, facecolor='white')
