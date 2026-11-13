"""Publish compact, verified development records and learning curves, not weights."""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from scripts.verify_neural_run import verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh summary directory')
    checked = []
    for path in args.runs:
        receipt = verify(path)
        checked.append((path, json.loads((path / 'run.json').read_text()), receipt))
    args.output.mkdir(parents=True)
    rows = []
    labels = {'fine_tuned_weighted': 'Fine-tuned DistilBERT',
              'frozen_encoder_weighted': 'Frozen encoder + trained head',
              'fine_tuned_unweighted': 'Fine-tuned, unweighted loss'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    for path, report, receipt in checked:
        name = f'{report["variant"]}_seed{report["seed"]}'
        dest = args.output / name
        dest.mkdir()
        for filename in ['run.json', 'history.json', 'dev_predictions.csv', 'manifest.json']:
            shutil.copy2(path / filename, dest / filename)
        (dest / 'verification.json').write_text(json.dumps(receipt, indent=2))
        metrics = report['selected_development_metrics']
        rows.append({'model': report['variant'], 'seed': report['seed'],
                     'selected_epoch': report['selected_epoch'],
                     'development_edge_precision': metrics['edge_precision'],
                     'development_edge_recall': metrics['edge_recall'],
                     'development_edge_f1': metrics['edge_f1'],
                     'development_macro_f1': metrics['macro_f1'],
                     'elapsed_seconds': report['elapsed_seconds'],
                     'trainable_parameters': report['parameter_counts']['trainable_parameters']})
        history = report['history']
        title = labels[report['variant']]
        for axis, values in zip(axes, [[r['mean_training_batch_loss'] for r in history],
                                      [r['dev']['unweighted_cross_entropy'] for r in history],
                                      [r['dev']['edge_f1'] for r in history]]):
            axis.plot([r['epoch'] for r in history], values, marker='o', label=title)
            axis.set_xlabel('Epoch')
            axis.set_xticks([r['epoch'] for r in history])
            axis.grid(alpha=.2)
    baseline = json.loads((ROOT / 'artifacts/runs/phase1_extraction_verified_20261101/metrics.json').read_text())
    classical = next(row['dev'] for row in baseline['search'] if row['C'] == baseline['selected_C'])
    rows.append({'model': 'historical_tfidf_lr_C10', 'seed': 42, 'selected_epoch': None,
                 **{f'development_{key}': classical[key] for key in ['edge_precision','edge_recall','edge_f1','macro_f1']},
                 'elapsed_seconds': None, 'trainable_parameters': None})
    axes[2].axhline(classical['edge_f1'], linestyle='--', color='#777777', label='Historical TF-IDF/LR dev')
    for ax, title in zip(axes, ['Mean training batch loss', 'Development cross-entropy', 'Development directed edge F1']):
        ax.set_title(title)
    axes[2].set_ylim(0,1)
    handles, names = axes[2].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('Development only: one seed, identical data partitions. Test evaluation pending.', fontsize=11)
    fig.tight_layout(rect=(0,.1,1,.93))
    fig.savefig(args.output / 'learning_curves.png', dpi=160)
    plt.close(fig)
    summary = {'status': 'DEVELOPMENT_ONLY', 'test_examples_evaluated': 0, 'comparison': rows,
               'limitations': ['One seed per neural variant so far',
                  'Historical ML and neural models have different tuning budgets',
                  'Same learning rate and epoch budget for both neural variants; frozen head may need different tuning',
                  'Development-selected scores are optimistic and do not replace final held-out testing',
                  'Training elapsed time includes checkpoint saving and is not an inference latency benchmark'],
               'classical_reference': 'artifacts/runs/phase1_extraction_verified_20261101/metrics.json'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(args.output / 'comparison.csv', index=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
