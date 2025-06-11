import argparse
import os
import subprocess
import re
import matplotlib.pyplot as plt


def parse_metrics(output):
    """Extract accuracy, precision, recall and F1 from solver output."""
    pattern = r"Accuracy\s*:\s*([0-9\.]+),\s*Precision\s*:\s*([0-9\.]+),\s*Recall\s*:\s*([0-9\.]+),\s*F-score\s*:\s*([0-9\.]+)"
    match = re.search(pattern, output)
    if match:
        return {
            "accuracy": float(match.group(1)),
            "precision": float(match.group(2)),
            "recall": float(match.group(3)),
            "f1": float(match.group(4)),
        }
    return None

def run_phase(args, tag, start, end, load_model=None):
    """Train and evaluate a model on a data segment."""
    train_cmd = [
        'python', 'main.py',
        '--mode', 'train',
        '--dataset', args.dataset,
        '--data_path', args.data_path,
        '--win_size', str(args.win_size),
        '--input_c', str(args.input_c),
        '--output_c', str(args.output_c),
        '--batch_size', str(args.batch_size),
        '--num_epochs', str(args.num_epochs),
        '--anormly_ratio', str(args.anormly_ratio),
        '--model_save_path', args.model_save_path,
        '--model_tag', tag,
        '--train_start', str(start),
        '--train_end', str(end)
    ]
    if load_model:
        train_cmd += ['--load_model', load_model]
    subprocess.run(train_cmd, check=True)

    test_cmd = [
        'python', 'main.py',
        '--mode', 'test',
        '--dataset', args.dataset,
        '--data_path', args.data_path,
        '--win_size', str(args.win_size),
        '--input_c', str(args.input_c),
        '--output_c', str(args.output_c),
        '--batch_size', str(args.batch_size),
        '--anormly_ratio', str(args.anormly_ratio),
        '--model_save_path', args.model_save_path,
        '--model_tag', tag
    ]
    result = subprocess.run(test_cmd, capture_output=True, text=True, check=True)
    metrics = parse_metrics(result.stdout)
    if metrics:
        print(
            f"Performance for {tag} - Accuracy: {metrics['accuracy']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, "
            f"F-score: {metrics['f1']:.4f}"
        )
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, required=True)
    parser.add_argument('--output_c', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--anormly_ratio', type=float, default=1.0)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    args = parser.parse_args()

    os.makedirs(args.model_save_path, exist_ok=True)

    inc_results = []

    # initial training with first 50%
    metrics = run_phase(args, 'inc_init50', 0.0, 0.5)
    if metrics:
        inc_results.append((0.5, metrics))
    prev_tag = 'inc_init50'

    # incremental updates
    start = 0.5
    for i in range(5):
        end = start + 0.1
        tag = f'inc_update_{i+1}'
        load = os.path.join(args.model_save_path, f'{prev_tag}_checkpoint.pth')
        metrics = run_phase(args, tag, start, end, load)
        if metrics:
            inc_results.append((end, metrics))
        prev_tag = tag
        start = end

    # baseline models trained from scratch with varying fractions
    baseline_results = []
    for frac in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        tag = f'baseline_{int(frac*100)}'
        metrics = run_phase(args, tag, 0.0, frac)
        if metrics:
            baseline_results.append((frac, metrics))

    if inc_results and baseline_results:
        inc_fracs = [r[0] for r in inc_results]
        inc_f1s = [r[1]['f1'] for r in inc_results]
        base_fracs = [r[0] for r in baseline_results]
        base_f1s = [r[1]['f1'] for r in baseline_results]
        plt.figure()
        plt.plot(inc_fracs, inc_f1s, marker='o', label='Incremental')
        plt.plot(base_fracs, base_f1s, marker='s', label='Baseline')
        plt.xlabel('Training Data Fraction')
        plt.ylabel('F1 Score')
        plt.title('Incremental vs. Baseline Performance')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(args.model_save_path, 'incremental_results.png'))


if __name__ == '__main__':
    main()
