"""Run continual training with CPD-triggered replay."""

import argparse
import os

from main import main as run_main



def train_and_test(args: argparse.Namespace) -> None:
    """Train then evaluate using the given arguments."""
    args.mode = "train"
    run_main(args)
    args.mode = "test"
    args.load_model = os.path.join(args.model_save_path,
                                   f"{args.model_tag}_checkpoint.pth")
    run_main(args)

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
        '--model_type', args.model_type,
        '--latent_dim', str(args.latent_dim),
        '--beta', str(args.beta),
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
        '--model_type', args.model_type,
        '--latent_dim', str(args.latent_dim),
        '--beta', str(args.beta),
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

    parser.add_argument('--model_type', type=str, default='transformer_vae',
                        choices=['transformer', 'transformer_vae'])
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--model_tag', type=str, default='dynamic')

    parser.add_argument('--model_type', type=str, default='transformer',
                        choices=['transformer', 'transformer_vae'])
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--beta', type=float, default=1.0)

    args = parser.parse_args()

    os.makedirs(args.model_save_path, exist_ok=True)

    train_and_test(args)


if __name__ == '__main__':
    main()
