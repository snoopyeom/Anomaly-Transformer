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
    args = parser.parse_args()

    os.makedirs(args.model_save_path, exist_ok=True)

    train_and_test(args)


if __name__ == '__main__':
    main()
