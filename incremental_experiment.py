"""Run continual training where CPD triggers replay."""

import argparse
import os

from main import main as run_main


def train_and_test(args: argparse.Namespace) -> None:
    """Train then evaluate, printing separate summaries."""
    args.mode = "train"
    solver = run_main(args)

    if hasattr(solver, "update_count"):
        print(f"Total CPD updates: {solver.update_count}")

    if getattr(solver, "history", None):
        last_f1 = solver.history[-1][1]
        last_auc = solver.history[-1][2]
        print(
            f"Continual learning - Final F1: {last_f1:.4f}, AUC: {last_auc:.4f}")

    args.mode = "test"
    args.load_model = os.path.join(
        args.model_save_path, f"{args.model_tag}_checkpoint.pth")
    solver.load_model = args.load_model
    acc, prec, rec, f1, auc = solver.test()
    print(
        "Batch evaluation - Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, "
        "F1: {:.4f}, AUC: {:.4f}".format(acc, prec, rec, f1, auc))



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--input_c', type=int, required=True)
    parser.add_argument('--output_c', type=int, required=True)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--anomaly_ratio', type=float, default=1.0)
    parser.add_argument('--model_save_path', type=str, default='checkpoints')
    parser.add_argument('--min_cpd_gap', type=int, default=30,
                        help='minimum gap between CPD change points')

    parser.add_argument(
        '--model_type',
        type=str,
        default='transformer_vae',
        choices=['transformer', 'transformer_vae'],
        help='VAE 브랜치를 사용하려면 transformer_vae 선택')
    parser.add_argument('--latent_dim', type=int, default=16)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--replay_horizon', type=int, default=None)
    parser.add_argument('--store_mu', action='store_true',
                        help='store mu/logvar instead of sampled z')
    parser.add_argument('--freeze_after', type=int, default=None,
                        help='freeze encoder after this many steps')
    parser.add_argument('--ema_decay', type=float, default=None,
                        help='EMA decay for encoder stabilization')
    parser.add_argument('--decoder_type', type=str, default='mlp',
                        choices=['mlp', 'rnn', 'attention'],
                        help='decoder architecture')
    parser.add_argument('--model_tag', type=str, default='dynamic')
    parser.add_argument(
        '--cpd_penalty',
        type=int,
        default=20,
        help='Penalty value for ruptures change point detection',
    )
    parser.add_argument('--cpd_top_k', type=int, default=3,
                        help='number of zoomed views for CPD visualization')
    parser.add_argument(
        '--cpd_extra_ranges', type=str, default='0:4000',
        help='comma-separated start:end pairs for additional CPD zoom views'
    )
    parser.add_argument('--cpd_log_interval', type=int, default=20,
                        help='log metrics every N CPD updates')

    def _parse_ranges(arg):
        if not arg:
            return None
        pairs = []
        for part in arg.split(','):
            if ':' not in part:
                continue
            start, end = part.split(':', 1)
            pairs.append((int(start), int(end)))
        return pairs or None

    args = parser.parse_args()
    args.cpd_extra_ranges = _parse_ranges(args.cpd_extra_ranges)

    os.makedirs(args.model_save_path, exist_ok=True)

    train_and_test(args)


if __name__ == '__main__':
    main()
