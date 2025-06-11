import argparse
import os
import subprocess


def run_phase(args, tag, start, end, load_model=None):
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
    subprocess.run(test_cmd, check=True)


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

    # initial training with first 50%
    run_phase(args, 'init50', 0.0, 0.5)
    prev_tag = 'init50'

    # incremental updates
    start = 0.5
    for i in range(5):
        end = start + 0.1
        tag = f'update_{i+1}'
        load = os.path.join(args.model_save_path, f'{prev_tag}_checkpoint.pth')
        run_phase(args, tag, start, end, load)
        prev_tag = tag
        start = end

    # full data baseline
    run_phase(args, 'full_batch', 0.0, 1.0)


if __name__ == '__main__':
    main()
