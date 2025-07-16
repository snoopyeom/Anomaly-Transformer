import os
import json
import argparse
import logging


def load_aas_files(directory: str) -> list[dict]:
    """Load AAS JSON files from ``directory``.

    Parameters
    ----------
    directory : str
        Directory containing ``.json`` files.

    Returns
    -------
    list[dict]
        Loaded JSON objects.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.json')]
    logging.debug(f"Found JSON files: {files}")
    aas_list: list[dict] = []
    for fname in files:
        path = os.path.join(directory, fname)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.debug(f"Loaded {fname}: {data.get('name')}")
        aas_list.append(data)
    return aas_list


def filter_candidates(aas_list: list[dict], status: str = 'active') -> list[dict]:
    """Filter AAS candidates by ``status``."""
    candidates = [aas for aas in aas_list if aas.get('status') == status]
    logging.debug(
        f"Candidates after filtering ({status}): {[c.get('name') for c in candidates]}"
    )
    return candidates


def choose_best_candidate(candidates: list[dict]) -> dict | None:
    """Return candidate with highest ``score``."""
    if not candidates:
        logging.warning('No candidates found')
        return None
    best = max(candidates, key=lambda a: a.get('score', 0))
    logging.debug(f"Chosen candidate: {best.get('name')}")
    return best


def visualize_flow(aas_list: list[dict], out_path: str) -> None:
    """Write a simple HTML visualization of the candidate flow."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('<html><body><ul>')
        for aas in aas_list:
            f.write(f"<li>{aas.get('name')}</li>")
        f.write('</ul></body></html>')
    logging.info(f"Saved flow visualisation to '{out_path}'.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Find best AAS candidate.')
    parser.add_argument('--aas-dir', required=True, help='Directory with AAS JSON files')
    parser.add_argument('--status', default='active', help='Status used for filtering')
    parser.add_argument('--log-level', default='INFO', help='Logging level (e.g. DEBUG)')
    parser.add_argument('--visualize', action='store_true', help='Save process flow HTML')
    parser.add_argument('--output', default='process_flow.html', help='HTML output path')
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format='%(message)s'
    )

    logging.info(f"Reading AAS files from {args.aas_dir}")
    aas_list = load_aas_files(args.aas_dir)
    if args.visualize:
        visualize_flow(aas_list, args.output)
    candidates = filter_candidates(aas_list, status=args.status)
    best = choose_best_candidate(candidates)
    if best:
        print(best.get('name', ''))
    else:
        print('')


if __name__ == '__main__':
    main()
