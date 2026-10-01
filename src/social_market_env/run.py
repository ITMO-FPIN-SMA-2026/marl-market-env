"""Command line entry point: python -m social_market_env.run <config.yaml>."""

import argparse
import logging
from pathlib import Path

from social_market_env.config import load_config
from social_market_env.runner.experiment import run_experiment


def main(argv: list[str] | None = None) -> int:
    """Run an experiment from a YAML config and print the run directory."""
    parser = argparse.ArgumentParser(
        prog='python -m social_market_env.run', description='Run an experiment from a YAML config.'
    )
    parser.add_argument('config', type=Path, help='path to the experiment YAML')
    parser.add_argument('--runs-dir', type=Path, default=Path('runs'), help='where run directories are created')
    parser.add_argument('--strict', action='store_true', help='stop on the first error in team code')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    config = load_config(args.config)
    if args.strict:
        config.strict = True
    print(run_experiment(config=config, runs_root=args.runs_dir))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
