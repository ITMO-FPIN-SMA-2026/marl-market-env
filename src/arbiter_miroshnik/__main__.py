"""Run Miroshnik PPO + GRU from the current checkout."""

from pathlib import Path
import sys

from .research.cli import main


def entrypoint(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    root = Path(__file__).resolve().parents[2]
    if not any(arg == '--environment-repo' or arg.startswith('--environment-repo=') for arg in arguments):
        arguments += ['--environment-repo', str(root)]
    if not any(arg == '--out' or arg.startswith('--out=') for arg in arguments):
        arguments += ['--out', str(root / 'runs/transferred_ppo')]
    return main(arguments)


if __name__ == '__main__':
    entrypoint()
