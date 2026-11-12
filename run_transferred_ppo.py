"""Run from the native marl-market-env checkout; add no dependencies to its core."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from arbiter_miroshnik.research.cli import main

if __name__ == "__main__":
    arguments = sys.argv[1:]
    if "--environment-repo" not in arguments:
        arguments += ["--environment-repo", str(ROOT)]
    if "--out" not in arguments:
        arguments += ["--out", str(ROOT / "runs/transferred_ppo")]
    main(arguments)
