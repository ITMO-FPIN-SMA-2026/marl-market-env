"""Keep pytest's temporary files separate for Windows security identities."""

import csv
import os
import subprocess
from pathlib import Path


def pytest_configure(config) -> None:
    """Avoid sharing pytest's owner-only temp directory across Windows accounts."""
    if os.name != 'nt' or config.option.basetemp is not None:
        return
    identity = subprocess.check_output(
        ['whoami', '/user', '/fo', 'csv', '/nh'], text=True
    )
    sid = next(csv.reader(identity.splitlines()))[1]
    project = Path(__file__).resolve().parents[1]
    config.option.basetemp = str(project / f'.pytest_tmp_{sid}')
