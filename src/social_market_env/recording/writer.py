"""Run directory with meta data and per-episode step logs."""

import json
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from social_market_env import CONTRACTS_VERSION, __version__
from social_market_env.config import ExperimentConfig, config_to_dict


class EpisodeWriter:
    """Writes episode.json and streams step records of one episode to steps.jsonl."""

    def __init__(self, *, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self._steps = (directory / 'steps.jsonl').open('w', encoding='utf-8')

    def write_episode_info(self, info: dict[str, Any]) -> None:
        """Write episode.json with what the episode was started with."""
        _write_json(path=self.directory / 'episode.json', data=info)

    def write_step(self, record: dict[str, Any]) -> None:
        """Append one step record."""
        self._steps.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + '\n')

    def close(self, *, summary: dict[str, Any], slow_calls: dict[str, int]) -> None:
        """Finish the step log and write the summary and timings."""
        self._steps.close()
        _write_json(path=self.directory / 'summary.json', data=summary)
        _write_json(path=self.directory / 'timings.json', data={'slow_calls': slow_calls})

    def abort(self) -> None:
        """Close the step log without writing a summary."""
        self._steps.close()


class RunRecorder:
    """Creates runs/<stamp>_<name>/ with meta.json and hands out episode writers."""

    def __init__(self, *, runs_root: Path, config: ExperimentConfig) -> None:
        stamp = datetime.now(tz=UTC).strftime('%Y%m%d-%H%M%S')
        self.run_dir = _unique_directory(runs_root / f'{stamp}_{config.name}')
        _write_json(path=self.run_dir / 'meta.json', data=build_meta(config))

    def open_episode(self, *, seed: int, episode: int) -> EpisodeWriter:
        """Return a writer for one episode."""
        return EpisodeWriter(directory=self.run_dir / f'seed_{seed}' / f'episode_{episode}')


def build_meta(config: ExperimentConfig) -> dict[str, Any]:
    """Return what is needed to reproduce and identify a run."""
    specs = [config.agents.default, *config.agents.overrides.values(), config.coalitions.policy, config.arbiter.policy]
    return {
        'config': config_to_dict(config),
        'environment_version': __version__,
        'contracts_version': CONTRACTS_VERSION,
        'component_versions': {spec.impl: _package_version(spec.impl) for spec in specs},
        'created_at': datetime.now(tz=UTC).isoformat(),
    }


def _package_version(impl: str) -> str:
    package = impl.split(':')[0].split('.')[0]
    try:
        return version(package)
    except PackageNotFoundError:
        return 'unknown'


def _unique_directory(base: Path) -> Path:
    candidate = base
    suffix = 2
    while candidate.exists():
        candidate = base.with_name(f'{base.name}_{suffix}')
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def _write_json(*, path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, sort_keys=True, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
