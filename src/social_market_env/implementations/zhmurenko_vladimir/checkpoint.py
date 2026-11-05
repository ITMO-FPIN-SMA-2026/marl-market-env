"""Versioned CPU checkpoints, atomic writes, and explicit output paths."""

import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from .features import FEATURE_NAMES, FEATURE_VERSION
from .settings import writable_path

CHECKPOINT_VERSION = 2


def read_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location='cpu', weights_only=True)
    if payload.get('checkpoint_version') not in (1, CHECKPOINT_VERSION):
        raise ValueError('Unsupported checkpoint version')
    if payload.get('feature_version') != FEATURE_VERSION or tuple(payload.get('feature_names', ())) != FEATURE_NAMES:
        raise ValueError('Checkpoint feature schema does not match this policy')
    return payload


def write_checkpoint(path: str | Path, payload: dict[str, Any], *, overwrite: bool = False) -> Path:
    destination = writable_path(path)
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=destination.name + '.', suffix='.tmp', dir=destination.parent)
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return destination
