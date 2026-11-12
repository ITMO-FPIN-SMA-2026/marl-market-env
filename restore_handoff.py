"""Restore every archived stage, verifying hashes and preserving existing states."""

import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def restore(manifest):
    record = json.loads(manifest.read_text(encoding='utf-8'))
    archive = ROOT / record['archive']
    if not archive.exists() or archive.stat().st_size != record['bytes']:
        raise RuntimeError(f'Copy the separately transferred {archive.name} into {archive.parent}; '
                           'archive missing or its size differs from the manifest')
    if digest(archive) != record['sha256']:
        raise RuntimeError(f'Archive checksum differs: {archive}')
    missing = []
    for row in record['files']:
        path = ROOT / row['path']
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError('Archive path escapes the checkout')
        if path.exists():
            if digest(path) != row['sha256']:
                raise FileExistsError(f'Preserving changed state: {path}. Restore into a fresh clone.')
        else:
            missing.append(row)
    if missing:
        with tempfile.TemporaryDirectory(prefix='.handoff-restore-', dir=ROOT) as directory:
            staging = Path(directory)
            with tarfile.open(archive, 'r:gz') as tar:
                tar.extractall(staging, filter='data')
            for row in record['files']:
                if digest(staging / row['path']) != row['sha256']:
                    raise RuntimeError(f'Extracted file checksum differs: {row["path"]}')
            for row in missing:
                destination = ROOT / row['path']
                destination.parent.mkdir(parents=True, exist_ok=True)
                (staging / row['path']).replace(destination)
    print(f'{manifest.name}: verified {len(record["files"])} files; restored {len(missing)}', flush=True)


def main():
    for manifest in sorted((ROOT / 'artifacts').glob('*_manifest.json')):
        restore(manifest)
    source = ROOT / 'research/pure_market/runs/team_transfer'
    target = ROOT / 'runs/transferred_ppo'
    if not target.exists():
        shutil.copytree(source, target)
        print('Restored the native 64-game continuation for all three models.', flush=True)
    print('Ready: python run_transferred_ppo.py evaluate --episodes 20', flush=True)


if __name__ == '__main__':
    main()
