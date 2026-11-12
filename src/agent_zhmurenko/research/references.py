"""Load unchanged peer Git sources and the preserved V4 under separate names."""

import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from ..agent import output_path

PEERS = {'askarov_agent': 'e2d6e7ae9813125c772bcca64e220dd39f548244',
         'tverdokhleb_agent': '1c1a437127afbe44eed8fad77243b37be82410b9',
         'rodina_agent': '889475f55d6fc37fa8c8256347df8a0716c38468'}


def prepare(root, *, v4_archive=None):
    root = output_path(root)
    source = root / 'reference-runtime'
    source.mkdir(parents=True, exist_ok=True)
    repository = next(p for p in Path(__file__).resolve().parents if (p / '.git').exists())
    manifest = {}
    for package, commit in PEERS.items():
        paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', commit, f'src/{package}'], cwd=repository, text=True).splitlines()
        hashes = {}
        for name in paths:
            if not name.endswith('.py'):
                continue
            data = subprocess.check_output(['git', 'show', f'{commit}:{name}'], cwd=repository)
            destination = source / name.removeprefix('src/')
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        manifest[package] = dict(commit=commit, source_sha256=hashes)
    sys.path.insert(0, str(source)) if str(source) not in sys.path else None
    if v4_archive is not None:
        prefix = 'src/social_market_env/implementations/zhmurenko_vladimir/'
        legacy = source / 'v4_reference'
        with zipfile.ZipFile(v4_archive) as archive:
            for name in archive.namelist():
                if name.startswith(prefix) and name.endswith('.py'):
                    relative = Path(name.removeprefix(prefix))
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('Invalid reference archive path')
                    destination = legacy / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(archive.read(name))
        spec = importlib.util.spec_from_file_location('v4_reference', legacy / '__init__.py', submodule_search_locations=[str(legacy)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['v4_reference'] = module
        spec.loader.exec_module(module)
        manifest['v4_archive_sha256'] = hashlib.sha256(Path(v4_archive).read_bytes()).hexdigest()
    (root / 'reference-sources.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def askarov(*, market=None, model_path=None):
    cls = importlib.import_module('askarov_agent').AskarovAgent
    return cls.from_checkpoint(model_path, market=market) if model_path else cls(market=market)


def tverdokhleb(*, market=None, model_path=None):
    return importlib.import_module('tverdokhleb_agent').TverdokhlebAgent(market=market, model_path=model_path)


def activate(root, *, v4=False):
    source = Path(root) / 'reference-runtime'
    if not source.is_dir():
        raise FileNotFoundError('Run the prepare stage first')
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    if v4 and 'v4_reference' not in sys.modules:
        legacy = source / 'v4_reference'
        spec = importlib.util.spec_from_file_location('v4_reference', legacy / '__init__.py', submodule_search_locations=[str(legacy)])
        module = importlib.util.module_from_spec(spec)
        sys.modules['v4_reference'] = module
        spec.loader.exec_module(module)
