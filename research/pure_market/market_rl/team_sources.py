"""Read real team implementations from local Git refs without changing them."""

import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

BRANCHES = {
    "agent-askarov": "src/askarov_agent",
    "agent-tverdokhleb": "src/tverdokhleb_agent",
    "agent-rodina": "src/rodina_agent",
    "agent-zhmurenko": "src/social_market_env/implementations/zhmurenko_vladimir",
    "coalition": "src/coalition_policy",
}
IMPLEMENTATIONS = (
    "askarov_agent:AskarovAgent", "tverdokhleb_agent:TverdokhlebAgent",
    "rodina_agent:RodinaAgent",
    "social_market_env.implementations.zhmurenko_vladimir.agent:ZhmurenkoVladimirAgent",
)


def prepare_sources(repository, directory):
    repository, directory = Path(repository).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    revisions = {}
    for branch, path in BRANCHES.items():
        revision = subprocess.check_output(
            ["git", "-C", str(repository), "rev-parse", "origin/" + branch], text=True).strip()
        revisions[branch] = revision
        archive = subprocess.check_output(["git", "-C", str(repository), "archive", revision, path])
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(directory, filter="data")
    files = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(directory.rglob("*")) if p.is_file()}
    manifest = {"repository": str(repository), "revisions": revisions,
                "implementations": IMPLEMENTATIONS, "files": files}
    (directory / "sources.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def activate_sources(repository, directory):
    """Use the ACTUAL environment; extend only its implementation search path."""
    repository, directory = Path(repository).resolve(), Path(directory).resolve()
    sys.path[:0] = [str(repository / "src"), str(directory / "src")]
    import social_market_env

    if not Path(social_market_env.__file__).resolve().is_relative_to(repository / "src"):
        raise RuntimeError("Another social_market_env is already imported")
    extension = str(directory / "src/social_market_env")
    if extension not in social_market_env.__path__:
        social_market_env.__path__.append(extension)
