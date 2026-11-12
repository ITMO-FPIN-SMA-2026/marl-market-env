"""Install only new, reviewed files; refuse to replace any existing destination."""

import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
TARGET = WORKSPACE.parent / "sma_itmo/marl-market-env"
DATA = WORKSPACE / "runs/team_transfer"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    prepared = json.loads((DATA / "prepared.json").read_text())
    assert TARGET.resolve() == Path(prepared["environment"]).resolve()
    for name, digest in prepared["environment_files"].items():
        if sha(TARGET / name) != digest:
            raise RuntimeError(f"Native source changed: {name}")
    destinations = {
        HERE / "transferred_ppo_gru": TARGET / "src/transferred_ppo_gru",
        HERE / "run_transferred_ppo.py": TARGET / "run_transferred_ppo.py",
        HERE / "TRANSFERRED_PPO.md": TARGET / "TRANSFERRED_PPO.md",
        DATA: TARGET / "runs/transferred_ppo",
    }
    for destination in destinations.values():
        if destination.exists():
            raise FileExistsError(f"Refusing to replace {destination}")
    files = {}
    for source, destination in destinations.items():
        if source.is_dir():
            shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
            pairs = [(p, destination / p.relative_to(source)) for p in source.rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
        else:
            shutil.copy2(source, destination)
            pairs = [(source, destination)]
        for original, installed in pairs:
            digest = sha(original)
            if sha(installed) != digest:
                raise RuntimeError(f"Copy differs: {installed}")
            files[str(installed.relative_to(TARGET))] = digest
    for name, digest in prepared["environment_files"].items():
        assert sha(TARGET / name) == digest, name
    record = {"native_environment_unchanged": True, "files": files}
    (TARGET / "runs/transferred_ppo/installation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({"installed": str(TARGET), "verified_files": len(files)}))


if __name__ == "__main__":
    main()
