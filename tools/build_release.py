"""Build a deployable archive without local credentials, data, or development output."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / ".artifacts" / "missingly-release.zip"
EXCLUDED_PARTS = {".git", ".venv", ".pytest_cache", "__pycache__", ".artifacts"}
EXCLUDED_NAMES = {".env", "users.db", "server.out.log", "server.err.log"}
EXCLUDED_SUFFIXES = {".db", ".log", ".pyc", ".zip"}


def should_include(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    if any(part in EXCLUDED_PARTS for part in relative.parts):
        return False
    if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
        return False
    return path.is_file()


def build_release(output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(path for path in ROOT.rglob("*") if should_include(path))
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    return len(files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count = build_release(args.output.resolve())
    print(f"Built {args.output.resolve()} with {count} files.")
