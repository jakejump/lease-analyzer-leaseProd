from __future__ import annotations

import shutil
from pathlib import Path
from backend.paths import _project_root


def _storage_root() -> Path:
    root = _project_root() / "temp" / "storage"
    root.mkdir(parents=True, exist_ok=True)
    return root


def put_file(local_path: str | Path, relative_destination: str) -> str:
    src = Path(local_path)
    dst = _storage_root() / relative_destination
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    # Return a pseudo-URL (file path) usable by the app
    return str(dst)

import os
import shutil
from pathlib import Path
from typing import BinaryIO

from backend.paths import _project_root


def _storage_root() -> Path:
    root = _project_root() / "storage"
    root.mkdir(parents=True, exist_ok=True)
    return root


def put_file(src_path: str, dest_rel: str) -> str:
    """Copy a file into storage under dest_rel and return absolute path as URL placeholder."""
    dest_abs = _storage_root() / dest_rel
    dest_abs.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path, dest_abs)
    return str(dest_abs)


def put_bytes(data: bytes, dest_rel: str) -> str:
    dest_abs = _storage_root() / dest_rel
    dest_abs.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_abs, "wb") as f:
        f.write(data)
    return str(dest_abs)


def exists(url: str) -> bool:
    return Path(url).exists()


def get_file(url: str) -> bytes:
    with open(url, "rb") as f:
        return f.read()


