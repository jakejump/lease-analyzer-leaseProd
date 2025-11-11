from __future__ import annotations

from pathlib import Path


def _project_root() -> Path:
    # Assumes this file lives under backend/paths.py
    return Path(__file__).resolve().parents[1]


def _temp_root() -> Path:
    root = _project_root() / "temp"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _doc_dir(doc_id: str) -> Path:
    base = _temp_root() / doc_id
    base.mkdir(parents=True, exist_ok=True)
    return base


def doc_id_from_pdf_path(pdf_path: str | Path) -> str:
    p = Path(pdf_path)
    # Expected pattern: temp/<doc_id>/lease.pdf
    try:
        doc_id = p.parent.name
        assert len(doc_id) >= 8
        return doc_id
    except Exception:
        # Fallback: use parent folder name even if it doesn't look like a hash
        return p.parent.name

from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _temp_root() -> Path:
    temp_dir = _project_root() / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def _doc_dir(doc_id: str) -> Path:
    directory = _temp_root() / doc_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def doc_id_from_pdf_path(pdf_path: str | Path) -> str:
    return Path(pdf_path).resolve().parent.name


