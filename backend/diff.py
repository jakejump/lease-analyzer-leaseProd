from __future__ import annotations

import re
from typing import List, Dict
from backend.lease_chain import get_or_build_clauses_for_doc
from backend.paths import _doc_dir


def _parse_clause_number_and_body(text: str) -> tuple[str | None, str]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return None, ""
    header = lines[0]
    m = re.match(r"^(?:Section|Clause|Article)?\s*(\d{1,2}(?:\.\d{1,2})?)\b[:\-\.)]?\s*(.*)$", header, re.IGNORECASE)
    if m:
        num = m.group(1)
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""
        return num, body.strip()
    # fallback: try to find a number anywhere
    m2 = re.search(r"(\d{1,2}(?:\.\d{1,2})?)", header)
    return (m2.group(1) if m2 else None), "\n".join(lines[1:]).strip()


def _index_by_number(clauses: List[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for c in clauses:
        num, body = _parse_clause_number_and_body(c)
        if num:
            out[num] = body or c
    return out


def _modified(a: str, b: str) -> bool:
    def _norm(s: str) -> str:
        return re.sub(r"\s+", " ", s or "").strip().lower()
    return _norm(a) != _norm(b)


def diff_pdfs(base_pdf: str, compare_pdf: str) -> List[Dict[str, str]]:
    base_id = base_pdf.split("/")[-2]
    compare_id = compare_pdf.split("/")[-2]
    base_clauses = get_or_build_clauses_for_doc(base_id)
    compare_clauses = get_or_build_clauses_for_doc(compare_id)

    ai = _index_by_number(base_clauses)
    bi = _index_by_number(compare_clauses)

    changes: list[dict[str, str]] = []

    # Modified or removed
    for num, body in ai.items():
        if num not in bi:
            changes.append({"type": "removed", "clause": num, "before": body, "after": ""})
        else:
            if _modified(body, bi[num]):
                changes.append({"type": "modified", "clause": num, "before": body, "after": bi[num]})

    # Added
    for num, body in bi.items():
        if num not in ai:
            changes.append({"type": "added", "clause": num, "before": "", "after": body})

    return changes

from typing import List, Dict, Tuple
import re
from difflib import SequenceMatcher

from backend.lease_chain import get_or_build_clauses_for_doc
from backend.paths import doc_id_from_pdf_path


def _parse_clause_number_and_body(clause_text: str) -> Tuple[str | None, str]:
    text = clause_text.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_line = lines[0] if lines else ""
    m = re.match(r"^(?:Section|Clause|Article)?\s*(\d{1,2}(?:\.\d{1,2})?)\b[\s\-:\.)]*", header_line, re.IGNORECASE)
    number = m.group(1) if m else None
    body = "\n".join(lines[1:]) if len(lines) > 1 else ("\n".join(lines) if lines else "")
    body = " ".join(body.split())
    return number, body


def _index_by_number(clauses: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for c in clauses:
        num, body = _parse_clause_number_and_body(c)
        if num:
            out[num] = body
    return out


def _modified(a: str, b: str, threshold: float = 0.9) -> bool:
    # Simple similarity on normalized strings
    a_n = " ".join(a.split())
    b_n = " ".join(b.split())
    ratio = SequenceMatcher(None, a_n, b_n).ratio()
    return ratio < threshold


def diff_pdfs(base_pdf_path: str, compare_pdf_path: str) -> List[Dict]:
    base_id = doc_id_from_pdf_path(base_pdf_path)
    compare_id = doc_id_from_pdf_path(compare_pdf_path)

    # Use cached clauses; this will compute once and persist if missing
    base_clauses = get_or_build_clauses_for_doc(base_id)
    compare_clauses = get_or_build_clauses_for_doc(compare_id)

    base_idx = _index_by_number(base_clauses)
    compare_idx = _index_by_number(compare_clauses)

    changes: List[Dict] = []

    # Removed or modified
    for num, before in base_idx.items():
        if num not in compare_idx:
            changes.append({"type": "removed", "clause_no": num, "before": before, "after": None})
        else:
            after = compare_idx[num]
            if _modified(before, after):
                changes.append({"type": "modified", "clause_no": num, "before": before, "after": after})

    # Added
    for num, after in compare_idx.items():
        if num not in base_idx:
            changes.append({"type": "added", "clause_no": num, "before": None, "after": after})

    return changes


