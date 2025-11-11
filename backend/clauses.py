import re
from typing import List, Tuple


def normalize_clause_numbers(text: str) -> str:
    return re.sub(r"(\d),(\d)", r"\1.\2", text)


def split_candidate_headers(text: str) -> List[Tuple[int, str]]:
    t = normalize_clause_numbers(text)
    pattern = re.compile(
        r"(?:(?:^|\n|[\.!?]\s))"  # boundary
        r"(?:Section|Clause|Article)?\s*"
        r"(\d{1,2}(?:\.\d{1,2})?)\s*"  # number
        r"(?:[:\-\.]\s+)?"  # punctuation
        r"(?!\()"  # not subsection like (b)
        r"([A-Z][^\n]{0,80})?",
        re.IGNORECASE,
    )
    return [(m.start(1), m.group(1)) for m in pattern.finditer(t)]


def is_real_header(full_text: str, header_line: str) -> bool:
    forbidden = re.compile(r"\b(below|above|pursuant|provided|as defined|per|see)\b", re.IGNORECASE)
    return len(header_line) <= 80 and not forbidden.search(header_line or "")


def split_into_clauses(text: str) -> List[str]:
    norm = normalize_clause_numbers(text)
    headers = split_candidate_headers(norm)
    if headers:
        merged = []
        for pos, num in headers:
            if merged and merged[-1][1] == num and pos - merged[-1][0] < 40:
                continue
            merged.append((pos, num))
        clauses: List[str] = []
        for i, (pos, _num) in enumerate(merged):
            end = merged[i + 1][0] if i + 1 < len(merged) else len(norm)
            start_line = norm.rfind("\n", 0, pos) + 1
            chunk = norm[start_line:end].strip()
            if chunk:
                clauses.append(chunk)
        if 3 <= len(clauses) <= max(150, len(norm) // 150):
            return clauses
    # Fallback
    return [p.strip() for p in re.split(r"\n{2,}", norm) if len(p.strip()) > 40]


def format_clause(raw_text: str, layout_titles: List[str] | None = None) -> str:
    text = raw_text.strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    header_line = lines[0] if lines else ""

    header_match = re.match(r"^(?:Section|Clause|Article)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*[-:.)]?\s*(.*)$", header_line, re.IGNORECASE)
    looks_short = len(header_line) <= 80
    forbidden = re.compile(r"\b(below|above|pursuant|provided|as defined|per|see)\b", re.IGNORECASE)
    not_cross = not forbidden.search(header_line)
    is_real = bool(header_match) and looks_short and not_cross

    if is_real:
        clause_no = header_match.group(1)
        clause_title = header_match.group(2).strip()
    else:
        any_num = re.search(r"(\d{1,2}(?:\.\d{1,2})?)", header_line)
        clause_no = any_num.group(1) if any_num else ""
        clause_title = ""

    if not is_real and layout_titles:
        def norm(s: str) -> str:
            return re.sub(r"\s+", " ", s or "").strip().lower()
        nt = [norm(t) for t in layout_titles if isinstance(t, str)]
        if nt:
            starts = norm(text)
            for t in nt:
                if len(t) > 4 and (starts.startswith(t) or norm(header_line).startswith(t) or t.startswith(norm(header_line))):
                    clause_title = layout_titles[0]
                    m = re.match(r"^(?:Section|Clause|Article)?\s*(\d{1,2}(?:\.\d{1,2})?)\b", text, re.IGNORECASE)
                    clause_no = m.group(1) if m else clause_no
                    is_real = True
                    break

    body_text = text
    if is_real and header_line and len(lines) > 1:
        body_text = "\n".join(lines[1:])
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", body_text) if p.strip()]
    normalized = [" ".join(p.split()) for p in paragraphs] or [" ".join(body_text.split())]

    heading_parts = []
    if clause_no:
        heading_parts.append(clause_no)
    if clause_title:
        heading_parts.append(clause_title)
    heading = " ".join(heading_parts).strip()

    if not heading:
        return "\n".join(["  " + p for p in normalized]).strip()
    return (heading + ":\n" + "\n".join(["  " + p for p in normalized])).strip()


