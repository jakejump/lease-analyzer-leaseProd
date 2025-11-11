from __future__ import annotations

# Latest uploaded document id (used by simple, non-project endpoints)
LATEST_DOC_ID: str | None = None

# Process-wide cache keyed by doc_id. Values may include:
# - "retriever": a retriever object
# - "clauses": list[str]
DOC_CACHE: dict[str, dict] = {}

from typing import Dict, Any, Optional


LATEST_DOC_ID: Optional[str] = None
DOC_CACHE: Dict[str, Dict[str, Any]] = {}


