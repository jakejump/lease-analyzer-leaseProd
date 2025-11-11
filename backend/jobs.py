from __future__ import annotations

import json
import os
import shutil
from hashlib import md5
from redis import Redis
from rq import Queue
from backend.db import session_scope
from backend.models import LeaseVersion, LeaseVersionStatus, RiskScore, AbnormalityRecord
from backend.paths import _doc_dir
from backend.lease_chain import (
    _get_or_build_vectorstore_for_doc,
    get_or_build_clauses_for_doc,
    evaluate_general_risks,
    detect_abnormalities,
)


def get_queue() -> Queue:
    redis_url = os.getenv("REDIS_URL")
    conn = Redis.from_url(redis_url) if redis_url else Redis()
    return Queue("default", connection=conn)


def _set_progress(version_id: str, stage: str | None = None, progress: int | None = None) -> None:
    try:
        conn = Redis()
    except Exception:
        return
    key = f"version:{version_id}:status"
    data: dict[str, str] = {}
    if stage is not None:
        data["stage"] = stage
    if progress is not None:
        data["progress"] = str(progress)
    if data:
        try:
            conn.hset(key, mapping=data)
        except Exception:
            pass


def process_version(version_id: str) -> None:
    """Process a version: copy file into doc_dir, build index, compute analyses."""
    with session_scope() as s:
        v = s.get(LeaseVersion, version_id)
        if not v or not v.file_url:
            print(f"[job] abort: version {version_id} missing or no file_url")
            return
        try:
            print(f"[job] start {version_id} -> file {v.file_url}")
            _set_progress(version_id, stage="copy", progress=10)

            # Compute doc_id from file content
            with open(v.file_url, "rb") as f:
                content = f.read()
            digest = md5(content).hexdigest()
            v.doc_id = digest
            print(f"[job] computed doc_id={digest}")

            # Copy file into temp/<doc_id>/lease.pdf
            target_dir = _doc_dir(digest)
            target_pdf = target_dir / "lease.pdf"
            if not target_pdf.exists():
                shutil.copy(v.file_url, target_pdf)
                print(f"[job] copied file to {target_pdf}")

            _set_progress(version_id, stage="index", progress=40)
            print(f"[job] building vectorstore for {digest}")
            _get_or_build_vectorstore_for_doc(digest)
            print(f"[job] vectorstore ready for {digest}")

            _set_progress(version_id, stage="clauses", progress=55)
            print(f"[job] computing clauses for {digest}")
            get_or_build_clauses_for_doc(digest)
            print(f"[job] clauses cached for {digest}")

            _set_progress(version_id, stage="risk", progress=75)
            print(f"[job] evaluating risks for {digest}")
            risks = evaluate_general_risks(str(target_pdf))
            s.add(RiskScore(lease_version_id=v.id, payload=json.dumps(risks), model="gpt-4o"))
            print(f"[job] risk stored for version {v.id}")

            _set_progress(version_id, stage="abnormalities", progress=88)
            print(f"[job] detecting abnormalities for {digest}")
            abns = detect_abnormalities(str(target_pdf))
            s.add(AbnormalityRecord(lease_version_id=v.id, payload=json.dumps(abns), model="gpt-4o"))
            print(f"[job] abnormalities stored for version {v.id}")

            v.status = LeaseVersionStatus.processed
            s.flush()
            _set_progress(version_id, stage="done", progress=100)
            print(f"[job] done {version_id}")
        except Exception as e:
            print("[job] process_version error:", e)
            try:
                v.status = LeaseVersionStatus.failed
                s.flush()
            except Exception:
                pass
            _set_progress(version_id, stage="failed", progress=100)

import json
from rq import Queue
from redis import Redis
from typing import Optional

from backend.models import LeaseVersion, LeaseVersionStatus, RiskScore, AbnormalityRecord
from backend.db import session_scope
from backend.paths import _doc_dir
from backend.lease_chain import evaluate_general_risks, detect_abnormalities, _get_or_build_vectorstore_for_doc
from backend.state import DOC_CACHE as _DOC_CACHE
import logging


def get_queue() -> Queue:
    redis_url = None  # default localhost
    conn = Redis.from_url(redis_url) if redis_url else Redis()
    return Queue("default", connection=conn)


def _set_progress(version_id: str, stage: str, progress: int) -> None:
    try:
        conn = Redis()
        conn.hset(f"version:{version_id}:status", mapping={"stage": stage, "progress": progress})
    except Exception:
        pass


def process_version(version_id: str) -> None:
    """Run extraction/indexing/analyses for a lease version (simplified synchronous pipeline)."""
    with session_scope() as s:
        v: Optional[LeaseVersion] = s.get(LeaseVersion, version_id)
        if not v or not v.file_url:
            return
        # Copy file to expected working path if needed
        # For now assume v.file_url is accessible; put it under temp/{version_id}/lease.pdf for current pipeline
        # Build vectorstore
        try:
            _set_progress(version_id, "copy", 5)
            # Use stable doc_id per-version (no dedupe): doc_id = version_id
            import os, shutil
            os.makedirs("temp", exist_ok=True)
            temp_pdf = f"temp/{version_id}.pdf"
            if v.file_url and os.path.exists(v.file_url):
                shutil.copy(v.file_url, temp_pdf)
            from backend.lease_chain import _doc_dir
            doc_id = version_id
            # Persist doc_id early for better observability even if later steps fail
            v.doc_id = doc_id
            s.flush()
            target_dir = _doc_dir(doc_id)
            target_pdf = str(target_dir / "lease.pdf")
            if not os.path.exists(target_pdf):
                shutil.copy(temp_pdf, target_pdf)
            _set_progress(version_id, "index", 40)
            # Ensure we rebuild for this doc_id (no cache reuse)
            try:
                if doc_id in _DOC_CACHE:
                    del _DOC_CACHE[doc_id]
            except Exception:
                pass
            _get_or_build_vectorstore_for_doc(doc_id)
            # Precompute and persist clauses so downstream ops (diff, UI) don't re-extract text
            try:
                from backend.lease_chain import get_or_build_clauses_for_doc
                get_or_build_clauses_for_doc(doc_id)
            except Exception:
                pass

            # Analyses
            _set_progress(version_id, "risk", 60)
            risks = evaluate_general_risks(target_pdf)
            _set_progress(version_id, "abnormalities", 80)
            abn = detect_abnormalities(target_pdf)
            s.add(RiskScore(lease_version_id=version_id, payload=json.dumps(risks), model="gpt-4o"))
            s.add(AbnormalityRecord(lease_version_id=version_id, payload=json.dumps(abn), model="gpt-4o"))
            v.doc_id = doc_id
            v.status = LeaseVersionStatus.processed
            _set_progress(version_id, "done", 100)
        except Exception as e:
            logging.exception("process_version failed for %s", version_id)
            v.status = LeaseVersionStatus.failed
            try:
                _set_progress(version_id, "failed", 100)
            except Exception:
                pass
        s.flush()


