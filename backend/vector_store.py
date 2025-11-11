from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Tuple, List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from backend.paths import _doc_dir


def _chunks_path(doc_id: str) -> Path:
    return _doc_dir(doc_id) / "chunks.json"


def save_chunks_json(doc_id: str, docs: List[Document]) -> None:
    data = []
    for d in docs:
        data.append({
            "page_content": d.page_content,
            "metadata": getattr(d, "metadata", {}) or {},
        })
    _chunks_path(doc_id).write_text(json.dumps(data), encoding="utf-8")


def load_chunks_json(doc_id: str) -> List[Document] | None:
    path = _chunks_path(doc_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        docs = [Document(page_content=item.get("page_content", ""), metadata=item.get("metadata", {})) for item in data]
        return docs
    except Exception:
        return None


def get_or_build_vectorstore_for_doc(doc_id: str, docs_builder: Callable[[], List[Document]]) -> Tuple[FAISS, List[Document]]:
    # Try to use cached chunks to avoid rebuilding embeddings repeatedly
    docs = load_chunks_json(doc_id)
    if docs is None:
        docs = docs_builder()
        try:
            save_chunks_json(doc_id, docs)
        except Exception:
            pass

    index_dir = _doc_dir(doc_id)
    vs_path = index_dir / "index.faiss"
    pkl_path = index_dir / "index.pkl"

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if vs_path.exists() and pkl_path.exists():
        try:
            vs = FAISS.load_local(str(index_dir), embeddings=embeddings, allow_dangerous_deserialization=True)
            return vs, docs
        except Exception:
            # Fall back to rebuild
            pass

    vs = FAISS.from_documents(docs, embedding=embeddings)
    vs.save_local(str(index_dir))
    return vs, docs

from typing import List, Tuple
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from backend.paths import _doc_dir
from backend.state import DOC_CACHE
import json


def _chunks_path(doc_id: str):
    return _doc_dir(doc_id) / "chunks.json"


def save_chunks_json(doc_id: str, docs: List[Document]) -> None:
    data = [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]
    _chunks_path(doc_id).write_text(json.dumps(data), encoding="utf-8")


def load_chunks_json(doc_id: str) -> List[Document] | None:
    p = _chunks_path(doc_id)
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
        return [Document(page_content=it.get("page_content", ""), metadata=it.get("metadata", {})) for it in raw]
    except Exception:
        return None


def get_or_build_vectorstore_for_doc(doc_id: str, docs_builder: callable) -> Tuple[FAISS, List[Document]]:
    if doc_id in DOC_CACHE:
        cached = DOC_CACHE[doc_id]
        if "vectorstore" in cached and "docs" in cached:
            return cached["vectorstore"], cached["docs"]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    folder = str(_doc_dir(doc_id))

    try:
        vs = FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
        docs = load_chunks_json(doc_id)
        if docs is None:
            docs = list(vs.docstore._dict.values())  # type: ignore[attr-defined]
        DOC_CACHE[doc_id] = {"vectorstore": vs, "docs": docs}
        return vs, docs
    except Exception:
        pass

    docs = docs_builder()
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(folder)
    save_chunks_json(doc_id, docs)
    DOC_CACHE[doc_id] = {"vectorstore": vs, "docs": docs}
    return vs, docs


