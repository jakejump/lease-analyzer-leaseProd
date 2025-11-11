from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from typing import List, Dict, Any
import re
import json
 

from pathlib import Path
from hashlib import md5
from typing import Optional
from backend.paths import _doc_dir, _project_root, doc_id_from_pdf_path as _DOC_ID_FROM_PATH
from backend.state import LATEST_DOC_ID as _LATEST_DOC_ID, DOC_CACHE as _DOC_CACHE
from backend.vector_store import (
    get_or_build_vectorstore_for_doc as _vs_get,
    save_chunks_json as _save_chunks_json,
    load_chunks_json as _load_chunks_json,
)

# Tracks the most recently uploaded document id so that endpoints can
# default to operating on the latest document without an explicit id.
# Values are provided by backend.state and imported above as _LATEST_DOC_ID and _DOC_CACHE

def _temp_root() -> Path:
    temp_dir = _project_root() / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir

def _compute_doc_id_for_file(file_path: str | Path) -> str:
    global _LATEST_DOC_ID
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        content = f.read()
    # Use MD5 to match existing 32-hex folder names in temp/
    doc_id = md5(content).hexdigest()
    _LATEST_DOC_ID = doc_id
    # Ensure directory exists for downstream operations
    _doc_dir(doc_id)
    return doc_id


def _doc_id_from_pdf_path(pdf_path: str | Path) -> str:
    return _DOC_ID_FROM_PATH(pdf_path)

def _chunks_path(doc_id: str) -> Path:
    return _doc_dir(doc_id) / "chunks.json"

def _clauses_path(doc_id: str) -> Path:
    return _doc_dir(doc_id) / "clauses.json"

def _get_or_build_vectorstore_for_doc(doc_id: str) -> tuple[FAISS, List[Document]]:
    def _builder():
        pdf_path = str(_doc_dir(doc_id) / "lease.pdf")
        return load_lease_docs(pdf_path)
    return _vs_get(doc_id, _builder)



def extract_text_from_pdf(pdf_path: str) -> str:
    print(f"[chain] extract_text_from_pdf start: {pdf_path}")
    # 1) Try lightweight direct PyPDF read first
    def _pypdf_direct(path: str) -> str:
        from pypdf import PdfReader
        reader = PdfReader(path)
        contents: list[str] = []
        for page in reader.pages:
            try:
                text = page.extract_text() or ""
                contents.append(text)
            except Exception as e:
                print("pypdf page extract failed:", e)
        return "\n".join(contents)

    # 2) OCR fallback for scanned PDFs
    def _ocr_with_tesseract(path: str, lang: str = "eng") -> str:
        """High-fidelity OCR for scanned PDFs with preprocessing and 400 DPI.

        - 400 DPI rendering (can be slower but more accurate)
        - OpenCV preprocessing: grayscale, denoise, adaptive threshold, deskew
        - Parallel Tesseract OCR with tuned configs and psm fallback
        """
        try:
            from pdf2image import convert_from_path
            import pytesseract
            from PIL import Image
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import os
            import cv2
            import numpy as np
        except Exception as e:
            print("OCR stack not available:", e)
            return ""

        try:
            images = convert_from_path(path, dpi=400, thread_count=max(1, (os.cpu_count() or 2) // 2))
        except Exception as e:
            print("pdf2image conversion failed (is Poppler installed and on PATH?):", e)
            return ""

        def deskew(image_np: np.ndarray) -> np.ndarray:
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            coords = np.column_stack(np.where(thresh > 0))
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = image_np.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated

        def preprocess(pil_img: Image.Image) -> Image.Image:
            np_img = np.array(pil_img)
            np_img = deskew(np_img)
            gray = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
            denoised = cv2.fastNlMeansDenoising(gray, h=10)
            thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 31, 2)
            return Image.fromarray(thresh)

        def ocr_page(img: Image.Image) -> str:
            try:
                proc = preprocess(img)
                config = "--oem 3 --psm 6"
                text = pytesseract.image_to_string(proc, lang=lang, config=config)
                if len(text.strip()) < 40:
                    # retry with a different page segmentation mode
                    text = pytesseract.image_to_string(proc, lang=lang, config="--oem 3 --psm 4")
                return text
            except Exception as e:  # noqa: BLE001
                print("tesseract OCR failed for a page:", e)
                return ""

        max_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
        text_chunks: list[str] = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(ocr_page, img) for img in images]
            for f in as_completed(futures):
                t = f.result()
                if t:
                    text_chunks.append(t)

        out = "\n".join(text_chunks)
        print(f"[chain] OCR extracted {len(out)} chars")
        return out

    # 3) Unstructured as a last resort (may try to fetch NLTK if missing)
    def _unstructured(path: str) -> str:
        try:
            loader = UnstructuredPDFLoader(path, mode="elements")
            docs = loader.load()
            return "\n".join(doc.page_content for doc in docs if getattr(doc, "page_content", None))
        except Exception as e:
            print("UnstructuredPDFLoader failed:", e)
            return ""

    # Try direct pypdf first
    try:
        direct_text = _pypdf_direct(pdf_path)
        if len(direct_text.strip()) >= 800:
            return direct_text
        print("Direct pypdf produced little text; attempting OCR...")
    except Exception as e:
        print("Direct pypdf failed; attempting OCR...", e)

    # Try OCR
    ocr_text = _ocr_with_tesseract(pdf_path)
    if len(ocr_text.strip()) >= 800:
        return ocr_text

    # If still too short, try PyPDFLoader (page-level parsing)
    try:
        pypdf_loader = PyPDFLoader(pdf_path)
        docs = pypdf_loader.load()
        combined = "\n".join(doc.page_content for doc in docs if getattr(doc, "page_content", None))
        if len(combined.strip()) >= 400:
            return combined
    except Exception as e:
        print("PyPDFLoader also failed:", e)

    # Final fallback: Unstructured
    print("Falling back to Unstructured; this may attempt to use NLTK.")
    un_text = _unstructured(pdf_path)
    if un_text:
        print(f"[chain] Unstructured extracted {len(un_text)} chars")
        return un_text

    raise RuntimeError("All PDF extraction methods failed or produced empty text.")


def get_or_build_clauses_for_doc(doc_id: str) -> list[str]:
    """Return clause-like segments for a document, cached on disk and in-memory.

    Preference order:
    - In-memory `_DOC_CACHE[doc_id]['clauses']`
    - Sidecar file `<doc_dir>/clauses.json`
    - Compute from extracted text once and persist
    """
    # In-memory
    if doc_id in _DOC_CACHE and isinstance(_DOC_CACHE.get(doc_id), dict):
        cached = _DOC_CACHE[doc_id]
        if "clauses" in cached and isinstance(cached["clauses"], list):
            return cached["clauses"]  # type: ignore[return-value]

    path = _clauses_path(doc_id)
    # Disk cache
    try:
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                # Cache in-memory for the process lifetime
                _DOC_CACHE.setdefault(doc_id, {})["clauses"] = data
                return data
    except Exception:
        pass

    # Compute and persist
    pdf_path = str(_doc_dir(doc_id) / "lease.pdf")
    text = extract_text_from_pdf(pdf_path)
    clauses = split_into_paragraphs_or_clauses(text)
    try:
        path.write_text(json.dumps(clauses), encoding="utf-8")
    except Exception:
        pass
    _DOC_CACHE.setdefault(doc_id, {})["clauses"] = clauses
    return clauses

def split_into_paragraphs_or_clauses(text: str) -> List[str]:
    # Normalize common OCR issues first (e.g., comma used as decimal separator between digits)
    norm = re.sub(r"(\d),(\d)", r"\1.\2", text)
    # Normalize weird dashes
    norm = norm.replace("–", "-")

    # Helper to decide if a header token is a real clause header or a cross-reference
    def is_real_header(full_text: str, m: re.Match) -> bool:
        start = m.start(2) if m.lastindex and m.lastindex >= 2 else m.start()
        # Check preceding context for boundary like start, newline, or period + space
        pre = full_text[max(0, start - 3):start]
        boundary_ok = bool(re.search(r"(^|\n|[\.!?]\s)$", pre))
        # If preceded by 'Section ' but not at start of line, likely a reference
        pre_window = full_text[max(0, start - 12):start]
        preceded_section = bool(re.search(r"Section\s*$", pre_window, re.IGNORECASE))
        at_line_start = (start == 0) or full_text[start - 1] == "\n"
        if preceded_section and not at_line_start:
            return False
        # Following context: if immediately followed by subsection like (b) within a reference phrase, likely not header
        post = full_text[m.end(): m.end() + 40]
        ref_words = re.compile(r"\b(below|above|pursuant|provided|as defined|per|see)\b", re.IGNORECASE)
        if ref_words.search(full_text[max(0, start - 40): m.end() + 40]):
            # Allow line-start real headers despite these words
            if not at_line_start and not boundary_ok:
                return False
        # Heuristic: title length should be reasonable and should not be empty if punctuation suggests title
        title = (m.group(3) if m.lastindex and m.lastindex >= 3 else m.group(2)) or ""
        if len(title) > 100:
            return False
        return boundary_ok or at_line_start

    # Regex to find candidate headers: boundary + optional label + number + optional punctuation + Title starting with a letter
    # Disallow immediate subsection markers like "(b)" after the number to avoid cross-references like "24.05(b)"
    header_regex = re.compile(
        r"(?:(?:^|\n|[\.!?]\s))"                 # safe boundary
        r"(?:Section|Clause|Article)?\s*"           # optional label
        r"(\d{1,2}(?:\.\d{1,2})?)\s*"           # clause number
        r"(?:[:\-\.]\s+)?"                        # optional punctuation then space
        r"(?!\()"                                   # do not allow immediate '(' (subsection refs)
        r"([A-Z][^\n]{0,80})?",                    # optional title starting with capital
        re.IGNORECASE
    )

    # Scan for headers and build clause slices
    candidates = []
    for m in header_regex.finditer(norm):
        candidates.append(m)

    headers = []
    for m in candidates:
        if is_real_header(norm, m):
            headers.append((m.start(1), m.group(1)))

    headers.sort(key=lambda x: x[0])

    clauses: List[str] = []
    if headers:
        # Merge same-number headers if noisy duplicates (e.g., line breaks or repeated number)
        merged = []
        for pos, num in headers:
            if merged and merged[-1][1] == num and pos - merged[-1][0] < 40:
                # Skip duplicate header very near the previous
                continue
            merged.append((pos, num))

        for i, (pos, num) in enumerate(merged):
            end = merged[i + 1][0] if i + 1 < len(merged) else len(norm)
            # Extend start to the nearest line start
            start_line = norm.rfind("\n", 0, pos) + 1
            chunk = norm[start_line:end].strip()
            if chunk:
                clauses.append(chunk)

    # Fallback if results look unreasonable
    total_chars = len(norm)
    if len(clauses) < 3 or len(clauses) > max(150, total_chars // 150):
        print("⚠️ Smart clause split fallback to paragraphs.")
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", norm) if len(p.strip()) > 40]
        return paragraphs

    return clauses


def _build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " "]
    )

def _layout_titles_path(doc_id: str) -> Path:
    return _doc_dir(doc_id) / "layout_titles.json"

def _get_or_build_layout_titles(doc_id: str, pdf_path: str) -> list[dict]:
    """Extract page-level Title blocks using Unstructured's hi_res pipeline.
    Falls back to empty list if the model or deps are unavailable.
    """
    path = _layout_titles_path(doc_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        from unstructured.partition.pdf import partition_pdf
        elements = partition_pdf(filename=pdf_path, strategy="hi_res", infer_table_structure=False)
        titles: list[dict] = []
        for el in elements:
            name = getattr(el, "category", None) or el.__class__.__name__
            if str(name).lower() == "title":
                meta = getattr(el, "metadata", None)
                page_no = getattr(meta, "page_number", None) if meta else None
                # Coordinates may be None depending on pipeline
                coords = getattr(meta, "coordinates", None)
                titles.append({
                    "page": int(page_no) if page_no is not None else None,
                    "text": getattr(el, "text", "") or "",
                    "coordinates": str(coords) if coords is not None else None,
                })
        # Save sidecar
        try:
            path.write_text(json.dumps(titles), encoding="utf-8")
        except Exception:
            pass
        return titles
    except Exception as e:
        print("Layout title extraction unavailable:", e)
        return []

def _normalize_line(line: str) -> str:
    # Collapse whitespace and remove stray artifacts for comparison
    return " ".join(line.strip().split())

def _find_common_header_footer_lines(page_texts: list[str], top_k_lines: int = 3, bottom_k_lines: int = 3, freq_threshold: float = 0.6) -> tuple[set[str], set[str]]:
    from collections import Counter
    num_pages = max(1, len(page_texts))
    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()

    for text in page_texts:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        top = lines[:top_k_lines]
        bot = lines[-bottom_k_lines:] if bottom_k_lines > 0 else []
        header_counter.update(_normalize_line(ln) for ln in top)
        footer_counter.update(_normalize_line(ln) for ln in bot)

    # Always include explicit page number patterns
    explicit_page_patterns = (
        r"^page\s*\d+(\s*/\s*\d+|\s+of\s+\d+)?$",
        r"^\d+\s*/\s*\d+$",
        r"^\-?\s*\d+\s*\-?$",
    )

    header_set: set[str] = set()
    footer_set: set[str] = set()

    for line, cnt in header_counter.items():
        if cnt / num_pages >= freq_threshold and 5 <= len(line) <= 120:
            header_set.add(line)
    for line, cnt in footer_counter.items():
        if cnt / num_pages >= freq_threshold and 3 <= len(line) <= 120:
            footer_set.add(line)

    # Add explicit page-number matches
    import re as _re
    for text in page_texts:
        for ln in text.splitlines():
            norm = _normalize_line(ln).lower()
            if any(_re.match(pat, norm, _re.IGNORECASE) for pat in explicit_page_patterns):
                footer_set.add(norm)

    return header_set, footer_set

def _fix_hyphenation(lines: list[str]) -> list[str]:
    fixed: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.rstrip().endswith("-") and i + 1 < len(lines):
            # Merge hyphenated word across newline
            merged = line.rstrip()[:-1] + lines[i + 1].lstrip()
            fixed.append(merged)
            i += 2
        else:
            fixed.append(line)
            i += 1
    return fixed

def _clean_page_text(text: str, header_set: set[str], footer_set: set[str]) -> str:
    import re as _re
    lines = text.splitlines()
    lines = [ln for ln in lines if ln is not None]

    # Drop leading header lines if match set
    start = 0
    while start < len(lines) and _normalize_line(lines[start]) in header_set:
        start += 1
    # Drop trailing footer lines if match set
    end = len(lines)
    while end > start and _normalize_line(lines[end - 1]) in footer_set:
        end -= 1
    sliced = lines[start:end]

    # Remove explicit page number lines anywhere
    page_num_patterns = (
        r"^\s*page\s*\d+(\s*/\s*\d+|\s+of\s+\d+)?\s*$",
        r"^\s*\d+\s*/\s*\d+\s*$",
        r"^\s*\-?\s*\d+\s*\-?\s*$",
    )
    def is_page_num(ln: str) -> bool:
        norm = _normalize_line(ln).lower()
        return any(_re.match(pat, norm, _re.IGNORECASE) for pat in page_num_patterns)

    sliced = [ln for ln in sliced if not is_page_num(ln)]

    # Fix hyphenation across lines
    sliced = _fix_hyphenation(sliced)

    # Collapse multiple spaces inside a line for cleanliness
    cleaned = [" ".join(ln.split()) for ln in sliced]
    return "\n".join(cleaned).strip()

def load_lease_docs(pdf_path: str) -> List[Document]:
    print(f"[chain] load_lease_docs for: {pdf_path}")
    # Prefer PyMuPDF for higher-fidelity page extraction
    try:
        page_docs = PyMuPDFLoader(pdf_path).load()
        # Title detection via ML layout model to refine headers
        doc_id = _doc_id_from_pdf_path(pdf_path)
        layout_titles = _get_or_build_layout_titles(doc_id, pdf_path)
        titles_by_page = {}
        for t in layout_titles:
            p = t.get("page")
            if p is not None:
                titles_by_page.setdefault(int(p), []).append(t.get("text", "").strip())
        # Remove headers/footers/page numbers using cross-page frequency
        page_texts = [d.page_content for d in page_docs]
        header_set, footer_set = _find_common_header_footer_lines(page_texts)
        cleaned_pages = [_clean_page_text(t, header_set, footer_set) for t in page_texts]
        splitter = _build_text_splitter()
        split_docs: List[Document] = []
        for d, cleaned in zip(page_docs, cleaned_pages):
            parts = splitter.split_text(cleaned)
            for idx, part in enumerate(parts):
                meta = dict(getattr(d, "metadata", {}))
                meta.update({"page": meta.get("page", meta.get("page_number")), "chunk": idx})
                # Attach ML-detected titles for the page if available (helps downstream heuristics)
                page_num = meta.get("page")
                if page_num in titles_by_page:
                    meta["layout_titles"] = titles_by_page[page_num]
                split_docs.append(Document(page_content=part, metadata=meta))
        if split_docs:
            print(f"[chain] PyMuPDF produced {len(split_docs)} chunks")
            return split_docs
    except Exception as e:
        print("PyMuPDF page-aware load failed; trying PyPDFLoader:", e)
        try:
            pypdf_loader = PyPDFLoader(pdf_path)
            page_docs = pypdf_loader.load()
            doc_id = _doc_id_from_pdf_path(pdf_path)
            layout_titles = _get_or_build_layout_titles(doc_id, pdf_path)
            titles_by_page = {}
            for t in layout_titles:
                p = t.get("page")
                if p is not None:
                    titles_by_page.setdefault(int(p), []).append(t.get("text", "").strip())
            page_texts = [d.page_content for d in page_docs]
            header_set, footer_set = _find_common_header_footer_lines(page_texts)
            cleaned_pages = [_clean_page_text(t, header_set, footer_set) for t in page_texts]
            splitter = _build_text_splitter()
            split_docs: List[Document] = []
            for d, cleaned in zip(page_docs, cleaned_pages):
                parts = splitter.split_text(cleaned)
                for idx, part in enumerate(parts):
                    meta = dict(getattr(d, "metadata", {}))
                    meta.update({"page": meta.get("page", meta.get("page_number")), "chunk": idx})
                    page_num = meta.get("page")
                    if page_num in titles_by_page:
                        meta["layout_titles"] = titles_by_page[page_num]
                    split_docs.append(Document(page_content=part, metadata=meta))
            if split_docs:
                print(f"[chain] PyPDFLoader produced {len(split_docs)} chunks")
                return split_docs
        except Exception as e2:
            print("PyPDFLoader page-aware load failed; falling back to raw text:", e2)

    text = extract_text_from_pdf(pdf_path)
    paragraphs = split_into_paragraphs_or_clauses(text)
    splitter = _build_text_splitter()
    split_docs = []
    for i, para in enumerate(paragraphs):
        for j, part in enumerate(splitter.split_text(para)):
            split_docs.append(Document(page_content=part, metadata={"para_index": i, "chunk": j}))
    print(f"[chain] Fallback paragraphs produced {len(split_docs)} chunks")
    return split_docs


def _get_retriever(doc_id: str):
    print(f"[chain] get retriever for {doc_id}")
    if doc_id in _DOC_CACHE and "retriever" in _DOC_CACHE[doc_id]:
        return _DOC_CACHE[doc_id]["retriever"]
    vs, docs = _get_or_build_vectorstore_for_doc(doc_id)
    emb_retriever = vs.as_retriever(search_type="mmr", search_kwargs={"k": 12, "fetch_k": 40})
    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 12
    ensemble = EnsembleRetriever(retrievers=[emb_retriever, bm25], weights=[0.65, 0.35])
    filter = EmbeddingsFilter(embeddings=OpenAIEmbeddings(model="text-embedding-3-small"), k=8, similarity_threshold=0.35)
    retriever = ContextualCompressionRetriever(base_compressor=filter, base_retriever=ensemble)
    _DOC_CACHE[doc_id]["retriever"] = retriever
    return retriever

def run_rag_pipeline(pdf_path: str, question: str):
    print(f"[chain] RAG question: {question[:60]}... for {pdf_path}")
    doc_id = _doc_id_from_pdf_path(pdf_path)
    retriever = _get_retriever(doc_id)

    SYSTEM = """
    You are a contract analyst reviewing a commercial lease agreement. Based on the provided context,
    answer the user's question. Return your answer in plain English.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM.strip()),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain.invoke(question)


def evaluate_general_risks(pdf_path: str):
    print(f"[chain] risk evaluation start for {pdf_path}")
    doc_id = _doc_id_from_pdf_path(pdf_path)
    retriever = _get_retriever(doc_id)

    SYSTEM = """
    You are a risk analyst evaluating a lease document. You are an analyst for a firm that is purchasing or puttng together commercial real-estate deals, so the risk should be from the perspective of the lessor. Based on the following context, score the lease across the following general risk categories from 1 (high risk) to 10 (low risk) and explain each score:

    - Cash Flow Adjustments:
        Lease Structure (Who Pays What?)
            Gross Lease: Landlord pays most or all property expenses (riskier for landlord).
            Net Lease: Tenant pays some or all operating expenses.
                Single Net: Tenant pays property taxes.
                Double Net: Taxes + insurance.
                Triple Net (NNN): Taxes + insurance + maintenance.
            Risk: Gross leases shift cost risk to you; NNN leases shift it to tenants (safer).
        Capital Expenditure (CapEx) Obligations
            Who is responsible for big repairs like roof, HVAC, structure?
            Tenant Improvement (TI) Allowances: Did the landlord promise money for upgrades?
            Risk: You could be on the hook for big unexpected costs.
        Co-tenancy clauses (in retail leases: tenant can pay less or leave if anchor tenants leave).
        Free rent periods or concessions built into the lease.
        
    - Future Cash Flow:
        Renewal options (does tenant have options to stay longer, and at what rates?)
        Risk: Short-term leases = turnover risk, renewal uncertainty.
        Scheduled rent escalations (fixed bumps? CPI-linked increases?)
        Risk: If market rents are falling, or if you're locked into below-market leases, it hurts cash flow and future value.
        Outline exposure to inflation and changes in interest rates

    - Inflation/Interest Rate Exposure:
	    Macro implications of the lease contract. 
        If there is high inflation, how do rent escalations hold up, how do renewal options affect the value of the lease contract, how does the specific working of cash flow adjustments (like TI and lease structure) hold up. 
        Is it beneficial for the lessor or is it a negative.
        Apply the same logic to changes in global/nationwide interest rates.

    - Use and Exclusivity Clauses:
        Permitted use: What exactly can the tenant do on the property?
        Exclusive use rights: Do they have rights that could restrict future tenants?
        Risk: Restrictions can limit re-leasing flexibility.
        Sublease or assignment rights (can tenant sublease easily? Risk of poor subtenants.)
        SNDA agreements (Subordination, Non-Disturbance, and Attornment).

    - Default and Termination Clauses:
        Early termination rights (can the tenant break the lease? On what terms?)
        Default provisions (what triggers an eviction? Cure periods?)
        Risk: Easy outs or weak default clauses mean unstable cash flow.

    - Collateral and Insurance:
        Security Deposits, Guarantees, and Collateral
            Security deposit size and conditions.
            Personal or corporate guarantees (especially important for smaller tenants).
            Letters of credit or other forms of collateral.
            Risk: More security = better recovery in a default.
        Insurance Requirements
            Tenant’s insurance obligations (and evidence they maintain them).
            Landlord's insurance coverage (especially for common areas).
            Risk: Poor insurance setups = risk of uncovered losses.

    Please return your result strictly in the following JSON format, and nothing else:

    {{
      "cash_flow_adjustments": {{"score": int, "explanation": str}},
      "future_cash_flow": {{"score": int, "explanation": str}},
      "inflation/interest_rate_exposure": {{"score": int, "explanation": str}},
      "use_and_exclusivity_clauses": {{"score": int, "explanation": str}},
      "default_and_termination_clauses": {{"score": int, "explanation": str}},
      "collateral_and_insurance": {{"score": int, "explanation": str}}
    }}

    Do not include any commentary or markdown — only valid JSON.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM.strip()),
        ("human", "Context:\n{context}\n\nEvaluate the lease risks.")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    raw_output = chain.invoke("Evaluate the lease risks.")
    print(f"[chain] risk llm responded {len(str(raw_output))} chars")

    try:
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned.removeprefix("```json").removesuffix("```")
        result = json.loads(cleaned)
        assert isinstance(result, dict)
        print("risks", result)
        return result
    except Exception as e:
        print("[chain] invalid JSON in risk output; returning fallback")
        return {
            "termination_risk": {"score": None, "explanation": "Could not parse response."},
            "financial_exposure": {"score": None, "explanation": "Could not parse response."},
            "legal_ambiguity": {"score": None, "explanation": "Could not parse response."},
            "operational_complexity": {"score": None, "explanation": "Could not parse response."},
            "assignment_subletting_risk": {"score": None, "explanation": "Could not parse response."},
            "renewal_escalation_risk": {"score": None, "explanation": "Could not parse response."}
        }
        
def detect_abnormalities(pdf_path: str):
    print(f"[chain] abnormalities start for {pdf_path}")
    doc_id = _doc_id_from_pdf_path(pdf_path)
    retriever = _get_retriever(doc_id)

    SYSTEM = """
    You are an expert lease reviewer. Identify any unusual, uncommon, or non-standard clauses in this lease.
    For each item, assess whether it is beneficial to the landlord/lessor or harmful to the landlord/lessor.
    Only return items that deviate from common practice. If everything is normal, return an empty list.

    Return strictly JSON as a list of objects with fields:
    [
      {{"text": str, "impact": "beneficial" | "harmful" | "neutral"}}
    ]
    Do not include any markdown or commentary outside JSON.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM.strip()),
        ("human", "Context:\n{context}\n\nIdentify abnormalities with impact for landlord.")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    result = chain.invoke("Identify abnormalities with impact for landlord.")
    print(f"[chain] abnormalities llm responded {len(str(result))} chars")
    def _robust_parse(text: str):
        cleaned = text.strip()
        # Strip common code fences
        if cleaned.startswith("```json") and cleaned.endswith("```"):
            cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
        elif cleaned.startswith("```") and cleaned.endswith("```"):
            cleaned = cleaned.removeprefix("```").removesuffix("```").strip()
        try:
            return json.loads(cleaned)
        except Exception:
            # Try to extract JSON array or object region
            start = cleaned.find("[")
            end = cleaned.rfind("]")
            if start != -1 and end != -1 and end > start:
                candidate = cleaned[start:end+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    pass
            start = cleaned.find("{")
            end = cleaned.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = cleaned[start:end+1]
                try:
                    return json.loads(candidate)
                except Exception:
                    pass
            raise
    try:
        parsed = _robust_parse(result)
        if isinstance(parsed, list):
            normalized = []
            for item in parsed:
                if isinstance(item, dict) and "text" in item:
                    impact = item.get("impact")
                    if impact not in ("beneficial", "harmful", "neutral"):
                        impact = "harmful"
                    normalized.append({"text": item.get("text", ""), "impact": impact})
                elif isinstance(item, str):
                    normalized.append({"text": item, "impact": "harmful"})
            return normalized
        if isinstance(parsed, dict) and "text" in parsed:
            impact = parsed.get("impact")
            if impact not in ("beneficial", "harmful", "neutral"):
                impact = "harmful"
            return [{"text": parsed.get("text", ""), "impact": impact}]
        return [{"text": "No abnormalities found.", "impact": "beneficial"}]
    except Exception:
        return [{"text": "Could not parse LLM response.", "impact": "harmful"}]


def get_clauses_for_topic(pdf_path: str, topic: str):
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    doc_id = _doc_id_from_pdf_path(pdf_path)
    vectorstore, docs = _get_or_build_vectorstore_for_doc(doc_id)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    topic_embedding = embeddings.embed_query(topic)
    stored_embeddings = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)
    stored_docs = list(vectorstore.docstore._dict.values())

    similarities = cosine_similarity([topic_embedding], stored_embeddings)[0]
    doc_scores = list(zip(stored_docs, similarities))
    # Sort all candidates by similarity descending
    doc_scores.sort(key=lambda x: x[1], reverse=True)

    # Take a broader candidate set for LLM re-ranking
    CANDIDATE_K = 12
    candidates = doc_scores[:CANDIDATE_K]

    def _format_clause(raw_text: str, meta: dict | None = None) -> str:
        import re as _re

        text = raw_text.strip()
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        header_line = lines[0] if lines else ""

        # Try to extract clause number and optional title from the first line
        # Patterns like: "Section 5.2 Title", "5.2 Title", "Clause 7 - Title", "Article 12: Title"
        header_match = _re.match(r"^(?:Section|Clause|Article)?\s*(\d{1,2}(?:\.\d{1,2})?)\s*[-:.)]?\s*(.*)$", header_line, _re.IGNORECASE)
        # Heuristics: treat as a real header only if the line is short and not a cross-reference
        forbidden_words = r"\b(below|above|pursuant|provided|as defined|per|see)\b"
        looks_short = len(header_line) <= 80
        not_cross_ref = not _re.search(forbidden_words, header_line, _re.IGNORECASE)

        is_real_header = bool(header_match) and looks_short and not_cross_ref

        if is_real_header:
            clause_no = header_match.group(1)
            clause_title = header_match.group(2).strip()
        else:
            # Fallback: try to find a leading number anywhere
            any_num = _re.search(r"(\d{1,2}(?:\.\d{1,2})?)", header_line)
            clause_no = any_num.group(1) if any_num else ""
            clause_title = header_line if is_real_header else ""

        # Use ML layout titles (if present in metadata) to boost heading detection deterministically
        layout_titles = []
        if isinstance(meta, dict):
            layout_titles = [t.strip() for t in meta.get("layout_titles", []) if isinstance(t, str)]
        def _norm(s: str) -> str:
            return _re.sub(r"\s+", " ", s or "").strip().lower()
        norm_header = _norm(header_line)
        if not is_real_header and layout_titles:
            for t in layout_titles:
                nt = _norm(t)
                starts_with_title = _norm(text).startswith(nt)
                header_eq_title = norm_header.startswith(nt) or nt.startswith(norm_header)
                if len(nt) > 4 and (starts_with_title or header_eq_title):
                    clause_title = t.strip()
                    num_match = _re.match(r"^(?:Section|Clause|Article)?\s*(\d{1,2}(?:\.\d{1,2})?)\b", text, _re.IGNORECASE)
                    if num_match:
                        clause_no = num_match.group(1)
                    is_real_header = True
                    break

        # Body paragraphs: split by blank lines; ensure each paragraph is one line
        body_text = text
        # Remove the header line from body only if we confidently detected a header
        if is_real_header and header_line and len(lines) > 1:
            body_text = "\n".join(lines[1:])
        paragraphs = [p.strip() for p in _re.split(r"\n{2,}", body_text) if p.strip()]
        # Normalize each paragraph to a single line (replace internal newlines with spaces)
        normalized_paragraphs = [" ".join(p.split()) for p in paragraphs] or [" ".join(body_text.split())]

        # Compose heading and paragraphs
        heading_parts = []
        if clause_no:
            heading_parts.append(clause_no)
        if clause_title:
            heading_parts.append(clause_title)
        heading = " ".join(heading_parts).strip()
        # If ML layout titles exist for the page, and the clause title is empty, try to use the first title
        # Note: this function formats text only; we do not have metadata here, so this fallback is limited.
        if not heading:
            # If we couldn't confidently detect a header, avoid misleading header text
            formatted = "\n".join(["  " + para for para in normalized_paragraphs])
        else:
            formatted = heading + ":\n" + "\n".join(["  " + para for para in normalized_paragraphs])
        return formatted.strip()

    # Further split within a single chunk if multiple headers exist inline (e.g., "23.02 ... 23.03 ...")
    def _split_inline_headers(text: str) -> list[str]:
        import re as _re
        # Normalize 23,03 -> 23.03
        t = _re.sub(r"(\d),(\d)", r"\\1.\\2", text)
        # Boundary: start, newline, or punctuation+space; avoid subsection like (b)
        pattern = _re.compile(
            r"(?:(?<=^)|(?<=\n)|(?<=[\.!?]\s))(?=(?:Section|Clause|Article)?\s*\d{1,2}(?:\.\d{1,2})?\s*(?:[:\-\.]\s+)?(?!\())",
            _re.IGNORECASE,
        )
        parts: list[str] = []
        last = 0
        for m in pattern.finditer(t):
            idx = m.start()
            if idx > last:
                seg = t[last:idx].strip()
                if seg:
                    parts.append(seg)
            last = idx
        tail = t[last:].strip()
        if tail:
            parts.append(tail)
        return parts or [text]

    # Build formatted clause candidates from the top chunks
    candidate_texts: list[str] = []
    for doc, _score in candidates:
        meta = getattr(doc, "metadata", {})
        for segment in _split_inline_headers(doc.page_content):
            if not segment or len(segment.strip()) < 40:
                continue
            candidate_texts.append(_format_clause(segment, meta))
            if len(candidate_texts) >= CANDIDATE_K:
                break
        if len(candidate_texts) >= CANDIDATE_K:
            break

    # Fallback if nothing reasonable
    if not candidate_texts:
        return []

    # LLM re-rank: ask to pick the best 4 in order (most relevant first)
    TOP_K = 4
    try:
        llm = ChatOpenAI(model="gpt-4o", temperature=0)
        from textwrap import shorten
        lines = []
        for i, txt in enumerate(candidate_texts):
            # Keep messages compact; the model still sees full content below
            _san = txt.replace("\n", " ")
            _sh = shorten(_san, width=240, placeholder='…')
            lines.append(f"{i}: {_sh}")
        SYSTEM = (
            "You are ranking lease clauses for relevance to a user's topic. "
            "Return ONLY JSON: an array of 4 integer indices (0-based) of the most relevant clauses, "
            "ordered from most to least relevant. No commentary."
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM),
            ("human", (
                "Topic: {topic}\n\n"
                "Clauses (index: text):\n{candidates}\n\n"
                "Return JSON array of 4 indices only."
            )),
        ])
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke({"topic": topic, "candidates": "\n".join(lines)})
        def _parse_indices(s: str) -> list[int]:
            t = s.strip()
            if t.startswith("```json") and t.endswith("```"):
                t = t.removeprefix("```json").removesuffix("```").strip()
            import json as _json
            try:
                arr = _json.loads(t)
            except Exception:
                # try bracket extraction
                a, b = t.find("["), t.rfind("]")
                if a != -1 and b != -1 and b > a:
                    arr = _json.loads(t[a:b+1])
                else:
                    raise
            out = []
            for v in arr:
                try:
                    out.append(int(v))
                except Exception:
                    pass
            return out[:TOP_K]
        idxs = _parse_indices(raw)
        if len(idxs) < TOP_K:
            # fill with highest-similarity order as fallback
            idxs = (idxs + list(range(len(candidate_texts))))[:TOP_K]
        ranked = [candidate_texts[i] for i in idxs if 0 <= i < len(candidate_texts)]
        if ranked:
            return ranked[:TOP_K]
    except Exception as e:
        print("[clauses] rerank failed:", e)

    # Hard fallback: use similarity order top-k
    return candidate_texts[:TOP_K]
