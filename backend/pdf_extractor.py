from langchain_community.document_loaders import UnstructuredPDFLoader, PyPDFLoader, PyMuPDFLoader
from typing import List
from langchain.schema import Document
import re


def normalize_commas_in_numbers(text: str) -> str:
    return re.sub(r"(\d),(\d)", r"\1.\2", text)


def extract_text_unstructured(path: str) -> str:
    try:
        loader = UnstructuredPDFLoader(path, mode="elements")
        docs = loader.load()
        return "\n".join(doc.page_content for doc in docs if getattr(doc, "page_content", None))
    except Exception:
        return ""


def pypdf_page_docs(pdf_path: str) -> List[Document]:
    return PyPDFLoader(pdf_path).load()


def pymupdf_page_docs(pdf_path: str) -> List[Document]:
    return PyMuPDFLoader(pdf_path).load()


