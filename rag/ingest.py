import json
import os
from typing import List, Dict, Any

import requests
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from config.config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# categories we care about
AI_CATEGORIES = {"cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"}


from typing import Iterator


def load_metadata(path: str) -> Iterator[Dict[str, Any]]:
    logger.info(f"Loading metadata from {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def filter_records(record: Dict[str, Any]) -> bool:
    cats = {c.strip() for c in record.get("categories", "").split()}
    return bool(cats & AI_CATEGORIES)


def record_to_doc(record: Dict[str, Any]) -> Dict[str, Any]:
    # combine title/abstract
    text = record.get("title", "") + "\n\n" + record.get("abstract", "")
    metadata = {
        "title": record.get("title"),
        "authors": record.get("authors"),
        "categories": record.get("categories"),
        "published": record.get("update_date") or record.get("published"),
        "arxiv_id": str(record.get("id")),
        "doi": record.get("doi"),
        # more metadata as needed
    }
    return {"text": text, "metadata": metadata}


def ingest(arxiv_path: str = None, persist_directory: str = None, force: bool = False):
    arxiv_path = arxiv_path or settings.arxiv_metadata_path
    persist_directory = persist_directory or settings.vector_store_path

    if not force and os.path.exists(persist_directory) and os.listdir(persist_directory):
        logger.info("Vector store already exists; skipping ingestion.")
        return
    if force and os.path.exists(persist_directory):
        logger.info("Force flag set, clearing existing vector store")
        for fname in os.listdir(persist_directory):
            path = os.path.join(persist_directory, fname)
            try:
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    import shutil

                    shutil.rmtree(path)
            except Exception:
                pass

    logger.info("Starting ingestion pipeline")
    docs = []
    for record in load_metadata(arxiv_path):
        if filter_records(record):
            docs.append(record_to_doc(record))

    logger.info(f"Filtered to {len(docs)} documents")

    texts = [d["text"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = []
    chunk_metas = []
    for text, meta in zip(texts, metadatas):
        for chunk in splitter.split_text(text):
            chunks.append(chunk)
            chunk_metas.append(meta)

    logger.info(f"Created {len(chunks)} chunks")

    embed_model = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embed_model,
        metadatas=chunk_metas,
        persist_directory=persist_directory,
    )
    vectordb.persist()
    logger.info("Ingestion complete and persisted to chroma directory")


def ingest_pdf(pdf_path: str, metadata: dict):
    """Process a downloaded PDF and add its text to the existing vectorstore."""
    from unstructured.partition.pdf import partition_pdf

    logger.info(f"Parsing PDF {pdf_path}")
    elements = partition_pdf(filename=pdf_path)
    full_text = "\n".join([el.get("text", "") for el in elements])
    # split and store
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(full_text)

    embed_model = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    vectordb = Chroma(
        persist_directory=settings.vector_store_path,
        embedding_function=embed_model,
    )
    vectordb.add_texts(chunks, metadatas=[metadata] * len(chunks))
    vectordb.persist()
    logger.info("PDF content ingested into vectorstore")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest arXiv metadata into Chroma vector store")
    parser.add_argument("--path", help="Path to arXiv metadata json file", default=None)
    parser.add_argument("--persist", help="Chroma persist directory", default=None)
    parser.add_argument("--force", help="Force re-ingestion (clear existing store)", action="store_true")
    args = parser.parse_args()
    ingest(arxiv_path=args.path, persist_directory=args.persist, force=args.force)
