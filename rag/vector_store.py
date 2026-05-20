"""
Vector store module — FAISS-backed persistent index.
Handles building, saving, and loading the document index.
Uses line/paragraph based chunking (no NLTK dependency).
"""
import os
import re
import json
import hashlib
import faiss
import numpy as np
from pathlib import Path

from rag.embeddings import embed_texts, get_vectorizer, VECTORIZER_PATH
from rag.file_manager import UPLOADS_DIR
from rag.file_parser import parse_file, is_supported_file

INDEX_PATH = os.path.join(os.path.dirname(__file__), "../database/clasico.index")
CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "../database/chunks.json")
INDEX_VERSION = 1


def _chunk_text(text, source):
    """
    Split text into meaningful chunks.
    Strategy: split on blank lines or bullet-point lines, preserving semantic units
    with overlap to keep related facts together.
    """
    chunks = []
    sources = []

    def add_chunk(chunk_text):
        chunk_text = chunk_text.strip()
        if len(chunk_text) >= 20:
            chunks.append(chunk_text)
            sources.append(source)

    paragraphs = re.split(r'\n\s*\n', text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        lines = [l.strip() for l in para.split('\n') if l.strip()]
        if len(lines) == 1:
            # If the paragraph is one long line, split into sentence chunks.
            sentences = re.split(r'(?<=[.!?])\s+', lines[0])
        else:
            # Normalize bullet markers to text lines.
            sentences = [l[1:].strip() if l.startswith('-') else l for l in lines]

        for i, sentence in enumerate(sentences):
            add_chunk(sentence)
            if i + 1 < len(sentences):
                add_chunk(f"{sentence} {sentences[i + 1]}")
            if i + 2 < len(sentences):
                add_chunk(f"{sentence} {sentences[i + 1]} {sentences[i + 2]}")

    return chunks, sources


def _load_documents_from_folder(folder: str, source_prefix: str = ""):
    """Load supported files from a folder and split into chunks with source metadata."""
    chunks = []
    sources = []

    for fname in sorted(os.listdir(folder)):
        path = os.path.join(folder, fname)
        if os.path.isdir(path) or fname.startswith('.'):
            continue

        if not is_supported_file(fname):
            continue

        try:
            text = parse_file(path)
        except Exception as e:
            print(f"Skipped unsupported or unreadable file {fname}: {e}")
            continue

        source_name = f"{source_prefix}{Path(fname).stem}"
        file_chunks, file_sources = _chunk_text(text, source_name)
        chunks.extend(file_chunks)
        sources.extend(file_sources)

    return chunks, sources


def load_documents(data_folder: str):
    """Load supported documents from the main data folder and uploaded files."""
    chunks, sources = _load_documents_from_folder(data_folder)

    if os.path.exists(UPLOADS_DIR):
        upload_chunks, upload_sources = _load_documents_from_folder(UPLOADS_DIR, source_prefix="upload:")
        chunks.extend(upload_chunks)
        sources.extend(upload_sources)

    return chunks, sources


def _hash_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def _load_source_metadata(data_folder: str) -> dict:
    metadata = {}

    for folder, prefix in [(data_folder, ""), (UPLOADS_DIR, "upload:")]:
        if not os.path.exists(folder):
            continue

        for fname in sorted(os.listdir(folder)):
            path = os.path.join(folder, fname)
            if os.path.isdir(path) or fname.startswith('.'):
                continue
            if not is_supported_file(fname):
                continue

            metadata[f"{prefix}{fname}"] = {
                "mtime": os.path.getmtime(path),
                "hash": _hash_file(path),
            }

    return metadata


def _cache_is_stale(data_folder: str) -> bool:
    if not os.path.exists(INDEX_PATH) or not os.path.exists(CHUNKS_PATH):
        return True

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    metadata = data.get("metadata") or {}
    if metadata.get("index_version") != INDEX_VERSION:
        return True

    current = _load_source_metadata(data_folder)
    if metadata.get("source_hashes") != {k: v["hash"] for k, v in current.items()}:
        return True

    return False


def build_index(data_folder: str, force_rebuild: bool = False):
    """
    Build (or load from cache) the FAISS index and chunks list.
    Returns (index, chunks, sources).
    """
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

    if not force_rebuild and not _cache_is_stale(data_folder):
        print("Loading cached FAISS index...")
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        vectorizer = get_vectorizer()
        if vectorizer is None or vectorizer.transform(["test"]).shape[1] != index.d:
            print("Cached TF-IDF vectorizer missing or dimension-mismatched. Rebuilding FAISS index and vectorizer.")
        else:
            return index, data["chunks"], data["sources"]

    print("Building FAISS index from documents...")
    chunks, sources = load_documents(data_folder)
    source_meta = _load_source_metadata(data_folder)
    print(f"Loaded {len(chunks)} chunks from {len(set(sources))} files")

    if not chunks:
        raise RuntimeError("No documents found to build FAISS index.")

    embeddings = embed_texts(chunks)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "chunks": chunks,
            "sources": sources,
            "metadata": {
                "index_version": INDEX_VERSION,
                "source_hashes": {k: v["hash"] for k, v in source_meta.items()},
                "build_time": os.path.getmtime(INDEX_PATH),
            },
        }, f, ensure_ascii=False)

    print(f"Index built: {index.ntotal} vectors, dim={dim}")
    return index, chunks, sources


def ingest_new_chunks(
    index,
    chunks: list,
    sources: list,
    new_chunks: list,
    new_sources: list
) -> tuple:
    """
    Add new chunks to an existing FAISS index.
    
    Args:
        index: Existing FAISS index
        chunks: Existing chunks list
        sources: Existing sources list
        new_chunks: New chunks to add
        new_sources: Sources for new chunks
    
    Returns:
        Tuple of (updated_index, updated_chunks, updated_sources)
    """
    from rag.embeddings import embed_texts
    
    if not new_chunks:
        return index, chunks, sources
    
    print(f"Ingesting {len(new_chunks)} new chunks...")
    
    new_embeddings = embed_texts(new_chunks)
    
    if new_embeddings.shape[1] != index.d:
        raise RuntimeError(
            f"Embedding dimension mismatch: FAISS index expects dim={index.d}, "
            f"but new embeddings have dim={new_embeddings.shape[1]}. "
            "Rebuild the index or restore the matching TF-IDF vectorizer."
        )

    index.add(new_embeddings)
    updated_chunks = chunks + new_chunks
    updated_sources = sources + new_sources
    
    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "chunks": updated_chunks,
            "sources": updated_sources,
            "metadata": {
                "index_version": INDEX_VERSION,
                "last_ingestion": os.path.getmtime(INDEX_PATH),
            },
        }, f, ensure_ascii=False)
    
    print(f"Index updated: {index.ntotal} vectors total")
    return index, updated_chunks, updated_sources
