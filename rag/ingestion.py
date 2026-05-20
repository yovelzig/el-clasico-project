"""
Ingestion module — Process and embed uploaded documents.
Handles chunking with overlap and embedding integration.
"""
import logging
import re
from typing import List, Tuple

logger = logging.getLogger(__name__)


def chunk_text_with_overlap(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Full text to chunk
        chunk_size: Target size in characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Find a good break point near chunk_size
        end = min(start + chunk_size, len(text))
        
        # If we're not at the end, try to break at a sentence boundary
        if end < len(text):
            # Look for sentence endings
            search_text = text[start:end + 200]
            matches = list(re.finditer(r'[.!?]\s+', search_text))
            
            if matches:
                # Break at the last sentence boundary found
                last_match = matches[-1]
                end = start + last_match.end()
            else:
                # No sentence found, look for period/space combo
                period_idx = text.rfind('. ', start, end)
                if period_idx > start:
                    end = period_idx + 2
                else:
                    # Last resort: find last space
                    space_idx = text.rfind(' ', start, end)
                    if space_idx > start:
                        end = space_idx + 1
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start for next chunk with overlap
        start = end - overlap
        if start <= 0:
            break
    
    return chunks


def ingest_document(
    text: str,
    source_name: str,
    chunk_size: int = 500,
    overlap: int = 100
) -> Tuple[List[str], List[str]]:
    """
    Ingest a document by chunking and preparing for embedding.
    
    Args:
        text: Full document text
        source_name: Source identifier (filename or doc name)
        chunk_size: Target chunk size in characters
        overlap: Overlap between chunks
    
    Returns:
        Tuple of (chunks, sources) where sources is the source_name repeated
    """
    chunks = chunk_text_with_overlap(text, chunk_size, overlap)
    sources = [source_name] * len(chunks)
    
    logger.info(f"Ingested '{source_name}': {len(chunks)} chunks")
    return chunks, sources


def merge_with_existing(
    new_chunks: List[str],
    new_sources: List[str],
    existing_chunks: List[str],
    existing_sources: List[str]
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Merge new chunks with existing chunks.
    Deduplicates identical chunks from the same source.

    Returns:
        Tuple of (deduped_new_chunks, deduped_new_sources, merged_chunks, merged_sources)
    """
    # Create a set of existing chunk identifiers to avoid duplicates
    existing_ids = {(chunk, source) for chunk, source in zip(existing_chunks, existing_sources)}
    
    merged_chunks = list(existing_chunks)
    merged_sources = list(existing_sources)
    deduped_new_chunks = []
    deduped_new_sources = []
    
    new_count = 0
    for chunk, source in zip(new_chunks, new_sources):
        if (chunk, source) not in existing_ids:
            merged_chunks.append(chunk)
            merged_sources.append(source)
            deduped_new_chunks.append(chunk)
            deduped_new_sources.append(source)
            existing_ids.add((chunk, source))
            new_count += 1
    
    logger.info(f"Merged: {new_count} new chunks added, {len(new_chunks) - new_count} duplicates skipped")
    return deduped_new_chunks, deduped_new_sources, merged_chunks, merged_sources
