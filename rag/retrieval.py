"""
Retrieval module — semantic search over the FAISS index using TF-IDF embeddings.
Returns top-k relevant chunks for a given query.
"""
import logging
import numpy as np
from rag.embeddings import embed_query

logger = logging.getLogger(__name__)
TOP_K = 10


def retrieve(query: str, index, chunks: list, sources: list, k: int = TOP_K):
    """
    Retrieve the most relevant document chunks for a query.
    Returns list of dicts: {text, source, distance}
    """
    q_emb = embed_query(query)

    # If the query contains no vocabulary words from the corpus, TF-IDF returns a zero vector.
    # In that case, fallback to returning all document chunks so the LLM can still use the data.
    if np.linalg.norm(q_emb) <= 1e-8:
        logger.info("[RAG] Query vector is empty; returning all chunks as fallback.")
        return [
            {"text": chunk, "source": source, "distance": 0.0}
            for chunk, source in zip(chunks, sources)
        ]

    distances, indices = index.search(q_emb, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            "text": chunks[idx],
            "source": sources[idx],
            "distance": float(dist),
        })

    logger.info("[RAG] Retrieved %d chunks for query '%s'", len(results), query)
    for i, r in enumerate(results[:k], 1):
        logger.info("[RAG]  %d. source=%s distance=%.6f text=%s", i, r['source'], r['distance'], r['text'][:120])

    # Keep top-k results as-is.
    return results[:k]


def format_context(results: list) -> str:
    """Format retrieved chunks into a clean context block for the LLM."""
    if not results:
        return ""
    lines = []
    seen = set()
    for r in results:
        text = r["text"]
        if text not in seen:
            seen.add(text)
            lines.append(f"[{r['source']}] {text}")
    return "\n".join(lines)
