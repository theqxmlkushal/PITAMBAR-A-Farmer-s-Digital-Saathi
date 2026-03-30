# tools/rag/vector_store.py
# In-memory Qdrant vector store with sentence-transformers embeddings.
# Uses qdrant-client v1.x API (query_points, not search).
# The client and model are module-level singletons — loaded once on first use.

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

from config.settings import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_VECTOR_SIZE   = 384          # all-MiniLM-L6-v2 output dimension
_DISTANCE      = Distance.COSINE
_COLLECTION    = settings.qdrant_collection
_EMBED_MODEL   = "all-MiniLM-L6-v2"   # short name works for SentenceTransformer

# ---------------------------------------------------------------------------
# Module-level singletons (lazy init)
# ---------------------------------------------------------------------------
_client: QdrantClient | None = None
_model:  SentenceTransformer | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        logger.info("Initialising Qdrant in-memory client")
        _client = QdrantClient(":memory:")
    return _client


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading SentenceTransformer model: %s", _EMBED_MODEL)
        _model = SentenceTransformer(_EMBED_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Collection management
# ---------------------------------------------------------------------------

def init_collection() -> None:
    """Create the Qdrant collection if it does not already exist (idempotent)."""
    client = _get_client()
    if not client.collection_exists(_COLLECTION):
        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(size=_VECTOR_SIZE, distance=_DISTANCE),
        )
        logger.info("Created Qdrant collection '%s'", _COLLECTION)
    else:
        logger.debug("Collection '%s' already exists — skipping create", _COLLECTION)


# ---------------------------------------------------------------------------
# Encoding helper (blocking — run in thread)
# ---------------------------------------------------------------------------

def _encode(texts: list[str]) -> list[list[float]]:
    """Encode a list of texts with the sentence transformer. Blocking."""
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def _doc_uuid(doc_id: str) -> str:
    """Derive a deterministic UUID string from an arbitrary doc id string."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, doc_id))


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------

async def index_documents(docs: list[dict[str, Any]]) -> int:
    """Encode and upsert documents into the vector store.

    Each doc must have: {"id": str, "text": str, "metadata": dict}

    Returns:
        Number of documents successfully indexed.
    """
    if not docs:
        return 0

    init_collection()
    texts = [d["text"] for d in docs]

    # Encode in a thread — SentenceTransformer is CPU-bound
    vectors: list[list[float]] = await asyncio.to_thread(_encode, texts)

    points = [
        PointStruct(
            id=_doc_uuid(doc["id"]),
            vector=vectors[i],
            payload={
                "doc_id":   doc["id"],
                "text":     doc["text"],
                **doc.get("metadata", {}),
            },
        )
        for i, doc in enumerate(docs)
    ]

    client = _get_client()
    client.upsert(collection_name=_COLLECTION, points=points)
    logger.info("Indexed %d documents into '%s'", len(points), _COLLECTION)
    return len(points)


async def search(query: str, top_k: int = 3) -> list[dict[str, Any]]:
    """Semantic search over indexed documents.

    Returns:
        List of dicts: {"text": str, "score": float, "metadata": dict}
    """
    if not query.strip():
        return []

    init_collection()

    # Encode query in a thread
    query_vector: list[float] = (await asyncio.to_thread(_encode, [query]))[0]

    client = _get_client()
    response = client.query_points(
        collection_name=_COLLECTION,
        query=query_vector,
        limit=top_k,
        with_payload=True,
    )

    results: list[dict[str, Any]] = []
    for hit in response.points:
        payload = hit.payload or {}
        results.append(
            {
                "text":     payload.get("text", ""),
                "score":    round(hit.score, 4),
                "metadata": {k: v for k, v in payload.items() if k not in ("text", "doc_id")},
            }
        )

    logger.debug("search | query='%s' | hits=%d", query[:60], len(results))
    return results


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio as _asyncio

    _SAMPLE_DOCS = [
        {"id": "doc-1", "text": "Wheat rust disease causes orange pustules on leaves. Apply propiconazole fungicide.", "metadata": {"source": "agri_guide.txt"}},
        {"id": "doc-2", "text": "PM-KISAN provides Rs 6000 per year to eligible small and marginal farmers.", "metadata": {"source": "schemes.txt"}},
        {"id": "doc-3", "text": "Drip irrigation reduces water usage by 40-50% and increases crop yield significantly.", "metadata": {"source": "irrigation.txt"}},
        {"id": "doc-4", "text": "Tomato early blight shows concentric ring spots on leaves. Use mancozeb spray.", "metadata": {"source": "agri_guide.txt"}},
        {"id": "doc-5", "text": "Onion mandi prices in Pune range from Rs 800 to Rs 2000 per quintal.", "metadata": {"source": "market.txt"}},
    ]

    async def _test() -> None:
        print("=== vector_store smoke-test ===\n")

        print("Step 1: init_collection()")
        init_collection()
        print("  collection created ✓\n")

        print("Step 2: index_documents()")
        count = await index_documents(_SAMPLE_DOCS)
        print(f"  indexed {count} documents ✓\n")

        print("Step 3: search('wheat disease')")
        hits = await search("wheat disease", top_k=3)
        for i, h in enumerate(hits, 1):
            print(f"  {i}. score={h['score']}  text={h['text'][:80]}")
            print(f"     metadata={h['metadata']}")
        print()

        print("Step 4: search('government subsidy scheme')")
        hits2 = await search("government subsidy scheme", top_k=2)
        for i, h in enumerate(hits2, 1):
            print(f"  {i}. score={h['score']}  text={h['text'][:80]}")
        print()

        print("vector_store ✓")

    _asyncio.run(_test())