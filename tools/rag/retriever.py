# tools/rag/retriever.py
# High-level retrieval interface used by all agents.
# Formats vector_store results into a plain-text context block.
# Also provides load_documents_from_folder() for bootstrapping the store.

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from config.settings import settings
from tools.rag.vector_store import index_documents, search

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public: context retrieval
# ---------------------------------------------------------------------------

async def retrieve_context(query: str, top_k: int | None = None) -> str:
    """Retrieve semantically relevant context for *query* from the vector store.

    Args:
        query:  The farmer's query or a distilled search phrase.
        top_k:  Number of results. Defaults to settings.rag_top_k.

    Returns:
        A formatted multi-line string ready to be injected into an LLM prompt.
        Empty string if the store has no relevant documents.
    """
    k = top_k if top_k is not None else settings.rag_top_k
    results = await search(query, top_k=k)

    if not results:
        logger.debug("retrieve_context | no results for query='%s'", query[:60])
        return ""

    lines = ["Relevant context:"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['text']}")
        if r["metadata"]:
            source = r["metadata"].get("source", "")
            if source:
                lines.append(f"   (source: {source}, relevance: {r['score']})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public: document loader
# ---------------------------------------------------------------------------

async def load_documents_from_folder(folder_path: str | None = None) -> int:
    """Read .txt and .pdf files from *folder_path* and index them.

    Args:
        folder_path: Directory to scan. Defaults to settings.data_documents_dir.

    Returns:
        Total number of documents indexed (0 if folder is empty or missing).
    """
    base = Path(folder_path or settings.data_documents_dir)
    if not base.exists():
        logger.warning("load_documents_from_folder: path does not exist: %s", base)
        return 0

    docs: list[dict] = []

    # --- .txt files ---
    for txt_file in sorted(base.glob("*.txt")):
        try:
            text = txt_file.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                docs.append(
                    {
                        "id":       txt_file.name,
                        "text":     text,
                        "metadata": {"source": txt_file.name, "type": "txt"},
                    }
                )
                logger.debug("Loaded txt: %s (%d chars)", txt_file.name, len(text))
        except Exception as exc:
            logger.warning("Could not read '%s': %s", txt_file, exc)

    # --- .pdf files (PyPDF2 optional) ---
    pdf_files = sorted(base.glob("*.pdf"))
    if pdf_files:
        try:
            import PyPDF2  # optional dependency — skip gracefully if absent

            for pdf_file in pdf_files:
                try:
                    text = await asyncio.to_thread(_extract_pdf_text, pdf_file)
                    if text:
                        docs.append(
                            {
                                "id":       pdf_file.name,
                                "text":     text,
                                "metadata": {"source": pdf_file.name, "type": "pdf"},
                            }
                        )
                        logger.debug(
                            "Loaded pdf: %s (%d chars)", pdf_file.name, len(text)
                        )
                except Exception as exc:
                    logger.warning("Could not read PDF '%s': %s", pdf_file, exc)

        except ImportError:
            logger.info(
                "PyPDF2 not installed — skipping %d PDF file(s). "
                "Run: pip install PyPDF2",
                len(pdf_files),
            )

    if not docs:
        logger.info("load_documents_from_folder: no documents found in '%s'", base)
        return 0

    indexed = await index_documents(docs)
    logger.info(
        "load_documents_from_folder: indexed %d/%d docs from '%s'",
        indexed, len(docs), base,
    )
    return indexed


# ---------------------------------------------------------------------------
# Internal PDF helper (blocking — called via to_thread)
# ---------------------------------------------------------------------------

def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF using PyPDF2. Blocking."""
    import PyPDF2

    pages: list[str] = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(page_text.strip())
    return "\n\n".join(pages)


# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio as _asyncio

    _SAMPLE_DOCS = [
        {
            "id":       "wheat-rust",
            "text":     "Wheat rust disease causes orange pustules on leaves. Apply propiconazole fungicide at early stage.",
            "metadata": {"source": "crop_guide.txt"},
        },
        {
            "id":       "pm-kisan",
            "text":     "PM-KISAN scheme provides Rs 6000 per year direct income support to all land-holding farmer families.",
            "metadata": {"source": "schemes.txt"},
        },
        {
            "id":       "drip-irrig",
            "text":     "Drip irrigation can reduce water consumption by 40-50%. PMKSY subsidy covers up to 55% of equipment cost.",
            "metadata": {"source": "irrigation.txt"},
        },
    ]

    async def _test() -> None:
        print("=== retriever smoke-test ===\n")

        # Index sample docs
        from tools.rag.vector_store import init_collection
        init_collection()
        count = await index_documents(_SAMPLE_DOCS)
        print(f"Indexed {count} sample documents\n")

        # Test retrieve_context
        queries = ["wheat disease treatment", "government scheme for farmers", "water saving irrigation"]
        for q in queries:
            print(f"Query : '{q}'")
            ctx = await retrieve_context(q, top_k=2)
            print(ctx if ctx else "  (no context returned)")
            print()

        # Test empty query
        print("Query : '' (empty)")
        ctx_empty = await retrieve_context("")
        print(f"  result: '{ctx_empty}' (expected empty string)")
        print()

        print("retriever ✓")

    _asyncio.run(_test())