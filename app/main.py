"""
PITAMBAR — app/main.py
FastAPI application entry point.
Startup sequence: init Qdrant collection → index documents → serve.
"""

import logging
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes.query_route import router as query_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  —  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

_BANNER = """
╔══════════════════════════════════════════════════╗
║          PITAMBAR Agricultural Advisor           ║
║     AI-powered advisory for Indian farmers       ║
║                 Version 1.0.0                    ║
╚══════════════════════════════════════════════════╝
"""

# ---------------------------------------------------------------------------
# Lifespan — replaces deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup tasks before serving; cleanup on shutdown."""
    # ── Startup ─────────────────────────────────────────────────────────────
    print(_BANNER)
    logger.info("Starting PITAMBAR service…")

    try:
        from tools.rag.vector_store import init_collection   # noqa: PLC0415
        from tools.rag.retriever import load_documents_from_folder  # noqa: PLC0415

        logger.info("Initialising Qdrant in-memory collection…")
        await init_collection()
        logger.info("Qdrant collection ready ✓")

        doc_folder = "data/documents/"
        logger.info("Loading documents from '%s'…", doc_folder)
        await load_documents_from_folder(doc_folder)
        logger.info("Document index ready ✓")

    except Exception as exc:
        # Non-fatal: app still starts; advisory quality degrades without RAG.
        logger.warning("RAG initialisation failed (non-fatal): %s", exc)

    logger.info("PITAMBAR is ready — listening on http://0.0.0.0:8000")

    yield  # ── server runs here ────────────────────────────────────────────

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("PITAMBAR shutting down…")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = FastAPI(
    title="PITAMBAR Agricultural Advisor",
    description=(
        "A production-grade multi-agent AI system that provides crop, market, "
        "policy, and technology advisory to Indian farmers in regional languages."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(query_router)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["System"], summary="Service health probe")
async def health_check() -> dict:
    """Lightweight liveness probe for load-balancers and monitoring."""
    return {"status": "ok", "service": "PITAMBAR", "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )