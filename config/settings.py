# config/settings.py
# Central configuration for PITAMBAR.
# All environment variables are loaded here via python-dotenv + Pydantic.
# Import `settings` anywhere in the project — never read os.environ directly.

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings  # pydantic v2 split package

# ---------------------------------------------------------------------------
# Load .env from the project root (two levels up from this file)
# ---------------------------------------------------------------------------
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

# ---------------------------------------------------------------------------
# Constants (not env-driven)
# ---------------------------------------------------------------------------
LOG_LEVEL: str = "INFO"

INTENT_TYPES: tuple[str, ...] = ("market", "policy", "tech", "crop", "multi")

SUPPORTED_LANGUAGES: tuple[str, ...] = (
    "en",   # English
    "hi",   # Hindi
    "mr",   # Marathi
    "pa",   # Punjabi
    "te",   # Telugu
    "ta",   # Tamil
    "kn",   # Kannada
    "gu",   # Gujarati
    "bn",   # Bengali
)

# ---------------------------------------------------------------------------
# Settings schema
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    # --- LLM ---
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")

    # --- Session ---
    session_ttl_seconds: int = Field(1800, env="SESSION_TTL_SECONDS")

    # --- Vector DB ---
    qdrant_collection: str = Field("pitambar_docs", env="QDRANT_COLLECTION")

    # --- Speech ---
    whisper_model: str = Field("base", env="WHISPER_MODEL")

    # --- Embedding ---
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL"
    )

    # --- Weather API ---
    open_meteo_base_url: str = Field(
        "https://api.open-meteo.com/v1/forecast", env="OPEN_METEO_BASE_URL"
    )

    # --- App ---
    app_host: str = Field("0.0.0.0", env="APP_HOST")
    app_port: int = Field(8000, env="APP_PORT")
    log_level: str = Field(LOG_LEVEL, env="LOG_LEVEL")

    # --- RAG ---
    rag_top_k: int = Field(5, env="RAG_TOP_K")
    data_documents_dir: str = Field("data/documents", env="DATA_DOCUMENTS_DIR")

    class Config:
        env_file = str(_ENV_PATH)
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# ---------------------------------------------------------------------------
# Singleton export
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]


settings: Settings = _load_settings()

# ---------------------------------------------------------------------------
# Configure root logger once at import time
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

# ---------------------------------------------------------------------------
# __main__ smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PITAMBAR Settings ===")
    print(f"  groq_model          : {settings.groq_model}")
    print(f"  qdrant_collection   : {settings.qdrant_collection}")
    print(f"  session_ttl_seconds : {settings.session_ttl_seconds}")
    print(f"  whisper_model       : {settings.whisper_model}")
    print(f"  embedding_model     : {settings.embedding_model}")
    print(f"  open_meteo_base_url : {settings.open_meteo_base_url}")
    print(f"  app_host            : {settings.app_host}:{settings.app_port}")
    print(f"  log_level           : {settings.log_level}")
    print(f"  rag_top_k           : {settings.rag_top_k}")
    print(f"  LOG_LEVEL constant  : {LOG_LEVEL}")
    print(f"  INTENT_TYPES        : {INTENT_TYPES}")
    print("Settings loaded successfully ✓")