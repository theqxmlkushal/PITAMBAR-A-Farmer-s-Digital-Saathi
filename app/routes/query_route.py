"""
PITAMBAR — app/routes/query_route.py
FastAPI router: POST /query  and  POST /query/voice
"""

import logging
from typing import Optional

from fastapi import APIRouter, Form, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.controllers.query_controller import handle_text_query, handle_voice_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------
class TextQueryRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Farmer's question in any supported language.")
    session_id: str = Field("default", description="Session identifier for conversation memory.")


# ---------------------------------------------------------------------------
# POST /query  — plain text
# ---------------------------------------------------------------------------
@router.post(
    "/",
    summary="Text query",
    description="Submit a plain-text agricultural question; receive an advisory response.",
)
async def text_query(request: TextQueryRequest) -> JSONResponse:
    """
    Accepts a JSON body with ``message`` and optional ``session_id``.
    Delegates to the orchestrator and returns the full result dict.
    """
    logger.info("POST /query  session=%s  msg=%r", request.session_id, request.message[:80])

    result = await handle_text_query(
        message=request.message,
        session_id=request.session_id,
    )

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# POST /query/voice  — audio upload
# ---------------------------------------------------------------------------
@router.post(
    "/voice",
    summary="Voice query",
    description="Upload a WAV audio file; receive text advisory + base64-encoded MP3 reply.",
)
async def voice_query(
    audio: UploadFile = File(..., description="WAV audio file containing the farmer's spoken query."),
    session_id: str = Form("default", description="Session identifier for conversation memory."),
    language: str = Form("hi", description="BCP-47 language code for STT hint and TTS output."),
) -> JSONResponse:
    """
    Accepts ``multipart/form-data`` with:
    - ``audio``: WAV file
    - ``session_id``: string (optional, default "default")
    - ``language``: BCP-47 code (optional, default "hi")

    Returns the orchestrator result plus ``audio_base64`` (MP3).
    """
    logger.info(
        "POST /query/voice  session=%s  lang=%s  filename=%s",
        session_id, language, audio.filename,
    )

    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

    result = await handle_voice_query(
        audio_bytes=audio_bytes,
        session_id=session_id,
        response_language=language,
    )

    if result.get("error"):
        # Transcription failure is a 422; pipeline failure is 500
        status = 422 if result["error"] == "Could not transcribe audio" else 500
        raise HTTPException(status_code=status, detail=result["error"])

    return JSONResponse(content=result)


# ---------------------------------------------------------------------------
# Standalone test (import check only — routes need a running app to invoke)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== PITAMBAR: query_route import check ===")
    print(f"Router prefix : {router.prefix}")
    print(f"Routes defined: {[r.path for r in router.routes]}")
    print("Import OK ✓")