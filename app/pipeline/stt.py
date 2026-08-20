"""Speech-to-Text via Sarvam Saaras v3 API.

Handles audio transcription and multilingual translation into English.
"""

import httpx
import time
import logging

from app.config import settings
from app.models import STTResponse

logger = logging.getLogger(__name__)


class SarvamSTT:
    """Client for Sarvam Saaras API."""

    def __init__(self):
        self.api_url = settings.sarvam_api_url
        self.api_key = settings.sarvam_api_key

    async def transcribe_and_translate(self, audio_bytes: bytes, filename: str = "audio.webm") -> STTResponse:
        """Send audio to Sarvam to transcribe and translate to English."""
        start_time = time.perf_counter()

        if not self.api_key:
            # Fallback for local testing without an API key
            logger.warning("No Sarvam API key found. Returning mock STT result.")
            return STTResponse(
                text="What is the chemical formula for water?",
                source_language="en",
                was_translated=False,
                confidence=1.0,
                latency_ms=(time.perf_counter() - start_time) * 1000
            )

        headers = {
            "api-subscription-key": self.api_key
        }
        
        # multipart/form-data
        files = {
            "file": (filename, audio_bytes, "audio/webm")
        }
        
        data = {
            "prompt": "",
            "model": "saaras:v1" # Let's use standard default unless v3 is strictly required via parameter. 
        }

        try:
            async with httpx.AsyncClient(timeout=settings.stt_timeout_s) as client:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    files=files,
                    data={"model": "saaras:v3"} 
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Assuming standard Sarvam translate output: {"transcript": "text", "language_code": "hi"}
                # Or {"text": "..."} depending on API version.
                # Let's extract safely.
                text = result.get("transcript", result.get("text", ""))
                lang = result.get("language_code", "en")
                
                latency = (time.perf_counter() - start_time) * 1000
                
                return STTResponse(
                    text=text.strip(),
                    source_language=lang,
                    was_translated=(lang != "en" and lang != "en-IN"),
                    confidence=1.0,
                    latency_ms=latency
                )
                
        except httpx.HTTPError as e:
            logger.error(f"Sarvam API failed: {e}")
            return STTResponse(
                text="",
                source_language="en",
                was_translated=False,
                confidence=0.0,
                latency_ms=(time.perf_counter() - start_time) * 1000
            )
