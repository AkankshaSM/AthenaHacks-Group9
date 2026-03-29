import json
import os
from typing import Any

import requests


class ElevenLabsServiceError(Exception):
    pass


class ElevenLabsClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("ELEVENLABS_API_KEY", "")
        self.model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_v3")
        self.stt_model_id = os.getenv("ELEVENLABS_STT_MODEL_ID", "scribe_v1")
        self.default_voice_id = os.getenv("ELEVENLABS_DEFAULT_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
        self.analysis_endpoint = os.getenv("ELEVENLABS_ANALYSIS_ENDPOINT", "").strip()

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ElevenLabsServiceError("Missing ELEVENLABS_API_KEY environment variable")
        return {"xi-api-key": self.api_key}

    def _derive_metadata_from_stt(self, stt_json: dict[str, Any]) -> dict[str, Any]:
        words = stt_json.get("words", [])
        if not isinstance(words, list):
            words = []

        normalized_words: list[dict[str, Any]] = []
        for item in words:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = float(item.get("start", 0.0) or 0.0)
            end = float(item.get("end", start) or start)
            confidence = item.get("confidence")
            normalized_words.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
                    "confidence": float(confidence) if isinstance(confidence, (int, float)) else None,
                }
            )

        duration_seconds = 0.0
        if normalized_words:
            duration_seconds = max(normalized_words[-1]["end"] - normalized_words[0]["start"], 0.0)

        word_count = len(normalized_words)
        speaking_rate_wpm = 0.0
        if duration_seconds > 0:
            speaking_rate_wpm = round((word_count / duration_seconds) * 60.0, 2)

        short_pauses: list[dict[str, float]] = []
        long_pauses: list[dict[str, float]] = []
        for idx in range(1, len(normalized_words)):
            prev_end = normalized_words[idx - 1]["end"]
            current_start = normalized_words[idx]["start"]
            gap = round(max(current_start - prev_end, 0.0), 3)
            if gap < 0.25:
                continue
            pause_entry = {
                "after_word_index": idx - 1,
                "duration_seconds": gap,
            }
            if gap <= 0.8:
                short_pauses.append(pause_entry)
            else:
                long_pauses.append(pause_entry)

        confidence_values = [w["confidence"] for w in normalized_words if isinstance(w.get("confidence"), float)]
        clarity_score = round((sum(confidence_values) / len(confidence_values)) * 100.0, 2) if confidence_values else None

        return {
            "provider": "elevenlabs",
            "analysis_source": "speech_to_text_fallback",
            "transcript": str(stt_json.get("text", "")),
            "timestamps": normalized_words,
            "speaking_rate": {
                "words_per_minute": speaking_rate_wpm,
                "word_count": word_count,
                "duration_seconds": round(duration_seconds, 3),
            },
            "pauses": {
                "short_pause_count": len(short_pauses),
                "long_pause_count": len(long_pauses),
                "short_pauses": short_pauses,
                "long_pauses": long_pauses,
            },
            "clarity": {
                "confidence_avg_percent": clarity_score,
                "note": "Estimated from STT confidence.",
            },
            "pitch_variation": {
                "value": None,
                "note": "Pitch metrics are not returned by ElevenLabs STT endpoint.",
            },
            "emotional_tone": {
                "value": "unknown",
                "note": "Emotional tone requires secondary model inference.",
            },
            "raw_json": stt_json,
        }

    def _analyze_with_speech_to_text(self, *, file_bytes: bytes, filename: str, content_type: str | None) -> dict[str, Any]:
        files = {
            "file": (filename, file_bytes, content_type or "audio/webm"),
        }
        data = {
            "model_id": self.stt_model_id,
            "timestamps_granularity": "word",
        }
        response = requests.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers=self._headers(),
            files=files,
            data=data,
            timeout=120,
        )
        if response.status_code >= 400:
            raise ElevenLabsServiceError(
                f"ElevenLabs speech-to-text error {response.status_code}: {response.text[:600]}"
            )

        try:
            stt_json = response.json()
        except json.JSONDecodeError as exc:
            raise ElevenLabsServiceError("ElevenLabs speech-to-text response is not valid JSON") from exc

        return self._derive_metadata_from_stt(stt_json)

    def analyze_audio(self, *, file_bytes: bytes, filename: str, content_type: str | None) -> dict[str, Any]:
        if not self.analysis_endpoint:
            return self._analyze_with_speech_to_text(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
            )

        files = {
            "file": (filename, file_bytes, content_type or "audio/webm"),
        }
        response = requests.post(
            self.analysis_endpoint,
            headers=self._headers(),
            files=files,
            timeout=120,
        )
        if response.status_code == 404:
            return self._analyze_with_speech_to_text(
                file_bytes=file_bytes,
                filename=filename,
                content_type=content_type,
            )

        if response.status_code >= 400:
            raise ElevenLabsServiceError(f"ElevenLabs analysis error {response.status_code}: {response.text[:600]}")

        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise ElevenLabsServiceError("ElevenLabs analysis response is not valid JSON") from exc

    def synthesize(self, *, text: str, voice_id: str | None = None) -> bytes:
        selected_voice_id = voice_id or self.default_voice_id
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{selected_voice_id}"
        payload = {
            "text": text,
            "model_id": self.model_id,
            "output_format": "mp3_44100_128",
        }

        response = requests.post(
            url,
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code >= 400:
            raise ElevenLabsServiceError(f"ElevenLabs synthesis error {response.status_code}: {response.text[:600]}")
        return response.content
