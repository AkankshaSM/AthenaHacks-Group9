import json
import os
from typing import Any

import requests


# Stable premade ElevenLabs voice IDs — no voices_read permission needed.
KNOWN_VOICES: list[dict[str, str]] = [
    {"voice_id": "21m00Tcm4TlvDq8ikWAM", "name": "Rachel",   "gender": "female", "accent": "American",   "description": "calm, narrative",           "energy": "low",    "use_cases": "audiobooks, meditation, storytelling"},
    {"voice_id": "29vD33N1CtxCmqQRPOHJ", "name": "Drew",     "gender": "male",   "accent": "American",   "description": "well-rounded, medium",       "energy": "medium", "use_cases": "general purpose, explainers, interviews"},
    {"voice_id": "EXAVITQu4vr4xnSDxMaL", "name": "Sarah",    "gender": "female", "accent": "American",   "description": "soft, news",                 "energy": "low",    "use_cases": "news, e-learning, corporate"},
    {"voice_id": "ErXwobaYiN019PkySvjV", "name": "Antoni",   "gender": "male",   "accent": "American",   "description": "well-rounded",               "energy": "medium", "use_cases": "general purpose, presentations"},
    {"voice_id": "GBv7mTt0atIp3Br8iCZE", "name": "Thomas",   "gender": "male",   "accent": "American",   "description": "calm",                       "energy": "low",    "use_cases": "meditation, bedtime stories, soft narration"},
    {"voice_id": "IKne3meq5aSn9XLyUdCD", "name": "Charlie",  "gender": "male",   "accent": "Australian", "description": "natural, conversational",    "energy": "medium", "use_cases": "casual talks, podcasts, social media"},
    {"voice_id": "JBFqnCBsd6RMkjVDRZzb", "name": "George",   "gender": "male",   "accent": "British",    "description": "warm, authoritative",        "energy": "medium", "use_cases": "documentaries, corporate, professional speeches"},
    {"voice_id": "LcfcDJNUP1GQjkzn1xUU", "name": "Emily",    "gender": "female", "accent": "American",   "description": "calm, composed",             "energy": "low",    "use_cases": "corporate, e-learning, professional"},
    {"voice_id": "MF3mGyEYCl7XYWbV9V6O", "name": "Elli",     "gender": "female", "accent": "American",   "description": "emotional, upbeat",          "energy": "high",   "use_cases": "social media, motivational speeches, ads, youth audience"},
    {"voice_id": "N2lVS1w4EtoT3dr4eOWO", "name": "Callum",   "gender": "male",   "accent": "American",   "description": "intense, characters",        "energy": "high",   "use_cases": "drama, characters, persuasive speeches, debates"},
    {"voice_id": "TX3LPaxmHKxFdv7VOQHJ", "name": "Liam",     "gender": "male",   "accent": "American",   "description": "narration",                  "energy": "medium", "use_cases": "audiobooks, explainers, narration"},
    {"voice_id": "ThT5KcBeYPX3keUQqHPh", "name": "Dorothy",  "gender": "female", "accent": "British",    "description": "pleasant, expressive",       "energy": "medium", "use_cases": "children content, storytelling, friendly presentations"},
    {"voice_id": "TxGEqnHWrfWFTfGW9XjX", "name": "Josh",     "gender": "male",   "accent": "American",   "description": "deep, resonant",             "energy": "medium", "use_cases": "trailers, promos, public speaking, authority roles"},
    {"voice_id": "VR6AewLTigWG4xSOukaG", "name": "Arnold",   "gender": "male",   "accent": "American",   "description": "crisp, narration",           "energy": "medium", "use_cases": "narration, documentaries, news"},
    {"voice_id": "XrExE9yKIg1WjnnlVkGX", "name": "Matilda",  "gender": "female", "accent": "American",   "description": "warm, friendly",             "energy": "medium", "use_cases": "customer service, friendly explainers, lifestyle"},
    {"voice_id": "onwK4e9ZLuTAKqWW03F9", "name": "Daniel",   "gender": "male",   "accent": "British",    "description": "authoritative, news",        "energy": "medium", "use_cases": "news, formal speeches, corporate, TED-style talks"},
    {"voice_id": "pFZP5JQG7iQjIQuC4Bku", "name": "Lily",     "gender": "female", "accent": "British",    "description": "warm, expressive",           "energy": "medium", "use_cases": "storytelling, lifestyle, expressive narration"},
    {"voice_id": "pNInz6obpgDQGcFmaJgB", "name": "Adam",     "gender": "male",   "accent": "American",   "description": "deep, narrative",            "energy": "low",    "use_cases": "audiobooks, deep narration, documentary"},
    {"voice_id": "AZnzlk1XvdvUeBnXmlld", "name": "Domi",     "gender": "female", "accent": "American",   "description": "strong, confident",          "energy": "high",   "use_cases": "motivational speeches, public speaking, leadership talks"},
    {"voice_id": "CYw3kZ02Hs0563khs1Fj", "name": "Dave",     "gender": "male",   "accent": "British",    "description": "conversational",             "energy": "medium", "use_cases": "podcasts, casual talks, interviews"},
]


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

    def list_voices(self) -> list[dict[str, Any]]:
        response = requests.get(
            "https://api.elevenlabs.io/v1/voices",
            headers=self._headers(),
            timeout=60,
        )
        if response.status_code >= 400:
            raise ElevenLabsServiceError(f"ElevenLabs voices error {response.status_code}: {response.text[:600]}")

        try:
            payload = response.json()
        except json.JSONDecodeError as exc:
            raise ElevenLabsServiceError("ElevenLabs voices response is not valid JSON") from exc

        voices = payload.get("voices", [])
        return voices if isinstance(voices, list) else []

    def resolve_voice_id(self, *, preferred_voice_id: str | None, preferred_name: str | None) -> str:
        voices = self.list_voices()
        if not voices:
            return self.default_voice_id

        # 1) Keep preferred voice_id only if it exists for this account.
        if preferred_voice_id:
            for voice in voices:
                if str(voice.get("voice_id", "")).strip() == preferred_voice_id:
                    return preferred_voice_id

        preferred_name_norm = (preferred_name or "").strip().lower()
        if preferred_name_norm:
            # 2) Exact name match.
            for voice in voices:
                if str(voice.get("name", "")).strip().lower() == preferred_name_norm:
                    return str(voice.get("voice_id", "")).strip() or self.default_voice_id

            # 3) Partial name match.
            for voice in voices:
                name = str(voice.get("name", "")).strip().lower()
                if preferred_name_norm in name or name in preferred_name_norm:
                    return str(voice.get("voice_id", "")).strip() or self.default_voice_id

        # 4) Prefer configured default if present.
        for voice in voices:
            if str(voice.get("voice_id", "")).strip() == self.default_voice_id:
                return self.default_voice_id

        # 5) Last resort first available voice.
        return str(voices[0].get("voice_id", "")).strip() or self.default_voice_id

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
            normalized_words.append(
                {
                    "text": text,
                    "start": start,
                    "end": end,
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
            # Low stability (Creative mode) makes eleven_v3 responsive to audio tags.
            "voice_settings": {
                "stability": 0.35,
                "similarity_boost": 0.75,
                "style": 0.45,
                "use_speaker_boost": True,
            },
        }

        response = requests.post(
            url,
            params={"output_format": "mp3_44100_128"},
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code >= 400:
            # Some Gemini recommendations may reference voices that are not available for this account.
            # Retry once with the configured default voice to keep the flow resilient.
            response_text = (response.text or "")[:600]
            if (
                response.status_code == 404
                and "voice_not_found" in response_text
                and selected_voice_id != self.default_voice_id
            ):
                fallback_url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.default_voice_id}"
                fallback_response = requests.post(
                    fallback_url,
                    headers={**self._headers(), "Content-Type": "application/json"},
                    json=payload,
                    timeout=120,
                )
                if fallback_response.status_code >= 400:
                    raise ElevenLabsServiceError(
                        f"ElevenLabs synthesis error {fallback_response.status_code}: {fallback_response.text[:600]}"
                    )
                return fallback_response.content

            raise ElevenLabsServiceError(f"ElevenLabs synthesis error {response.status_code}: {response_text}")
        return response.content
