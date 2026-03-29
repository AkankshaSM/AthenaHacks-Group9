import base64
import json
import os
from dataclasses import dataclass
from typing import Any

import requests


class GeminiServiceError(Exception):
    pass


@dataclass
class GeminiResult:
    payload: dict[str, Any]


class GeminiClient:
    def __init__(self) -> None:
        self.api_key = os.getenv("GEMINI_API_KEY", "")
        self.model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def _require_api_key(self) -> None:
        if not self.api_key:
            raise GeminiServiceError("Missing GEMINI_API_KEY environment variable")

    def _extract_text(self, response_json: dict[str, Any]) -> str:
        candidates = response_json.get("candidates", [])
        if not candidates:
            raise GeminiServiceError("Gemini returned no candidates")
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(part.get("text", "") for part in parts if "text" in part).strip()
        if not text:
            raise GeminiServiceError("Gemini returned empty text response")
        return text

    def _parse_json_text(self, text: str) -> dict[str, Any]:
        raw = text.strip()
        if raw.startswith("```"):
            raw = raw.replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GeminiServiceError(f"Gemini returned invalid JSON: {raw[:400]}") from exc

    def _repair_json_with_model(self, broken_json: str) -> dict[str, Any]:
        repair_prompt = f"""
Fix the following malformed JSON and return ONLY valid JSON with no markdown, comments, or extra text.

Malformed JSON:
{broken_json}
""".strip()

        payload = {
            "contents": [{"parts": [{"text": repair_prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 0.1,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
            },
        }
        response = requests.post(
            f"{self.base_url}?key={self.api_key}",
            json=payload,
            timeout=60,
        )
        if response.status_code >= 400:
            raise GeminiServiceError(f"Gemini API error {response.status_code}: {response.text[:600]}")
        repaired_text = self._extract_text(response.json())
        return self._parse_json_text(repaired_text)

    def _call_json(self, prompt: str, *, max_output_tokens: int = 2048) -> GeminiResult:
        self._require_api_key()

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "topP": 0.8,
                "maxOutputTokens": max_output_tokens,
                "responseMimeType": "application/json",
            },
        }
        response = requests.post(
            f"{self.base_url}?key={self.api_key}",
            json=payload,
            timeout=60,
        )
        if response.status_code >= 400:
            raise GeminiServiceError(f"Gemini API error {response.status_code}: {response.text[:600]}")

        text = self._extract_text(response.json())
        try:
            parsed = self._parse_json_text(text)
        except GeminiServiceError:
            parsed = self._repair_json_with_model(text)
        return GeminiResult(payload=parsed)

    def transcribe_audio(self, *, audio_bytes: bytes, mime_type: str) -> str:
        """Send audio to Gemini and return the transcribed text."""
        self._require_api_key()

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": audio_b64,
                            }
                        },
                        {
                            "text": (
                                "Transcribe the speech in this audio recording verbatim. "
                                "Return only the transcript text with no additional commentary."
                            )
                        },
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 4096,
            },
        }

        response = requests.post(
            f"{self.base_url}?key={self.api_key}",
            json=payload,
            timeout=120,
        )
        if response.status_code >= 400:
            raise GeminiServiceError(f"Gemini API error {response.status_code}: {response.text[:600]}")

        return self._extract_text(response.json())

    def generate_questions(self, context: str, script: str) -> list[str]:
        prompt = f"""
You are a speech-coaching assistant.
Task: validate whether context is sufficient to refine a speech script.

Rules:
- If context is enough, return empty questions.
- If context is insufficient, generate between 1 and 7 highly relevant clarifying questions.
- Questions must focus on audience, tone, duration, goal, delivery style, and formality when missing.
- Do not exceed 7 questions.
- Keep questions concise and specific.

Return ONLY valid JSON in this shape:
{{
  "needs_clarification": true/false,
  "questions": ["..."]
}}

Context:
{context}

Initial Script:
{script}
""".strip()

        result = self._call_json(prompt, max_output_tokens=8192).payload
        questions = result.get("questions", [])
        if not isinstance(questions, list):
            raise GeminiServiceError("Gemini questions payload is not a list")
        cleaned = [str(q).strip() for q in questions if str(q).strip()]
        return cleaned[:7]

    def refine_script(
        self,
        *,
        context: str,
        script: str,
        answers: list[dict[str, str]],
        previous_versions: list[str],
        instructions: str,
    ) -> str:
        prompt = f"""
You are an expert speech writing and coaching assistant.
Task: produce an improved script optimized for clarity, engagement, and delivery.

Constraints:
- Preserve user intent and factual meaning.
- Improve structure and flow.
- Keep language audience-appropriate based on context and Q&A.
- If instructions are provided, prioritize them.
- Return plain script text in one field.

Return ONLY valid JSON in this shape:
{{
  "refined_script": "..."
}}

Context:
{context}

Current Script:
{script}

Clarifying Answers (JSON):
{json.dumps(answers, ensure_ascii=False)}

Previous Versions (JSON array, newest first):
{json.dumps(previous_versions, ensure_ascii=False)}

Optional Instructions:
{instructions or ""}
""".strip()

        result = self._call_json(prompt).payload
        refined_script = str(result.get("refined_script", "")).strip()
        if not refined_script:
            raise GeminiServiceError("Gemini returned empty refined script")
        return refined_script

    def analyze_speech(
        self,
        *,
        final_script: str,
        metadata_json: dict[str, Any],
        context: str,
    ) -> dict[str, Any]:
        prompt = f"""
You are a professional speech and communication coach with expertise in delivery, pacing, and audience engagement.
You will analyze a recorded speech against its intended script and the delivery metadata JSON below.
Do NOT ignore any field in the metadata JSON — every metric must inform your feedback.

Your output must have three parts:

---
PART 1 — annotated_script
Rewrite the script with ElevenLabs v3 expressive delivery tags inserted at precise locations.
Allowed tags (use only these, verbatim):
  [short pause]  [long pause]  [whispers]  [shouting]  [dramatic tone]
  [conversational tone]  [quietly]  [loudly]  [curious]  [mischievously]
Rules:
- Insert tags ONLY where they meaningfully improve delivery.
- Base tag placement on the actual pause/speed/tone data in the metadata.
- Do not over-annotate — quality over quantity.

PART 2 — feedback_sections
Provide structured coaching feedback in exactly these 6 categories.
Each category MUST have 2–4 specific, evidence-based bullet points tied to the metadata:

1. Pacing & Speed
   - Comment on words-per-minute vs ideal range for context.
   - Identify moments that were too fast or too slow.

2. Pauses & Rhythm
   - Evaluate use of short pauses (under 0.8s) and long pauses (over 0.8s).
   - Recommend specific moments to add or remove pauses.

3. Clarity & Fluency
   - Assess overall clarity score.
   - Highlight words or sections needing stronger articulation.

4. Emotional Tone & Engagement
   - Evaluate whether emotional tone matched the context and audience.
   - Recommend where to intensify or soften energy.

5. Alignment with Context & Goal
   - Assess how well the delivery matched the stated context and purpose.
   - Note any structural or emphasis mismatches.

6. Priority Improvements
   - List the top 3 most impactful changes the speaker should make before their next delivery.

PART 3 — voice_recommendation
Recommend the best ElevenLabs voice for this context, tone, and audience.

---

Return ONLY valid JSON. No markdown. No commentary outside JSON. Use this exact shape:
{{
  "annotated_script": "...",
  "feedback_sections": [
    {{"category": "Pacing & Speed", "items": ["...", "..."]}},
    {{"category": "Pauses & Rhythm", "items": ["...", "..."]}},
    {{"category": "Clarity & Fluency", "items": ["...", "..."]}},
    {{"category": "Emotional Tone & Engagement", "items": ["...", "..."]}},
    {{"category": "Alignment with Context & Goal", "items": ["...", "..."]}},
    {{"category": "Priority Improvements", "items": ["...", "...", "..."]}}
  ],
  "feedback_summary": "One sentence overall assessment of the delivery.",
  "voice_recommendation": {{
    "voice_id": "",
    "name": "recommended voice name",
    "reason": "why this voice fits the context and tone"
  }}
}}

Context / Purpose of Speech:
{context}

Final Script:
{final_script}

Full Delivery Metadata JSON:
{json.dumps(metadata_json, ensure_ascii=False, indent=2)}
""".strip()

        result = self._call_json(prompt, max_output_tokens=8192).payload

        annotated_script = str(result.get("annotated_script", "")).strip()
        feedback_sections = result.get("feedback_sections", [])
        feedback_summary = str(result.get("feedback_summary", "")).strip()
        voice = result.get("voice_recommendation", {})

        if not annotated_script:
            raise GeminiServiceError("Gemini returned empty annotated script")

        if not isinstance(feedback_sections, list):
            feedback_sections = []

        # Normalise each section
        cleaned_sections = []
        for section in feedback_sections:
            if not isinstance(section, dict):
                continue
            items = section.get("items", [])
            cleaned_sections.append({
                "category": str(section.get("category", "General")).strip(),
                "items": [str(i).strip() for i in items if str(i).strip()],
            })

        if not isinstance(voice, dict):
            voice = {"name": "General Purpose", "reason": "Fallback recommendation"}

        return {
            "annotated_script": annotated_script,
            "feedback_sections": cleaned_sections,
            "feedback_summary": feedback_summary,
            "voice_recommendation": {
                "voice_id": str(voice.get("voice_id", "")).strip(),
                "name": str(voice.get("name", "General Purpose")).strip(),
                "reason": str(voice.get("reason", "Fit based on context and tone.")).strip(),
            },
        }
