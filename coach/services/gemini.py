import base64
import json
import os
from dataclasses import dataclass
from typing import Any

import requests

from coach.services.elevenlabs import KNOWN_VOICES


QUESTION_PROMPT = """
You are an expert speech coach.

Given the speaking context and draft script, decide whether context is sufficient.
If sufficient, return an empty question list.
If insufficient, return highly relevant clarifying questions.

Rules:
- At most 7 questions.
- Questions must target audience, tone, duration, intent, stakes, venue, and desired outcome.
- Avoid generic or repetitive questions.
- Return strict JSON only.

JSON schema:
{{
  "context_sufficient": boolean,
  "questions": ["string"]
}}

Context:
{context}

Draft:
{draft}
""".strip()


REFINE_PROMPT = """
You are refining speech content for delivery quality.

Use:
- Context
- Clarifying question answers
- Current script
- Additional instruction from user
- Prior refinements history

Goals:
- Preserve intent
- Increase clarity and flow
- Optimize rhetorical pacing
- Fit requested tone and audience
- Keep language natural and speakable

Return strict JSON only.

JSON schema:
{{
  "refined_script": "string"
}}

Context:
{context}

Answers:
{answers}

Additional instruction:
{instruction}

Current script:
{script}

History snippets:
{history}
""".strip()


ANALYSIS_PROMPT = """
You are a speech delivery analyst AND an expert at enhancing dialogue for ElevenLabs v3 speech synthesis.

STEP 1 — ANALYZE (use all provided inputs before writing anything):
Study the full speech metadata JSON, original script, and context carefully. Analyze:
- Speaking speed
- Pauses
- Clarity
- Emphasis
- Emotional tone
- Fluency
- Alignment with the stated context, audience, and purpose

Only after completing this analysis, produce the three outputs below.

Then produce three outputs:

1) ANNOTATED SCRIPT
Your role is to enhance the script for ElevenLabs v3 TTS by integrating audio tags.
Follow ALL of these directives strictly:

DO:
- Integrate audio tags to add expression, emotion, and realism — tags MUST describe something auditory.
- Ensure every tag is contextually appropriate and genuinely enhances the emotion or subtext at that moment.
- Strive for a diverse range of emotional expressions reflecting the nuances of the speech.
- Place tags immediately before OR after the segment they modify (e.g. "[annoyed] This is hard." or "This is hard. [sighs]").
- Add emphasis by capitalising key words, adding ellipses, or punctuation (! or ?) where it fits naturally.

DO NOT:
- Alter, add, or remove ANY words from the original script. Only add tags and emphasis punctuation.
- Create tags from existing narrative descriptions in the text (e.g. if text says "he laughed", do NOT change to "[laughs] he laughed" — instead add "[chuckles]" after if appropriate).
- Use non-auditory tags like [standing], [grinning], [pacing], [music].
- Use sound effect tags (gunshot, explosion, applause) for speech coaching scripts — they are off-limits here.
- Invent new dialogue lines.
- Select tags that contradict the original meaning or intent.

Available audio tags (non-exhaustive — you may infer similar auditory tags):
  Emotional:   [happy], [sad], [excited], [angry], [annoyed], [appalled], [thoughtful],
               [surprised], [sarcastic], [curious], [mischievously], [crying]
  Vocal style: [whispers], [sings], [dramatic], [conversational tone]
  Non-verbal:  [laughs], [laughs harder], [starts laughing], [wheezing], [chuckles],
               [snorts], [sighs], [exhales], [exhales sharply], [inhales deeply],
               [clears throat], [swallows], [gulps], [woo]
  Pacing:      [short pause], [long pause]

Examples:
  Input:    "Are you serious? I can't believe you did that!"
  Enhanced: "[appalled] Are you serious? [sighs] I can't believe you did that!"

  Input:    "I guess you're right. It's just... difficult."
  Enhanced: "I guess you're right. [sighs] It's just... [thoughtful] difficult."

2) ACTIONABLE FEEDBACK LIST
Rules for feedback only:
- Start with what worked well — be specific and sincere.
- Only flag a problem if it meaningfully affects the impact of this speech for this context and audience.
- Do NOT mention pauses as an issue unless they genuinely hurt comprehension or delivery for this use case.
  Pauses are often natural — only raise them if there is a real, clear problem.
- Every item must be tied to the stated context and purpose, not generic coaching rules.
- IMPORTANT: Do NOT reference audio tags, bracket notation, or delivery markup in feedback (e.g. do NOT say "add a [happy] tag" or "use [excited]").
  Instead, describe the delivery quality in plain conversational language (e.g. "could be delivered with more warmth and optimism" or "the closing would land better with an upbeat, excited tone").
- Label each item by importance:
    [Critical] — must fix before next delivery, significantly impacts effectiveness
    [Suggested] — worthwhile improvement, moderate impact
    [Minor] — small polish, low priority
    [Positive] — something done well
- Skip low-value observations. Quality over quantity. 3–7 items maximum.

3) VOICE RECOMMENDATION
Choose the single best-matching ElevenLabs voice from the list below.

Match on ALL of the following criteria in priority order:
  1. Use case alignment — match the voice's ideal use_cases to the speech context (e.g. public speech, pitch, academic, casual, motivational).
  2. Energy level — match energy to the speech's required impact:
       high energy  → motivational speeches, rallies, sales pitches, youth audiences, social media
       medium energy → professional talks, TED-style, interviews, explainers, corporate
       low energy   → narration, meditation, bedtime, soft storytelling
  3. Tone and formality — formal/authoritative vs. warm/friendly vs. conversational.
  4. Accent preference — only if the context strongly implies a regional preference.

You MUST choose a voice from the following list only — do not invent voice IDs:

{voices}

In the rationale, briefly explain why the energy level and use case of this voice fits the speech context.

Return strict JSON only.

JSON schema:
{{
  "annotated_script": "string",
  "feedback": ["string"],
  "voice_recommendation": {{
    "voice_id": "string",
    "voice_name": "string",
    "rationale": "string"
  }}
}}
    "speed_summary": "string",
    "pause_summary": "string",
    "clarity_summary": "string",
    "emotional_summary": "string",
    "fluency_summary": "string",
    "context_alignment": "string"
  }}
}}

Each feedback string must start with one of: [Critical], [Suggested], [Minor], or [Positive].
Example: "[Positive] Your opening hook was confident and immediately relevant to the audience."
Example: "[Critical] The closing felt rushed — for a persuasive pitch, the call to action needs more weight."
Example: "[Minor] Slight over-use of filler words in the middle section; not distracting but worth cleaning up."

Context:
{context}

Original script:
{script}

Speech metadata JSON:
{metadata}
""".strip()


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
        prompt = QUESTION_PROMPT.format(context=context, draft=script)
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
        prompt = REFINE_PROMPT.format(
            context=context,
            script=script,
            answers=json.dumps(answers, ensure_ascii=False),
            instruction=instructions or "",
            history=json.dumps(previous_versions, ensure_ascii=False),
        )
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
        voices_list = "\n".join(
            f"  voice_id={v['voice_id']}  name={v['name']}  gender={v['gender']}  accent={v['accent']}  energy={v['energy']}  use_cases={v['use_cases']}  description={v['description']}"
            for v in KNOWN_VOICES
        )
        prompt = ANALYSIS_PROMPT.format(
            context=context,
            script=final_script,
            metadata=json.dumps(metadata_json, ensure_ascii=False, indent=2),
            voices=voices_list,
        )
        result = self._call_json(prompt, max_output_tokens=8192).payload

        annotated_script = str(result.get("annotated_script", "")).strip()
        if not annotated_script:
            raise GeminiServiceError("Gemini returned empty annotated script")

        feedback = result.get("feedback", [])
        if not isinstance(feedback, list):
            feedback = []
        feedback = [str(i).strip() for i in feedback if str(i).strip()]

        voice = result.get("voice_recommendation", {})
        if not isinstance(voice, dict):
            voice = {}

        diagnostics = result.get("diagnostics", {})
        if not isinstance(diagnostics, dict):
            diagnostics = {}

        return {
            "annotated_script": annotated_script,
            "feedback": feedback,
            "voice_recommendation": {
                "voice_id": str(voice.get("voice_id", "")).strip(),
                "voice_name": str(voice.get("voice_name", "General Purpose")).strip(),
                "rationale": str(voice.get("rationale", "Fit based on context and tone.")).strip(),
            },
            "diagnostics": diagnostics,
        }


