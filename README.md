# AI Speech Coaching Platform (Django + Gemini + ElevenLabs)

Production-ready starter implementation for a two-phase AI speech coaching workflow:

1. Content Refinement (context validation, clarifying questions, iterative script improvements with version history)
2. Speech Recording & Analysis (audio upload, ElevenLabs metadata ingestion, Gemini coaching + annotation, synthesized playback)

## Project Structure

```text
AthenaHacks-Group9/
├── manage.py
├── requirements.txt
├── .env.example
├── README.md
├── speechcoach_project/
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── coach/
│   ├── __init__.py
│   ├── admin.py
│   ├── apps.py
│   ├── forms.py
│   ├── models.py
│   ├── urls.py
│   ├── views.py
│   ├── migrations/
│   │   ├── __init__.py
│   │   └── 0001_initial.py
│   └── services/
│       ├── __init__.py
│       ├── gemini.py
│       └── elevenlabs.py
├── templates/
│   ├── base.html
│   └── coach/
│       ├── home.html
│       ├── questions.html
│       ├── editor.html
│       ├── recording.html
│       └── analysis.html
├── static/
│   └── coach/
│       └── style.css
└── media/
    └── generated/
```

## Environment Variables

Copy `.env.example` to `.env` and set real keys.

Required:
- `GEMINI_API_KEY`
- `ELEVENLABS_API_KEY`

Optional:
- `GEMINI_MODEL` (default: `gemini-2.5-flash`)
- `ELEVENLABS_MODEL_ID` (default: `eleven_v3`)
- `ELEVENLABS_DEFAULT_VOICE_ID`
- `ELEVENLABS_ANALYSIS_ENDPOINT`

## Install & Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python manage.py migrate
python manage.py runserver
```

Open: `http://127.0.0.1:8000/`

## Data Model

- `Session`: stores context and lifecycle start time.
- `ScriptVersion`: stores all script iterations (`version_number`, `is_final`).
- `Question`: generated clarifying questions and user answers.
- `Recording`: uploaded/generated audio and ElevenLabs metadata JSON.
- `Analysis`: annotated script, coaching feedback, voice recommendation.

## API Endpoints

- `POST /api/generate-questions/`
  - input: `context`, `script`
  - output: `questions[]` (max 7)

- `POST /api/refine-script/`
  - input: `context`, `script`, `answers`, `previous_versions`, `instructions`
  - output: `refined_script`

- `POST /api/upload-audio/`
  - multipart form field: `audio`
  - output: `metadata_json`, `audio_url`

- `POST /api/analyze-speech/`
  - input: `script`, `metadata_json`, `context`
  - output: `annotated_script`, `feedback[]`, `voice_recommendation`

- `POST /api/synthesize-audio/`
  - input: `annotated_script`, optional `voice_id`
  - output: `audio_url`

## Gemini Prompt Examples

### 1) Question Generation / Context Validation

```text
You are a speech-coaching assistant.
Task: validate whether context is sufficient to refine a speech script.

Rules:
- If context is enough, return empty questions.
- If context is insufficient, generate between 1 and 7 highly relevant clarifying questions.
- Questions must focus on audience, tone, duration, goal, delivery style, and formality when missing.
- Do not exceed 7 questions.

Return ONLY valid JSON:
{
  "needs_clarification": true/false,
  "questions": ["..."]
}
```

### 2) Script Refinement

```text
You are an expert speech writing and coaching assistant.
Task: produce an improved script optimized for clarity, engagement, and delivery.

Constraints:
- Preserve user intent and factual meaning.
- Improve structure and flow.
- Keep language audience-appropriate based on context and Q&A.
- If instructions are provided, prioritize them.

Return ONLY valid JSON:
{
  "refined_script": "..."
}
```

### 3) Speech Analysis

```text
You are an advanced speech coach.
Analyze delivery using FULL ElevenLabs metadata JSON. Do not ignore any provided field.

Required dimensions:
- speaking speed
- pauses
- fluency
- clarity
- emotional tone
- alignment with context and intent

Output JSON only:
{
  "annotated_script": "...",
  "feedback": ["..."],
  "voice_recommendation": {
    "voice_id": "...",
    "name": "...",
    "reason": "..."
  }
}
```

## Notes for Production Hardening

- Add authentication/authorization if sessions are user-owned.
- Add API retry, circuit-breaker, and queue-based async processing for heavy analysis.
- Validate ElevenLabs metadata schema at ingestion to guarantee analysis consistency.
- Move from SQLite to PostgreSQL in production.
- Serve media/static through CDN/object storage.
