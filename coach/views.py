import json
import uuid
from typing import Any

from django.contrib import messages
from django.core.files.base import ContentFile
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import ensure_csrf_cookie
from django.views.decorators.http import require_GET, require_POST

from .forms import HomeInputForm, InstructionForm
from .models import Analysis, Question, Recording, ScriptVersion, Session
from .services.elevenlabs import ElevenLabsClient, ElevenLabsServiceError
from .services.gemini import GeminiClient, GeminiServiceError

SESSION_KEY = "coach_session_id"


def _active_session(request: HttpRequest) -> Session | None:
    session_id = request.session.get(SESSION_KEY)
    if not session_id:
        return None
    try:
        return Session.objects.get(id=session_id)
    except Session.DoesNotExist:
        return None


def _latest_script(session: Session) -> ScriptVersion | None:
    return session.script_versions.order_by("-version_number").first()


def _final_or_latest_script(session: Session) -> ScriptVersion | None:
    final_version = session.script_versions.filter(is_final=True).order_by("-version_number").first()
    if final_version:
        return final_version
    return _latest_script(session)


@require_GET
def home(request: HttpRequest) -> HttpResponse:
    return render(request, "coach/home.html", {"form": HomeInputForm()})


@require_POST
def start_session(request: HttpRequest) -> HttpResponse:
    form = HomeInputForm(request.POST)
    if not form.is_valid():
        return render(request, "coach/home.html", {"form": form}, status=400)

    context = form.cleaned_data["context"]
    script = form.cleaned_data["script"]

    session = Session.objects.create(context=context)
    ScriptVersion.objects.create(session=session, content=script, version_number=1, is_final=False)
    request.session[SESSION_KEY] = session.id

    client = GeminiClient()
    try:
        questions = client.generate_questions(context=context, script=script)
    except GeminiServiceError as exc:
        messages.warning(request, f"Question generation skipped due to AI service issue: {exc}")
        questions = []

    Question.objects.bulk_create([Question(session=session, question_text=q) for q in questions])

    if questions:
        return redirect("questions_page")
    return redirect("editor_page")


@require_GET
def questions_page(request: HttpRequest) -> HttpResponse:
    session = _active_session(request)
    if not session:
        return redirect("home")
    questions = session.questions.all()
    if not questions.exists():
        return redirect("editor_page")
    return render(request, "coach/questions.html", {"session_obj": session, "questions": questions})


@require_POST
def submit_answers(request: HttpRequest) -> HttpResponse:
    session = _active_session(request)
    if not session:
        return redirect("home")

    for question in session.questions.all():
        field_name = f"question_{question.id}"
        question.answer_text = request.POST.get(field_name, "").strip()
        question.save(update_fields=["answer_text"])

    return redirect("editor_page")


@require_GET
@ensure_csrf_cookie
def editor_page(request: HttpRequest) -> HttpResponse:
    session = _active_session(request)
    if not session:
        return redirect("home")

    latest = _latest_script(session)
    versions = session.script_versions.order_by("-version_number")
    answers = [
        {"question": q.question_text, "answer": q.answer_text}
        for q in session.questions.order_by("id")
    ]
    previous_versions = [v.content for v in versions]

    return render(
        request,
        "coach/editor.html",
        {
            "session_obj": session,
            "latest_script": latest.content if latest else "",
            "versions": versions,
            "instruction_form": InstructionForm(),
            "answers_json": json.dumps(answers),
            "previous_versions_json": json.dumps(previous_versions),
        },
    )


@require_POST
def save_version(request: HttpRequest) -> HttpResponse:
    session = _active_session(request)
    if not session:
        return redirect("home")

    content = request.POST.get("content", "").strip()
    if not content:
        messages.error(request, "Script content cannot be empty.")
        return redirect("editor_page")

    latest = _latest_script(session)
    next_version = 1 if latest is None else latest.version_number + 1

    is_final = request.POST.get("mark_final") == "on"
    if is_final:
        session.script_versions.update(is_final=False)

    ScriptVersion.objects.create(
        session=session,
        content=content,
        version_number=next_version,
        is_final=is_final,
    )
    messages.success(request, f"Saved version {next_version}.")
    return redirect("editor_page")


@require_GET
@ensure_csrf_cookie
def recording_page(request: HttpRequest) -> HttpResponse:
    session = _active_session(request)
    if not session:
        return redirect("home")

    script = _final_or_latest_script(session)
    if not script:
        messages.error(request, "No script found. Please create one first.")
        return redirect("editor_page")

    return render(
        request,
        "coach/recording.html",
        {
            "session_obj": session,
            "script": script.content,
            "context": session.context,
        },
    )


@require_GET
@ensure_csrf_cookie
def analysis_page(request: HttpRequest) -> HttpResponse:
    session = _active_session(request)
    if not session:
        return redirect("home")

    analysis = session.analyses.order_by("-created_at").first()
    recording = session.recordings.order_by("-created_at").first()

    voice_data: dict[str, Any] = {}
    feedback_list: list[str] = []

    if analysis:
        try:
            voice_data = json.loads(analysis.voice_recommendation)
        except json.JSONDecodeError:
            voice_data = {"voice_name": analysis.voice_recommendation, "rationale": ""}

        try:
            parsed_feedback = json.loads(analysis.feedback)
            if isinstance(parsed_feedback, list):
                feedback_list = [str(i) for i in parsed_feedback if str(i).strip()]
            else:
                feedback_list = (analysis.feedback or "").splitlines()
        except (json.JSONDecodeError, AttributeError):
            feedback_list = (analysis.feedback or "").splitlines()

    return render(
        request,
        "coach/analysis.html",
        {
            "session_obj": session,
            "analysis": analysis,
            "recording": recording,
            "voice_data": voice_data,
            "feedback_list": feedback_list,
        },
    )


@require_POST
def transcribe_audio_api(request: HttpRequest) -> JsonResponse:
    audio = request.FILES.get("audio")
    if not audio:
        return JsonResponse({"error": "Missing audio file"}, status=400)

    file_bytes = audio.read()
    mime_type = (audio.content_type or "audio/webm").split(";")[0].strip()

    try:
        transcript = GeminiClient().transcribe_audio(audio_bytes=file_bytes, mime_type=mime_type)
    except GeminiServiceError as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    return JsonResponse({"transcript": transcript})


@require_POST
def generate_questions_api(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    context = str(payload.get("context", "")).strip()
    script = str(payload.get("script", "")).strip()
    if not context or not script:
        return JsonResponse({"error": "Both context and script are required"}, status=400)

    try:
        questions = GeminiClient().generate_questions(context=context, script=script)
    except GeminiServiceError as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    return JsonResponse({"questions": questions[:7]})


@require_POST
def refine_script_api(request: HttpRequest) -> JsonResponse:
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    context = str(payload.get("context", "")).strip()
    script = str(payload.get("script", "")).strip()
    answers = payload.get("answers", [])
    previous_versions = payload.get("previous_versions", [])
    instructions = str(payload.get("instructions", "")).strip()

    if not context or not script:
        return JsonResponse({"error": "Both context and script are required"}, status=400)

    try:
        refined_script = GeminiClient().refine_script(
            context=context,
            script=script,
            answers=answers if isinstance(answers, list) else [],
            previous_versions=previous_versions if isinstance(previous_versions, list) else [],
            instructions=instructions,
        )
    except GeminiServiceError as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    return JsonResponse({"refined_script": refined_script})


@require_POST
def upload_audio_api(request: HttpRequest) -> JsonResponse:
    session = _active_session(request)
    if not session:
        return JsonResponse({"error": "No active session"}, status=400)

    audio = request.FILES.get("audio")
    if not audio:
        return JsonResponse({"error": "Missing audio file"}, status=400)

    file_bytes = audio.read()
    client = ElevenLabsClient()
    try:
        metadata_json = client.analyze_audio(
            file_bytes=file_bytes,
            filename=audio.name,
            content_type=audio.content_type,
        )
    except ElevenLabsServiceError as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    audio.seek(0)
    recording = Recording.objects.create(session=session, audio_file=audio, metadata_json=metadata_json)

    return JsonResponse(
        {
            "recording_id": recording.id,
            "metadata_json": metadata_json,
            "audio_url": recording.audio_file.url,
        }
    )


@require_POST
def analyze_speech_api(request: HttpRequest) -> JsonResponse:
    session = _active_session(request)
    if not session:
        return JsonResponse({"error": "No active session"}, status=400)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    script = str(payload.get("script", "")).strip()
    context = str(payload.get("context", "")).strip()
    metadata_json = payload.get("metadata_json")

    if not script or not context or not isinstance(metadata_json, dict):
        return JsonResponse({"error": "script, context, and metadata_json (object) are required"}, status=400)

    try:
        result = GeminiClient().analyze_speech(
            final_script=script,
            metadata_json=metadata_json,
            context=context,
        )
    except GeminiServiceError as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    feedback_payload = result.get("feedback", [])
    voice_text = json.dumps(result["voice_recommendation"], ensure_ascii=False)

    Analysis.objects.create(
        session=session,
        annotated_script=result["annotated_script"],
        feedback=json.dumps(feedback_payload, ensure_ascii=False),
        voice_recommendation=voice_text,
    )

    return JsonResponse(result)


@require_POST
def synthesize_audio_api(request: HttpRequest) -> JsonResponse:
    session = _active_session(request)
    if not session:
        return JsonResponse({"error": "No active session"}, status=400)

    try:
        payload = json.loads(request.body.decode("utf-8"))
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body"}, status=400)

    annotated_script = str(payload.get("annotated_script", "")).strip()
    if not annotated_script:
        return JsonResponse({"error": "annotated_script is required"}, status=400)

    voice_id = str(payload.get("voice_id", "")).strip() or None
    voice_name = str(payload.get("voice_name", "")).strip() or None

    if not voice_id:
        latest_analysis = session.analyses.order_by("-created_at").first()
        if latest_analysis:
            try:
                voice_data = json.loads(latest_analysis.voice_recommendation)
                voice_id = str(voice_data.get("voice_id", "")).strip() or None
                voice_name = str(voice_data.get("name", "")).strip() or voice_name
            except json.JSONDecodeError:
                voice_id = None

    elevenlabs_client = ElevenLabsClient()

    try:
        resolved_voice_id = elevenlabs_client.resolve_voice_id(
            preferred_voice_id=voice_id,
            preferred_name=voice_name,
        )
    except ElevenLabsServiceError as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    try:
        audio_bytes = elevenlabs_client.synthesize(text=annotated_script, voice_id=resolved_voice_id)
    except ElevenLabsServiceError as exc:
        return JsonResponse({"error": str(exc)}, status=502)

    filename = f"generated/{uuid.uuid4().hex}.mp3"
    recording = Recording(session=session, metadata_json={})
    recording.audio_file.save(filename, ContentFile(audio_bytes), save=False)
    recording.save()

    return JsonResponse({"audio_url": recording.audio_file.url, "voice_id": resolved_voice_id})
