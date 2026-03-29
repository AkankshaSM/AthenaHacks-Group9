from django.urls import path

from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("start-session/", views.start_session, name="start_session"),
    path("questions/", views.questions_page, name="questions_page"),
    path("questions/submit/", views.submit_answers, name="submit_answers"),
    path("editor/", views.editor_page, name="editor_page"),
    path("editor/save/", views.save_version, name="save_version"),
    path("recording/", views.recording_page, name="recording_page"),
    path("analysis/", views.analysis_page, name="analysis_page"),
    path("api/create-session/", views.create_session_api, name="create_session_api"),
    path("api/transcribe-audio/", views.transcribe_audio_api, name="transcribe_audio_api"),
    path("api/generate-questions/", views.generate_questions_api, name="generate_questions_api"),
    path("api/refine-script/", views.refine_script_api, name="refine_script_api"),
    path("api/upload-audio/", views.upload_audio_api, name="upload_audio_api"),
    path("api/analyze-speech/", views.analyze_speech_api, name="analyze_speech_api"),
    path("api/synthesize-audio/", views.synthesize_audio_api, name="synthesize_audio_api"),
]
