from django.contrib import admin

from .models import Analysis, Question, Recording, ScriptVersion, Session


@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ("id", "created_at")
    search_fields = ("context",)


@admin.register(ScriptVersion)
class ScriptVersionAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "version_number", "is_final", "created_at")
    list_filter = ("is_final",)


@admin.register(Question)
class QuestionAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "question_text")


@admin.register(Recording)
class RecordingAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "created_at")


@admin.register(Analysis)
class AnalysisAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "created_at")
