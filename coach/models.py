from django.db import models


class Session(models.Model):
    context = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"Session {self.id}"


class ScriptVersion(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="script_versions")
    content = models.TextField()
    version_number = models.PositiveIntegerField()
    is_final = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-version_number"]
        unique_together = ("session", "version_number")

    def __str__(self) -> str:
        return f"Session {self.session_id} v{self.version_number}"


class Question(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="questions")
    question_text = models.TextField()
    answer_text = models.TextField(blank=True, default="")

    def __str__(self) -> str:
        return self.question_text[:80]


class Recording(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="recordings")
    audio_file = models.FileField(upload_to="recordings/")
    metadata_json = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class Analysis(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE, related_name="analyses")
    annotated_script = models.TextField()
    feedback = models.TextField()
    voice_recommendation = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
