from django.contrib import admin
from .models import RecognizedFace

@admin.register(RecognizedFace)
class RecognizedFaceAdmin(admin.ModelAdmin):
    list_display = ('name', 'timestamp')
