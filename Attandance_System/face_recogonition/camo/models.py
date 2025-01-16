# models.py
from django.db import models

class RecognizedFace(models.Model):
    name = models.CharField(max_length=100)
    timestamp = models.DateTimeField()
    
    def __str__(self):
        return f"{self.name} - {self.timestamp}"
