from django.db import models
from account.models import User
from contest.models import Contest

class Class(models.Model):
    name = models.TextField(unique=True)
    teachers = models.ManyToManyField(User)
    students = models.ManyToManyField(User)
    contests = models.ManyToManyField(Contest)

    class Meta:
        db_table = "class"
