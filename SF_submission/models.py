from django.db import models

from utils.models import JSONField

from utils.shortcuts import rand_str


class JudgeStatus:
    COMPILE_ERROR = -2
    WRONG_ANSWER = -1
    ACCEPTED = 0
    CPU_TIME_LIMIT_EXCEEDED = 1
    REAL_TIME_LIMIT_EXCEEDED = 2
    MEMORY_LIMIT_EXCEEDED = 3
    RUNTIME_ERROR = 4
    SYSTEM_ERROR = 5
    PENDING = 6
    JUDGING = 7
    PARTIALLY_ACCEPTED = 8


class SFSubmission(models.Model):
    user_id = models.IntegerField(primary_key=True, db_index=True)
    create_time = models.DateTimeField(auto_now_add=True)
    code = models.TextField(default='')
    test_case = JSONField(default=dict)
    result = models.IntegerField(db_index=True, default=JudgeStatus.PENDING)
    output = JSONField(default=dict)
    # 从JudgeServer返回的判题详情
    statistic_info = JSONField(default=dict)
    info = JSONField(default=dict)
    language = models.TextField()

    class Meta:
        db_table = "SFSubmission"
        ordering = ("-create_time",)

    def __str__(self):
        return self.id
