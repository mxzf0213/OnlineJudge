import dramatiq

from account.models import User
from submission.models import Submission
from judge.dispatcher import JudgeDispatcher
from utils.shortcuts import DRAMATIQ_WORKER_ARGS


@dramatiq.actor(**DRAMATIQ_WORKER_ARGS())
def judge_task(submission_id=None, problem_id=None, user_id=None):
    if user_id:
        JudgeDispatcher(None, None, user_id).self_test()
    else:
        uid = Submission.objects.get(id=submission_id).user_id
        if User.objects.get(id=uid).is_disabled:
            return
        JudgeDispatcher(submission_id, problem_id, None).judge()
