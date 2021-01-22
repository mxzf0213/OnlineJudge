from account.decorators import super_admin_required
from judge.dispatcher import JudgeDispatcher
from judge.tasks import judge_task
# from judge.dispatcher import JudgeDispatcher
from utils.api import APIView
from ..models import Submission


class SubmissionRejudgeAPI(APIView):
    @super_admin_required
    def get(self, request):
        id = request.GET.get("id")
        if not id:
            return self.error("Parameter error, id is required")
        try:
            submission = Submission.objects.select_related("problem").get(id=id, contest_id__isnull=True)
        except Submission.DoesNotExist:
            return self.error("Submission does not exists")
        submission.statistic_info = {}
        submission.save()
        print("go rejudge")
        JudgeDispatcher(submission.id, submission.problem.id).judge()
        # judge_task.send(submission.id, submission.problem.id)
        return self.success()
