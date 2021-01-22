from account.decorators import login_required
from judge.dispatcher import JudgeDispatcher
from utils.api import APIView, validate_serializer
from ..models import SFSubmission, JudgeStatus
from ..serializers import CreateSFSubmissionSerializer
from judge.tasks import judge_task


class SFSubmissionAPI(APIView):
    @login_required
    def post(self, request):
        data = request.data
        sf_submission = None
        try:
            sf_submission = SFSubmission.objects.get(user_id=request.user.id)
        except:
            pass
        if not sf_submission:
            SFSubmission.objects.create(user_id=request.user.id,
                                        code=data["code"],
                                        language=data["language"],
                                        test_case=data["test_case"])
        else:
            sf_submission.code = data["code"]
            sf_submission.language = data["language"]
            sf_submission.test_case = data['test_case']
            sf_submission.result = JudgeStatus.PENDING
            sf_submission.save()

        # JudgeDispatcher(request.user.id).judge()
        judge_task.send(user_id=request.user.id)
        return self.success()

    @login_required
    def get(self, request):
        sf_submission = None
        try:
            sf_submission = SFSubmission.objects.get(user_id=request.user.id)
        except:
            pass
        if sf_submission:
            return self.success({"status": sf_submission.result, "output": sf_submission.output})
        else:
            return self.error()
