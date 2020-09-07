from utils.api import APIView, validate_serializer
from account.models import User
from contest.models import Contest

class ClassAPI(APIView):
    def common_checks(self, request):
        data = request.data
        teachers = data.pop('teachers')
        students = data.pop('students')
        contests = data.pop('contests')

        for item in teachers:
            try:
                teacher = User.objects.get(username = item)
            except User.DoesNotExist:
                return "Teacher {:s} does not exist!".format(item)

        for item in students:
            try:
                student = User.objects.get(username = item)
            except User.DoesNotExist:
                return "Student {:s} does not exist!".format(item)

        for item in contests:
            try:
                contest = Contest.objects.get()
