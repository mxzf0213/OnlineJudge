from .models import SFSubmission
from utils.api import serializers
from utils.serializers import LanguageNameChoiceField


class CreateSFSubmissionSerializer(serializers.Serializer):
    language = LanguageNameChoiceField()
    code = serializers.CharField(max_length=1024 * 1024)
    test_case = serializers


class SFSubmissionModelSerializer(serializers.ModelSerializer):

    class Meta:
        model = SFSubmission
        fields = "__all__"
