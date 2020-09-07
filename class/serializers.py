from utils.api import serializers

from .models import Class

class CreateOrEditClassSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=1024)
    teachers = serializers.ListField(child=serializers.CharField(max_length=32), allow_empty=False)
    students = serializers.ListField(child=serializers.CharField(max_length=32), allow_empty=False)
    contests = serializers.ListField(child=serializers.CharField(max_length=32), allow_empty=False)

class EditClassSerializer(CreateOrEditClassSerializer):
    id = serializers.IntegerField()

class ClassSerializer(serializers.ModelSerializer):
    class Meta:
        model = Class
        fields = "__all__"



