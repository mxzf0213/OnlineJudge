from django.conf.urls import url

from ..views.oj import SFSubmissionAPI

urlpatterns = [
    url(r"^sf_submission/?$", SFSubmissionAPI.as_view(), name="sf_submission_api")
]
