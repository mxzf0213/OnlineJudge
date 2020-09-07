from django.conf.urls import url

from ..views.oj import RecommendAPI

urlpatterns = [
    url(r"^recommend/?$", RecommendAPI.as_view(), name="recommend_api"),
]
