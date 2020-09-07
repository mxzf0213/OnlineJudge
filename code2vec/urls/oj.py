from django.conf.urls import url

from code2vec.app import code2vecAPI

urlpatterns = [
    url(r"^code2vec/?$", code2vecAPI.as_view(), name="code2vec_api"),
]
