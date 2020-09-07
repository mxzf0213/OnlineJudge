from django.conf.urls import url

from DKT.app import DktAPI

urlpatterns = [
    url(r"^dkt/?$", DktAPI.as_view(), name="dkt_api"),
]
