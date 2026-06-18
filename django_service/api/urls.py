from __future__ import annotations

from django.urls import path

from .views import health_view, query_view


urlpatterns = [
    path("health/", health_view, name="health"),
    path("query/", query_view, name="query"),
]