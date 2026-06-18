from __future__ import annotations

import json
from typing import Any, Dict, Optional

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods

from demo import ExtensibleFusionQueryManager


_MANAGER: Optional[ExtensibleFusionQueryManager] = None


def get_manager() -> ExtensibleFusionQueryManager:
    global _MANAGER
    if _MANAGER is None:
        _MANAGER = ExtensibleFusionQueryManager()
    return _MANAGER


def _with_cors(response: JsonResponse) -> JsonResponse:
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _json_response(status_code: int, payload: Dict[str, Any]) -> JsonResponse:
    return _with_cors(JsonResponse(payload, status=status_code, json_dumps_params={"ensure_ascii": False}))


def _options_response() -> HttpResponse:
    response = HttpResponse(status=204)
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def _extract_query_text(request: HttpRequest) -> str:
    if request.method == "GET":
        return request.GET.get("query", "").strip()

    raw_body = request.body.decode("utf-8", errors="ignore").strip()
    if not raw_body:
        return ""

    try:
        payload = json.loads(raw_body)
    except json.JSONDecodeError:
        return ""

    return str(payload.get("query", "")).strip()


@require_http_methods(["GET", "OPTIONS"])
def health_view(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_response()
    return _json_response(200, {"ok": True, "message": "service running"})


@require_http_methods(["GET", "POST", "OPTIONS"])
def query_view(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_response()

    query_text = _extract_query_text(request)
    if not query_text:
        return _json_response(400, {"ok": False, "error": "query 参数不能为空"})

    result = get_manager().query_detail(query_text)
    return _json_response(200, {"ok": True, **result})