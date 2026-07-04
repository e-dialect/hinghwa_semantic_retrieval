from __future__ import annotations

import json
from typing import Any, Dict, Optional

from django.http import HttpRequest, HttpResponse, JsonResponse
from django.views.decorators.http import require_http_methods

from demo import ExtensibleFusionQueryManager
from src.data_loader import get_word_word_dto_by_id
from src.result_formatter import format_result


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


def _dto_to_api_payload(dto: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ok": True,
        "query_id": dto.get("id"),
        "count": 1,
        "results": [dto],
        "formatted": format_result([dto]),
    }


@require_http_methods(["GET", "OPTIONS"])
def health_view(request: HttpRequest) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_response()
    return _json_response(200, {"ok": True, "message": "service running"})


@require_http_methods(["GET", "OPTIONS"])
def query_by_id_view(request: HttpRequest, word_id: int) -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_response()

    dto = get_word_word_dto_by_id(word_id)
    if dto is None:
        return _json_response(404, {"ok": False, "error": f"未找到 id={word_id} 对应的词条"})

    return _json_response(200, _dto_to_api_payload(dto))


@require_http_methods(["GET", "POST", "OPTIONS"])
def query_view(request: HttpRequest, query: str = "") -> JsonResponse:
    if request.method == "OPTIONS":
        return _options_response()

    query_text = query.strip() or _extract_query_text(request)
    if not query_text:
        return _json_response(400, {"ok": False, "error": "query 参数或路径参数不能为空"})

    result = get_manager().query_detail(query_text)
    return _json_response(200, {"ok": True, **result})