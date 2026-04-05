# Copyright (C) 2026 SkyWay.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""FastAPI entry: OpenAI-compatible `/v1/*` and custom `/custom/v1/*`."""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import os
import secrets
import statistics
import time
import uuid
from collections import defaultdict, deque
from typing import Any, Iterator

logger = logging.getLogger(__name__)

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.middleware.gzip import GZipMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from core.inference import GgufEngine, get_engine
from security.exceptions import SecurityError, ValidationError
from security.path_guard import (
    PathAccessMode,
    is_protected_path,
    is_writable_path,
    validate_project_path,
    validate_write_path,
)
from security.shell_guards import validate_shell_command


def _api_title() -> str:
    return (os.getenv("API_TITLE") or "Inference API").strip() or "Inference API"


def _service_name() -> str:
    return (os.getenv("SERVICE_NAME") or "inference").strip() or "inference"


def _public_model_id() -> str:
    return (os.getenv("CORE_PUBLIC_MODEL_ID") or "local").strip() or "local"


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _api_key() -> str:
    return (os.getenv("CORE_API_KEY") or "").strip()


def _auth_enabled() -> bool:
    return bool(_api_key())


def _paranoid_mode() -> bool:
    """Enable paranoid security mode (timing attack protection)."""
    return (os.getenv("CORE_PARANOID_MODE") or "").strip().lower() in {"1", "true", "yes", "on"}


def _auth_delay_ms() -> int:
    """Delay in milliseconds on auth failure (paranoid mode)."""
    return _env_int("CORE_AUTH_DELAY_MS", 50)


def _extract_api_key(request: Request) -> str:
    header = (request.headers.get("x-api-key") or "").strip()
    if header:
        return header
    auth = (request.headers.get("authorization") or "").strip()
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return ""


def _validate_api_key(provided_key: str) -> bool:
    """Validate API key using constant-time comparison to prevent timing attacks."""
    expected = _api_key()
    if not expected:
        return False
    # Use secrets.compare_digest for timing attack protection
    return secrets.compare_digest(provided_key, expected)


def _client_id(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _is_protected_route(path: str) -> bool:
    protected_prefixes = (
        "/v1/chat/completions",
        "/v1/completions",
        "/custom/v1/completions",
        "/custom/v1/security/",
    )
    return any(path.startswith(prefix) for prefix in protected_prefixes)


class RateLimiter:
    def __init__(self) -> None:
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = asyncio.Lock()

    async def is_allowed(self, client_id: str) -> tuple[bool, int, int, int]:
        async with self._lock:
            rpm = _env_int("CORE_RATE_LIMIT_RPM", 60)
            burst = _env_int("CORE_RATE_LIMIT_BURST", 10)
            now = time.time()
            window_start = now - 60.0
            self._requests[client_id] = [ts for ts in self._requests[client_id] if ts > window_start]
            current_count = len(self._requests[client_id])
            burst_count = sum(1 for ts in self._requests[client_id] if ts > now - 1.0)
            if burst_count >= burst:
                return False, 0, 1000, rpm
            if current_count >= rpm:
                retry_after_ms = int((self._requests[client_id][0] - window_start) * 1000)
                return False, 0, max(1000, retry_after_ms), rpm
            self._requests[client_id].append(now)
            return True, rpm - current_count - 1, 0, rpm

    async def cleanup_stale_clients(self) -> int:
        """Remove clients with no requests in the last hour. Returns count of removed clients."""
        async with self._lock:
            now = time.time()
            hour_ago = now - 3600.0
            stale_clients = [
                client_id
                for client_id, timestamps in self._requests.items()
                if not timestamps or max(timestamps) < hour_ago
            ]
            for client_id in stale_clients:
                del self._requests[client_id]
            return len(stale_clients)

    async def reset(self) -> None:
        async with self._lock:
            self._requests.clear()


_rate_limiter = RateLimiter()
_request_events: deque[tuple[float, int, float]] = deque(maxlen=4000)
_sorted_latencies: list[float] = []  # Maintained sorted for O(log n) percentile


def _record_request(status_code: int, latency_ms: float) -> None:
    _request_events.append((time.time(), int(status_code), float(latency_ms)))
    # Maintain sorted latencies for efficient percentile calculation
    bisect.insort(_sorted_latencies, float(latency_ms))


def _linear_percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return float(sorted_vals[lo]) * (1.0 - frac) + float(sorted_vals[hi]) * frac


def _metrics_snapshot() -> dict[str, Any]:
    cutoff = time.time() - 600.0
    recent = [event for event in _request_events if event[0] >= cutoff]
    status_buckets: dict[str, int] = defaultdict(int)
    # Use pre-sorted latencies for efficient percentile calculation
    # For recent-only latencies, we need to filter but can use the sorted list
    latencies = [event[2] for event in recent]
    latencies.sort()  # Sort only the recent subset (smaller than full list)
    for _, status_code, _ in recent:
        family = f"{status_code // 100}xx"
        status_buckets[family] += 1
    total = len(recent)
    error_count = sum(1 for _, status_code, _ in recent if status_code >= 400)
    return {
        "requests_10min": total,
        "errors_10min": error_count,
        "error_rate_10min": round(error_count / total, 4) if total else 0.0,
        "latency_p50_ms": round(float(statistics.median(latencies)), 2) if latencies else 0.0,
        "latency_p95_ms": round(_linear_percentile(latencies, 0.95), 2) if latencies else 0.0,
        "status_families_10min": dict(status_buckets),
        "auth_enabled": _auth_enabled(),
        "rate_limit_rpm": _env_int("CORE_RATE_LIMIT_RPM", 60),
        "rate_limit_burst": _env_int("CORE_RATE_LIMIT_BURST", 10),
    }


async def _reset_runtime_state() -> None:
    _request_events.clear()
    _sorted_latencies.clear()
    await _rate_limiter.reset()


async def _rate_limiter_cleanup_task() -> None:
    """Background task to cleanup stale rate limiter clients every hour."""
    while True:
        await asyncio.sleep(3600.0)  # Run every hour
        try:
            removed = await _rate_limiter.cleanup_stale_clients()
            if removed > 0:
                logger.info("RateLimiter cleanup: removed %d stale clients", removed)
        except Exception as exc:
            logger.warning("RateLimiter cleanup error: %s", exc)


app = FastAPI(
    title=_api_title(),
    description="Local GGUF inference with OpenAI-compatible HTTP APIs and policy guards",
    version="0.2.0",
)
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=[
        h.strip()
        for h in (os.getenv("CORE_ALLOWED_HOSTS") or "127.0.0.1,localhost,testserver").split(",")
        if h.strip()
    ],
)
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.on_event("startup")
async def _startup_background_tasks() -> None:
    """Start background tasks on app startup."""
    asyncio.create_task(_rate_limiter_cleanup_task())


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("CRITICAL ERROR: %s", exc)
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=500,
        content={"error": {"message": "Internal server error", "type": "server_error", "request_id": request_id}},
        headers={"X-Request-Id": request_id},
    )


@app.middleware("http")
async def apply_runtime_policies(request: Request, call_next):
    request_id = f"req_{uuid.uuid4().hex[:16]}"
    request.state.request_id = request_id
    started = time.perf_counter()
    path = request.url.path

    if _auth_enabled() and _is_protected_route(path):
        provided_key = _extract_api_key(request)
        if not _validate_api_key(provided_key):
            # Paranoid mode: add delay to prevent timing attacks
            if _paranoid_mode():
                await asyncio.sleep(_auth_delay_ms() / 1000.0)
            latency_ms = (time.perf_counter() - started) * 1000.0
            _record_request(401, latency_ms)
            return JSONResponse(
                status_code=401,
                content={"error": {"message": "Invalid or missing API key", "type": "auth_error"}},
                headers={"X-Request-Id": request_id},
            )

    if request.method != "OPTIONS" and _is_protected_route(path):
        allowed, remaining, retry_after_ms, rpm = await _rate_limiter.is_allowed(_client_id(request))
        if not allowed:
            latency_ms = (time.perf_counter() - started) * 1000.0
            _record_request(429, latency_ms)
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "Too many requests",
                        "type": "rate_limit_error",
                        "retry_after_ms": retry_after_ms,
                    }
                },
                headers={
                    "Retry-After": str(max(1, retry_after_ms // 1000)),
                    "X-RateLimit-Limit": str(rpm),
                    "X-RateLimit-Remaining": "0",
                    "X-Request-Id": request_id,
                },
            )

    response = await call_next(request)
    latency_ms = (time.perf_counter() - started) * 1000.0
    _record_request(response.status_code, latency_ms)
    response.headers["X-Request-Id"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    if _is_protected_route(path):
        response.headers["X-RateLimit-Limit"] = str(_env_int("CORE_RATE_LIMIT_RPM", 60))
    return response


class ChatMessage(BaseModel):
    role: str
    content: str | None = ""


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    max_tokens: int | None = Field(default=256, ge=1, le=128000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str = Field(..., min_length=1)
    max_tokens: int | None = Field(default=256, ge=1, le=128000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False
    system_prompt: str | None = None


class CustomCompletionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    system_prompt: str | None = None
    max_tokens: int = Field(default=256, ge=1, le=128000)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    stream: bool = False


class ShellValidationRequest(BaseModel):
    command: str = Field(..., min_length=1)


class PathValidationRequest(BaseModel):
    path: str = Field(..., min_length=1)
    mode: str = Field(default="read", pattern="^(read|write)$")
    must_exist: bool = False


def _kernel() -> GgufEngine:
    return get_engine()


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    health = _kernel().health_check()
    health["service"] = _service_name()
    health["model"] = _public_model_id()
    return health


@app.get("/readyz")
def readyz() -> JSONResponse:
    health = healthz()
    if health.get("status") != "READY":
        return JSONResponse(status_code=503, content=health)
    return JSONResponse(content=health)


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    engine_metrics = _kernel().metrics()
    return {
        "service": _service_name(),
        "model": _public_model_id(),
        "http": _metrics_snapshot(),
        "engine": engine_metrics,
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    mid = _public_model_id()
    owner = _service_name()
    return {
        "object": "list",
        "data": [
            {
                "id": mid,
                "object": "model",
                "created": int(time.time()),
                "owned_by": owner,
            }
        ],
    }


@app.post("/v1/chat/completions")
def chat_completions(body: ChatCompletionRequest) -> Any:
    k = _kernel()
    messages = [m.model_dump() for m in body.messages]
    max_tok = int(body.max_tokens or 256)
    temp = body.temperature

    if body.stream:

        def gen() -> Iterator[bytes]:
            completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            model = body.model or _public_model_id()
            try:
                stream = k.create_chat_completion(
                    messages,
                    max_tokens=max_tok,
                    temperature=temp,
                    stream=True,
                )
                for chunk in stream:
                    if isinstance(chunk, dict):
                        chunk.setdefault("id", completion_id)
                        chunk.setdefault("created", created)
                        chunk.setdefault("model", model)
                        # Ensure finish_reason is proxied from engine
                        for choice in chunk.get("choices", []):
                            if "finish_reason" not in choice:
                                choice["finish_reason"] = None
                    yield f"data: {json.dumps(chunk, ensure_ascii=False, default=str)}\n\n".encode(
                        "utf-8"
                    )
                yield b"data: [DONE]\n\n"
            except Exception as exc:
                err = {"error": {"message": str(exc), "type": "server_error"}}
                yield f"data: {json.dumps(err)}\n\n".encode("utf-8")

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        result = k.create_chat_completion(
            messages,
            max_tokens=max_tok,
            temperature=temp,
            stream=False,
        )
        if isinstance(result, dict):
            result.setdefault("id", f"chatcmpl-{uuid.uuid4().hex[:24]}")
            result.setdefault("created", int(time.time()))
            result.setdefault("model", body.model or _public_model_id())
            # Ensure usage field exists for OpenAI compatibility (LangChain, etc.)
            if "usage" not in result:
                # Estimate usage from engine metrics if available
                result["usage"] = {
                    "prompt_tokens": 0,
                    "completion_tokens": k._last_generation_tokens if hasattr(k, "_last_generation_tokens") else 0,
                    "total_tokens": k._last_generation_tokens if hasattr(k, "_last_generation_tokens") else 0,
                }
            # Ensure finish_reason is set in choices
            for choice in result.get("choices", []):
                if "finish_reason" not in choice or choice.get("finish_reason") is None:
                    choice["finish_reason"] = "stop"
        return JSONResponse(content=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/v1/completions")
def completions(body: CompletionRequest) -> Any:
    k = _kernel()
    max_tok = int(body.max_tokens or 256)
    temp = body.temperature

    if body.stream:

        def gen() -> Iterator[bytes]:
            completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            model = body.model or _public_model_id()
            try:
                for piece in k.stream_generate(
                    body.prompt,
                    system_prompt=body.system_prompt,
                    max_tokens=max_tok,
                    temperature=temp,
                ):
                    payload = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"text": piece, "index": 0, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            except Exception as exc:
                err = {"error": {"message": str(exc), "type": "server_error"}}
                yield f"data: {json.dumps(err)}\n\n".encode("utf-8")

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        text = k.generate(
            body.prompt,
            system_prompt=body.system_prompt,
            max_tokens=max_tok,
            temperature=temp,
        )
        return {
            "id": f"cmpl-{uuid.uuid4().hex[:24]}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": body.model or _public_model_id(),
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except MemoryError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/custom/v1/completions")
def custom_completions(body: CustomCompletionRequest) -> Any:
    k = _kernel()
    max_tok = body.max_tokens
    temp = body.temperature

    if body.stream:

        def gen() -> Iterator[bytes]:
            try:
                for piece in k.stream_generate(
                    body.prompt,
                    system_prompt=body.system_prompt,
                    max_tokens=max_tok,
                    temperature=temp,
                ):
                    payload = {"object": "text_completion.chunk", "text": piece}
                    yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                yield b"data: [DONE]\n\n"
            except Exception as exc:
                yield f"data: {json.dumps({'error': str(exc)})}\n\n".encode("utf-8")

        return StreamingResponse(gen(), media_type="text/event-stream")

    try:
        text = k.generate(
            body.prompt,
            system_prompt=body.system_prompt,
            max_tokens=max_tok,
            temperature=temp,
        )
        return {
            "object": "text_completion",
            "model": _public_model_id(),
            "text": text,
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/custom/v1/security/shell/validate")
def validate_shell(body: ShellValidationRequest) -> dict[str, Any]:
    result = validate_shell_command(body.command)
    return {
        "command": body.command,
        "allowed": result.allowed,
        "risk": result.risk.value,
        "reason": result.reason,
        "requires_confirmation": result.requires_confirmation,
        "blocked_patterns": result.blocked_patterns,
        "suggested_fix": result.suggested_fix,
    }


@app.post("/custom/v1/security/path/validate")
def validate_path_access(body: PathValidationRequest) -> dict[str, Any]:
    try:
        if body.mode == "write":
            resolved = validate_write_path(body.path)
            access_mode = PathAccessMode.WRITE_ALLOWED
        else:
            resolved = validate_project_path(body.path, must_exist=body.must_exist)
            access_mode = PathAccessMode.PROTECTED if is_protected_path(resolved) else (
                PathAccessMode.WRITE_ALLOWED if is_writable_path(resolved) else PathAccessMode.READ_ONLY
            )
        return {
            "allowed": True,
            "path": str(resolved),
            "mode": body.mode,
            "access_mode": access_mode.value,
            "writable": is_writable_path(resolved),
            "protected": is_protected_path(resolved),
        }
    except SecurityError as exc:
        return {
            "allowed": False,
            "path": body.path,
            "mode": body.mode,
            "reason": str(exc),
            "error_type": "security_error",
        }
    except ValidationError as exc:
        return {
            "allowed": False,
            "path": body.path,
            "mode": body.mode,
            "reason": str(exc),
            "error_type": "validation_error",
            "field": exc.field,
        }
