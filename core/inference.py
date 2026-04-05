# Copyright (C) 2026 SkyWay.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""GGUF inference engine: lazy-loaded weights and Cassette protocol."""

from __future__ import annotations

import hashlib
import importlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Iterator

from core.cassette import Cassette

logger = logging.getLogger(__name__)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _parse_stop_sequences() -> list[str]:
    raw = (os.getenv("CORE_STOP_SEQUENCES") or "").strip()
    if raw:
        return [s.strip() for s in raw.split(",") if s.strip()]
    return ["<|redacted_im_end|>", "<|end|>", "\nSystem:", "\nuser:"]


def _load_logit_bias_map(root: Path) -> dict[int, float] | None:
    """Optional token logit bias from JSON. Off by default."""
    flag = (os.getenv("CORE_LOGIT_BIAS_ENABLED", "0") or "0").strip().lower()
    if flag not in {"1", "true", "yes", "y"}:
        return None
    override = (os.getenv("CORE_LOGIT_BIAS_PATH") or "").strip()
    if override:
        p = Path(override).expanduser()
        path = p if p.is_absolute() else root / p
    else:
        path = root / "data" / "config" / "logit_bias.json"
    if not path.exists():
        logger.info(
            "logit_bias: enabled but file missing: %s",
            path.as_posix(),
        )
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            logger.warning("logit_bias JSON: expected object, skip")
            return None
        strength_env = (os.getenv("CORE_LOGIT_BIAS_STRENGTH") or "").strip()
        bias_override: float | None = None
        if strength_env:
            try:
                bias_override = float(strength_env)
            except ValueError:
                logger.warning("CORE_LOGIT_BIAS_STRENGTH is not a float, using JSON values")
        out: dict[int, float] = {}
        for k, v in raw.items():
            tid = int(k)
            out[tid] = bias_override if bias_override is not None else float(v)
        if not out:
            return None
        logger.info("logit_bias: loaded %d entries from %s", len(out), path.name)
        return out
    except Exception as exc:
        logger.warning("logit_bias load failed: %s", exc)
        return None


def _optional_sha256_check(model_path: Path) -> None:
    expected = _normalize_sha256(os.getenv("CORE_EXPECTED_MODEL_SHA256", ""))
    if not expected:
        return
    digest = hashlib.sha256()
    with model_path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    got = digest.hexdigest()
    if got != expected:
        logger.warning(
            "[ModelCheck] SHA256 mismatch: expected=%s... got=%s...",
            expected[:16],
            got[:16],
        )


def _normalize_sha256(value: str) -> str:
    return "".join(ch for ch in (value or "").strip().lower() if ch in "0123456789abcdef")


def _resolve_model_path(explicit: str | None) -> Path:
    root = _repo_root()
    env_path = (os.getenv("CORE_MODEL_PATH") or "").strip()
    candidates: list[Path | None] = [
        Path(explicit).expanduser() if explicit else None,
        Path(env_path).expanduser() if env_path else None,
        root / "model.gguf",
        root / "models" / "model.gguf",
    ]
    existing = [p for p in candidates if p is not None and p.exists()]
    if existing:
        return existing[0]
    if explicit:
        return Path(explicit).expanduser()
    if env_path:
        return Path(env_path).expanduser()
    return root / "model.gguf"


def _import_runtime_class() -> Any:
    try:
        mod = importlib.import_module("llama_cpp")
    except ImportError as exc:
        raise RuntimeError(
            "GGUF Python runtime not installed. Install dependencies from requirements.txt"
        ) from exc
    return getattr(mod, "Llama")


class GgufEngine(Cassette):
    """Lazy-loaded GGUF weights; implements Cassette."""

    def __init__(self, model_path: str | None = None) -> None:
        root = _repo_root()
        self.model_path = _resolve_model_path(model_path)
        self._safetensors_hint = next(
            (
                p
                for p in (
                    root / "data" / "models" / "model.safetensors",
                    root / "models" / "model.safetensors",
                    root / "model.safetensors",
                )
                if p.exists()
            ),
            None,
        )
        self.min_ram_mb = int(os.getenv("CORE_MIN_RAM_MB", "500") or "500")
        self.n_threads = int(os.getenv("CORE_N_THREADS", "0") or "0")
        self.n_batch = int(os.getenv("CORE_N_BATCH", "256") or "256")
        self.n_ctx = int(os.getenv("CORE_N_CTX", "2048") or "2048")
        self.n_gpu_layers = int(os.getenv("CORE_N_GPU_LAYERS", "0") or "0")
        chat_fmt = (os.getenv("CORE_CHAT_FORMAT") or "").strip()
        self.chat_format: str | None = chat_fmt if chat_fmt else None

        self.temperature = float(os.getenv("CORE_TEMPERATURE", "0.8") or "0.8")
        self.top_p = float(os.getenv("CORE_TOP_P", "0.95") or "0.95")
        self.min_p = float(os.getenv("CORE_MIN_P", "0.05") or "0.05")
        self.frequency_penalty = float(os.getenv("CORE_FREQUENCY_PENALTY", "0.0") or "0.0")
        self.presence_penalty = float(os.getenv("CORE_PRESENCE_PENALTY", "0.0") or "0.0")
        self.repeat_penalty = float(os.getenv("CORE_REPEAT_PENALTY", "1.1") or "1.1")

        self._runtime: Any | None = None
        self._last_tokens_per_second: float = 0.0
        self._last_generation_tokens: int = 0
        self._last_generation_time_sec: float = 0.0
        self._last_error: str | None = None
        self._model_hash: str | None = None
        self._logit_bias: dict[int, float] | None = _load_logit_bias_map(root)
        self._stop = _parse_stop_sequences()

    def _available_ram_mb(self) -> float:
        try:
            import psutil

            return float(psutil.virtual_memory().available / (1024 * 1024))
        except Exception:
            return 0.0

    def _resolve_threads(self) -> int:
        if self.n_threads > 0:
            return self.n_threads
        cpu_count = os.cpu_count() or 2
        return max(1, min(8, cpu_count - 1))

    def _ensure_loaded(self) -> None:
        if self._runtime is not None:
            return
        if not self.model_path.exists():
            if self._safetensors_hint is not None:
                raise RuntimeError(
                    f"Found safetensors at {self._safetensors_hint}. "
                    "Set CORE_MODEL_PATH to a .gguf file."
                )
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if self._available_ram_mb() < float(self.min_ram_mb):
            raise MemoryError(f"Available RAM below {self.min_ram_mb}MB")

        Model = _import_runtime_class()

        kwargs: dict[str, Any] = {
            "model_path": str(self.model_path),
            "n_ctx": self.n_ctx,
            "n_threads": self._resolve_threads(),
            "n_batch": self.n_batch,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": False,
        }
        if self.chat_format:
            kwargs["chat_format"] = self.chat_format

        self._runtime = Model(**kwargs)
        self._model_hash = self._compute_model_hash()
        try:
            _optional_sha256_check(self.model_path)
        except Exception as exc:
            logger.warning("[ModelCheck] %s", exc)

    def _compute_model_hash(self) -> str:
        digest = hashlib.sha256()
        with self.model_path.open("rb") as f:
            chunk = f.read(1024 * 1024)
            if not chunk:
                raise ValueError("GGUF file is empty or corrupted")
            digest.update(chunk)
        return digest.hexdigest()[:16]

    def _build_prompt(
        self,
        prompt: str,
        system_prompt: str | None = None,
        context: list[dict[str, str]] | None = None,
    ) -> str:
        parts: list[str] = []
        if system_prompt:
            parts.append(f"System: {system_prompt.strip()}")
        if context:
            for item in context[-32:]:
                role = str(item.get("role", "user"))
                content = str(item.get("content", "")).strip()
                if content:
                    parts.append(f"{role}: {content}")
        parts.append(f"user: {prompt.strip()}")
        parts.append("assistant:")
        return "\n".join(parts)

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        context: list[dict[str, str]] | None = None,
        max_tokens: int = 256,
        temperature: float | None = None,
    ) -> str:
        started = time.perf_counter()
        self._ensure_loaded()
        assert self._runtime is not None
        temp = float(self.temperature if temperature is None else temperature)
        full_prompt = self._build_prompt(prompt, system_prompt=system_prompt, context=context)
        result = self._runtime(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=self.top_p,
            min_p=self.min_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repeat_penalty=self.repeat_penalty,
            logit_bias=self._logit_bias,
            stop=self._stop,
        )
        text = str(result["choices"][0]["text"])
        elapsed = max(0.001, time.perf_counter() - started)
        self._last_generation_time_sec = elapsed
        self._last_generation_tokens = max(1, len(text.split()))
        self._last_tokens_per_second = self._last_generation_tokens / elapsed
        return text.strip()

    def stream_generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        context: list[dict[str, str]] | None = None,
        max_tokens: int = 256,
        temperature: float | None = None,
    ) -> Iterator[str]:
        started = time.perf_counter()
        self._ensure_loaded()
        assert self._runtime is not None
        temp = float(self.temperature if temperature is None else temperature)
        full_prompt = self._build_prompt(prompt, system_prompt=system_prompt, context=context)
        tokens = 0
        for chunk in self._runtime(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=self.top_p,
            min_p=self.min_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repeat_penalty=self.repeat_penalty,
            logit_bias=self._logit_bias,
            stream=True,
            stop=self._stop,
        ):
            text = str(chunk.get("choices", [{}])[0].get("text", ""))
            if text:
                tokens += 1
                yield text
        elapsed = max(0.001, time.perf_counter() - started)
        self._last_generation_time_sec = elapsed
        self._last_generation_tokens = tokens
        self._last_tokens_per_second = tokens / elapsed if tokens > 0 else 0.0

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        self._ensure_loaded()
        assert self._runtime is not None
        temp = float(self.temperature if temperature is None else temperature)
        return self._runtime.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
            top_p=self.top_p,
            min_p=self.min_p,
            frequency_penalty=self.frequency_penalty,
            presence_penalty=self.presence_penalty,
            repeat_penalty=self.repeat_penalty,
            logit_bias=self._logit_bias,
            stream=stream,
        )

    def health_check(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "status": "READY",
            "model_path": str(self.model_path),
            "loaded": self._runtime is not None,
            "available_ram_mb": round(self._available_ram_mb(), 2),
            "min_required_ram_mb": self.min_ram_mb,
            "model_hash": self._model_hash,
            "last_error": self._last_error,
        }
        try:
            if not self.model_path.exists():
                raise FileNotFoundError("GGUF model file is missing")
            if self.model_path.stat().st_size < 1024:
                raise ValueError("GGUF model file appears corrupted")
            if self._available_ram_mb() < float(self.min_ram_mb):
                raise MemoryError(f"Available RAM below {self.min_ram_mb}MB")
        except Exception as exc:
            self._last_error = str(exc)
            result["status"] = "ERROR"
            result["last_error"] = self._last_error
        return result

    def metrics(self) -> dict[str, Any]:
        process_mem_mb = 0.0
        try:
            import psutil

            p = psutil.Process()
            process_mem_mb = float(p.memory_info().rss / (1024 * 1024))
        except Exception:
            pass
        return {
            "process_rss_mb": round(process_mem_mb, 2),
            "tokens_per_second": round(float(self._last_tokens_per_second), 3),
            "n_threads": self._resolve_threads(),
            "n_batch": self.n_batch,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "model_path": str(self.model_path),
        }


_engine_singleton: GgufEngine | None = None


def get_engine() -> GgufEngine:
    global _engine_singleton
    if _engine_singleton is None:
        _engine_singleton = GgufEngine()
    return _engine_singleton
