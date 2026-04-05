# Copyright (C) 2026 SkyWay.
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import asyncio
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from adapter import main as main_module


class FakeEngine:
    def health_check(self) -> dict[str, object]:
        return {
            "status": "READY",
            "model_path": "C:/models/demo.gguf",
            "loaded": False,
            "available_ram_mb": 4096.0,
            "min_required_ram_mb": 500,
            "model_hash": "abc123",
            "last_error": None,
        }

    def metrics(self) -> dict[str, object]:
        return {
            "process_rss_mb": 32.0,
            "tokens_per_second": 42.5,
            "n_threads": 4,
            "n_batch": 256,
            "n_ctx": 2048,
            "n_gpu_layers": 0,
            "model_path": "C:/models/demo.gguf",
        }

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        context: list[dict[str, str]] | None = None,
        max_tokens: int = 256,
        temperature: float | None = None,
    ) -> str:
        return f"generated:{prompt}"

    def stream_generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        context: list[dict[str, str]] | None = None,
        max_tokens: int = 256,
        temperature: float | None = None,
    ):
        yield "hello"
        yield " world"

    def create_chat_completion(
        self,
        messages: list[dict[str, object]],
        *,
        max_tokens: int = 256,
        temperature: float | None = None,
        stream: bool = False,
    ):
        if stream:
            return iter(
                [
                    {
                        "object": "chat.completion.chunk",
                        "choices": [{"delta": {"content": "hello"}, "index": 0}],
                    }
                ]
            )
        return {
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
        }


class ApiTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.env_backup = {name: os.environ.get(name) for name in self._managed_env_names()}
        os.environ["CORE_API_KEY"] = "secret-key"
        os.environ["CORE_RATE_LIMIT_RPM"] = "10"
        os.environ["CORE_RATE_LIMIT_BURST"] = "10"
        asyncio.run(main_module._reset_runtime_state())
        self.get_engine_patcher = patch("adapter.main.get_engine", return_value=FakeEngine())
        self.get_engine_patcher.start()
        self.client = TestClient(main_module.app)

    def tearDown(self) -> None:
        self.get_engine_patcher.stop()
        asyncio.run(main_module._reset_runtime_state())
        for name, value in self.env_backup.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value

    @staticmethod
    def _managed_env_names() -> list[str]:
        return [
            "CORE_API_KEY",
            "CORE_RATE_LIMIT_RPM",
            "CORE_RATE_LIMIT_BURST",
        ]

    def test_healthz_is_public(self) -> None:
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "READY")
        self.assertIn("service", payload)

    def test_chat_requires_api_key(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["error"]["type"], "auth_error")

    def test_chat_accepts_bearer_token(self) -> None:
        response = self.client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer secret-key"},
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("X-Request-Id", response.headers)
        self.assertIn("choices", response.json())

    def test_metrics_collect_request_stats(self) -> None:
        self.client.get("/healthz")
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("http", payload)
        self.assertGreaterEqual(payload["http"]["requests_10min"], 1)
        self.assertTrue(payload["http"]["auth_enabled"])

    def test_shell_validation_endpoint_surfaces_guard_logic(self) -> None:
        response = self.client.post(
            "/custom/v1/security/shell/validate",
            headers={"x-api-key": "secret-key"},
            json={"command": "git tag v1.2.3"},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["allowed"])
        self.assertEqual(payload["risk"], "dangerous")

    def test_path_validation_endpoint_returns_access_mode(self) -> None:
        response = self.client.post(
            "/custom/v1/security/path/validate",
            headers={"x-api-key": "secret-key"},
            json={"path": "README.md", "mode": "read", "must_exist": True},
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn(payload["access_mode"], {"read_only", "write_allowed", "protected"})


if __name__ == "__main__":
    unittest.main()
