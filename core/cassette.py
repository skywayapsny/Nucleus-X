# Copyright (C) 2026 SkyWay.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pluggable inference backend ("cassette") protocol."""

from __future__ import annotations

from typing import Any, Iterator, Protocol, runtime_checkable


@runtime_checkable
class Cassette(Protocol):
    """Any backend that can run text generation (local GGUF, remote API, mock)."""

    def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        context: list[dict[str, str]] | None = None,
        max_tokens: int = 256,
        temperature: float | None = None,
    ) -> str: ...

    def stream_generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        context: list[dict[str, str]] | None = None,
        max_tokens: int = 256,
        temperature: float | None = None,
    ) -> Iterator[str]: ...

    def health_check(self) -> dict[str, Any]: ...

    def create_chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int = 256,
        temperature: float | None = None,
        stream: bool = False,
    ) -> dict[str, Any] | Iterator[dict[str, Any]]:
        """OpenAI-shaped chat completion; default impl may build a plain prompt."""
        ...
