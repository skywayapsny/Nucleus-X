# Copyright (C) 2026 SkyWay.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Exceptions for the security layer."""

from __future__ import annotations


class CoreError(Exception):
    """Base error."""

    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code or "CORE_ERROR"


class ValidationError(CoreError):
    """Validation failures (paths, inputs)."""

    def __init__(self, message: str, field: str | None = None) -> None:
        super().__init__(message, code="VALIDATION_ERROR")
        self.field = field


class SecurityError(CoreError):
    """Security policy violations."""

    def __init__(self, message: str) -> None:
        super().__init__(message, code="SECURITY_ERROR")
