# Copyright (C) 2026 SkyWay.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Path Guard for preventing directory traversal and unsafe path access.

Inspired by policy patterns used in command/file safety layers:
- writable zones and protected paths
- strict read/write path validation
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

from security.exceptions import SecurityError, ValidationError


class PathAccessMode(str, Enum):
    """Path access mode."""

    READ_ONLY = "read_only"
    WRITE_ALLOWED = "write_allowed"
    PROTECTED = "protected"


class PathGuard:
    """
    Prevents directory traversal and unsafe path usage.

    Ensures paths stay within the allowed base directory and can enforce:
    - writable zones
    - protected path patterns
    """

    def __init__(
        self,
        base_path: str | Path,
        allow_symlinks: bool = False,
        writable_zones: set[str] | None = None,
        protected_patterns: set[str] | None = None,
    ):
        """
        Initialize Path Guard.

        Args:
            base_path: Root directory that paths must stay inside
            allow_symlinks: Whether symlinks are allowed
            writable_zones: Directories where write operations are allowed (relative to base_path)
            protected_patterns: Glob patterns for protected files
        """
        self.base_path = Path(base_path).resolve()
        self.allow_symlinks = allow_symlinks

        self._writable_zones: set[Path] = set()
        if writable_zones:
            for zone in writable_zones:
                zone_path = self.base_path / zone
                self._writable_zones.add(zone_path)

        self._protected_patterns = protected_patterns or {
            ".env*",
            "*.key",
            "*.pem",
            "*secret*",
            "*credential*",
            "config.local.*",
            ".git/config",
            "id_rsa*",
            "*.p12",
            "*.pfx",
        }

    def validate_path(self, path: str | Path, must_exist: bool = False) -> Path:
        """
        Validate path safety.

        Args:
            path: Path to validate
            must_exist: Require target to exist

        Returns:
            Safe absolute path

        Raises:
            ValidationError: Invalid path format or existence issue
            SecurityError: Path escapes base_path or violates symlink policy
        """
        try:
            target_path = Path(path).resolve()

            if not self._is_safe_path(target_path):
                raise SecurityError(f"Path escapes allowed base directory: {path}")

            if not self.allow_symlinks and self._has_symlinks(target_path):
                raise SecurityError(f"Symlinks are not allowed: {path}")

            if must_exist and not target_path.exists():
                raise ValidationError(f"Path does not exist: {path}")

            return target_path

        except Exception as e:
            if isinstance(e, ValidationError | SecurityError):
                raise
            raise ValidationError(f"Path validation failed for {path}: {e}")

    def validate_write_path(self, path: str | Path) -> Path:
        """
        Strict validation for write operations.

        Checks:
        1. Path is inside base_path
        2. Path is inside writable_zones when zones are configured
        3. Path does not match protected patterns

        Args:
            path: Target path for writing

        Returns:
            Safe absolute path

        Raises:
            SecurityError: Path is outside writable zones or protected
        """
        target_path = self.validate_path(path)

        if self._is_protected_path(target_path):
            raise SecurityError(
                f"Write operation blocked for protected path: {path}. "
                "Path matches a protected pattern."
            )

        if not self._writable_zones:
            return target_path

        for zone in self._writable_zones:
            try:
                target_path.relative_to(zone)
                return target_path
            except ValueError:
                continue

        zones_str = ", ".join(str(z.relative_to(self.base_path)) for z in self._writable_zones)
        raise SecurityError(
            f"Write access is only allowed in: {zones_str}. "
            f"Requested path: {path}"
        )

    def get_access_mode(self, path: str | Path) -> PathAccessMode:
        """
        Determine effective access mode for a path.

        Args:
            path: Path to inspect

        Returns:
            PathAccessMode for the given path
        """
        try:
            target_path = self.validate_path(path)
        except (ValidationError, SecurityError):
            return PathAccessMode.PROTECTED

        if self._is_protected_path(target_path):
            return PathAccessMode.PROTECTED

        if self._writable_zones:
            for zone in self._writable_zones:
                try:
                    target_path.relative_to(zone)
                    return PathAccessMode.WRITE_ALLOWED
                except ValueError:
                    continue
            return PathAccessMode.READ_ONLY

        return PathAccessMode.WRITE_ALLOWED

    def is_writable(self, path: str | Path) -> bool:
        """Return True when writing to the path is allowed."""
        mode = self.get_access_mode(path)
        return mode == PathAccessMode.WRITE_ALLOWED

    def is_protected(self, path: str | Path) -> bool:
        """Return True when path is considered protected."""
        mode = self.get_access_mode(path)
        return mode == PathAccessMode.PROTECTED

    def _is_safe_path(self, path: Path) -> bool:
        """Return True when the path is inside base_path."""
        try:
            path.relative_to(self.base_path)
            return True
        except ValueError:
            return False

    def _has_symlinks(self, path: Path) -> bool:
        """Return True when the path or any parent is a symlink."""
        try:
            if path.is_symlink():
                return True

            return any(parent.is_symlink() for parent in path.parents)
        except (OSError, ValueError):
            return True

    def _is_protected_path(self, path: Path) -> bool:
        """
        Check whether path matches protected patterns.

        Args:
            path: Path to check

        Returns:
            True if path is protected
        """
        import fnmatch

        try:
            relative = path.relative_to(self.base_path)
            relative_str = str(relative).replace("\\", "/")
        except ValueError:
            return True

        filename = path.name.lower()

        for pattern in self._protected_patterns:
            if fnmatch.fnmatch(filename, pattern.lower()):
                return True
            if fnmatch.fnmatch(relative_str, pattern.lower()):
                return True

        return False

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename by removing dangerous characters and trimming length.
        """
        dangerous_chars = '<>:"/\\|?*\x00'
        sanitized = "".join(c for c in filename if c not in dangerous_chars)

        sanitized = "".join(c for c in sanitized if ord(c) >= 32)

        max_length = 255
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        sanitized = sanitized.strip(". ")

        if not sanitized:
            sanitized = "unnamed_file"

        return sanitized

    def is_safe_extension(self, filename: str, allowed_extensions: set | None = None) -> bool:
        """
        Check if a file extension is allowed.

        Args:
            filename: File name
            allowed_extensions: Allowed extension set (without dots)

        Returns:
            True when extension is allowed
        """
        if allowed_extensions is None:
            allowed_extensions = {"txt", "json", "py", "md", "log", "csv", "yaml", "yml"}

        ext = Path(filename).suffix.lower().lstrip(".")
        return ext in allowed_extensions

    def create_safe_path(self, *path_parts: str) -> Path:
        """
        Build a safe path from path parts.

        Args:
            path_parts: Path segments

        Returns:
            Safe absolute path
        """
        sanitized_parts = [self.sanitize_filename(part) for part in path_parts]

        safe_path = self.base_path.joinpath(*sanitized_parts)

        return self.validate_path(safe_path)


# Repository root (security/ is one level below root)
_project_root = Path(__file__).resolve().parent.parent

# Writable zones (relative to project root)
_DEFAULT_WRITABLE_ZONES = {
    "data",
    "data/scratch",
    "data/logs",
    "sandbox",
}

# External executable allowlist (outside project tree)
_EXTERNAL_EXECUTABLE_WHITELIST: set[str] = {
    "volumeid64.exe",
    "volumeid.exe",
    "cmd.exe",
    "powershell.exe",
    "pwsh.exe",
    "python.exe",
    "python3.exe",
}

# External directories allowed for read/execute access
_EXTERNAL_ALLOWED_DIRS: set[Path] = set()

def _init_external_dirs() -> None:
    """Initialize external allowlisted directories."""
    global _EXTERNAL_ALLOWED_DIRS

    downloads = Path.home() / "Downloads"
    if downloads.exists():
        _EXTERNAL_ALLOWED_DIRS.add(downloads)

    if os.name == "nt":
        system_root = os.environ.get("SYSTEMROOT", r"C:\Windows")
        system32 = Path(system_root) / "System32"
        if system32.exists():
            _EXTERNAL_ALLOWED_DIRS.add(system32)


_init_external_dirs()

project_guard = PathGuard(_project_root, writable_zones=_DEFAULT_WRITABLE_ZONES)
data_guard = PathGuard(_project_root / "data")
logs_guard = PathGuard(_project_root / "logs")
scratch_guard = PathGuard(_project_root / "data" / "scratch")


def validate_project_path(path: str | Path, must_exist: bool = False) -> Path:
    """Validate path inside project root."""
    return project_guard.validate_path(path, must_exist)


def validate_data_path(path: str | Path, must_exist: bool = False) -> Path:
    """Validate path inside the data directory."""
    return data_guard.validate_path(path, must_exist)


def validate_write_path(path: str | Path) -> Path:
    """
    Strict write-path validation.

    Checks writable zones and protected patterns.
    Use before any file write operation.

    Args:
        path: Path to write

    Returns:
        Safe path

    Raises:
        SecurityError: Path is outside writable zones or protected
    """
    return project_guard.validate_write_path(path)


def is_writable_path(path: str | Path) -> bool:
    """Return True if path is writable under active policy."""
    return project_guard.is_writable(path)


def is_protected_path(path: str | Path) -> bool:
    """Return True if path is protected under active policy."""
    return project_guard.is_protected(path)


def validate_external_path(path: str | Path, must_exist: bool = False) -> Path:
    """
    Validate external path (outside project root).

    Checks:
    1. Path is under allowed external directories (Downloads, System32)
    2. Or file name is in external executable allowlist

    Args:
        path: Path to validate
        must_exist: Require target to exist

    Returns:
        Absolute path

    Raises:
        SecurityError: External path is not allowed
    """
    try:
        target_path = Path(path).resolve()
    except Exception as e:
        raise ValidationError(f"Invalid path: {path}: {e}") from e

    if must_exist and not target_path.exists():
        raise ValidationError(f"Path does not exist: {path}")

    for allowed_dir in _EXTERNAL_ALLOWED_DIRS:
        try:
            target_path.relative_to(allowed_dir)
            return target_path
        except ValueError:
            continue

    filename = target_path.name.lower()
    if filename in {x.lower() for x in _EXTERNAL_EXECUTABLE_WHITELIST}:
        parent = target_path.parent
        for allowed_dir in _EXTERNAL_ALLOWED_DIRS:
            try:
                parent.relative_to(allowed_dir)
                return target_path
            except ValueError:
                continue

        return target_path

    raise SecurityError(
        f"External path not allowed: {path}. "
        f"Allowed directories: {[str(d) for d in _EXTERNAL_ALLOWED_DIRS]}"
    )


def is_external_executable_allowed(path: str | Path) -> bool:
    """Return True if external executable path passes policy checks."""
    try:
        validate_external_path(path)
        return True
    except (SecurityError, ValidationError):
        return False


def add_external_allowed_dir(path: str | Path) -> None:
    """
    Add an external directory to the allowlist.
    """
    global _EXTERNAL_ALLOWED_DIRS
    dir_path = Path(path).resolve()
    if dir_path.exists() and dir_path.is_dir():
        _EXTERNAL_ALLOWED_DIRS.add(dir_path)


def add_external_executable(name: str) -> None:
    """
    Add executable file name to external allowlist.

    Args:
        name: Executable name, for example "mytool.exe"
    """
    global _EXTERNAL_EXECUTABLE_WHITELIST
    _EXTERNAL_EXECUTABLE_WHITELIST.add(name.lower())


def safe_open_file(filepath: str | Path, mode: str = "r", encoding: str = "utf-8"):
    """
    Open file with project path validation.

    Args:
        filepath: File path
        mode: Open mode
        encoding: Text encoding

    Returns:
        Open file object

    Raises:
        ValidationError: Path is invalid
    """
    safe_path = project_guard.validate_path(filepath, must_exist=("r" in mode))
    return safe_path.open(mode, encoding=encoding)

