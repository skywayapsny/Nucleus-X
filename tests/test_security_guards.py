# Copyright (C) 2026 SkyWay.
# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from security.exceptions import SecurityError
from security.path_guard import PathGuard
from security.shell_guards import CommandRisk, validate_shell_command


class PathGuardTestCase(unittest.TestCase):
    def test_validates_safe_paths_inside_base_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            safe_file = base_dir / "safe.txt"
            safe_file.write_text("ok", encoding="utf-8")
            guard = PathGuard(base_path=base_dir, allow_symlinks=False)
            resolved = guard.validate_path(safe_file, must_exist=True)
            self.assertEqual(resolved, safe_file.resolve())

    def test_blocks_directory_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = Path(tmp)
            guard = PathGuard(base_path=base_dir, allow_symlinks=False)
            with self.assertRaises(SecurityError):
                guard.validate_path(base_dir / ".." / "outside.txt")

    def test_sanitizes_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            guard = PathGuard(base_path=tmp, allow_symlinks=False)
            self.assertEqual(guard.sanitize_filename("file:name?.txt"), "filename.txt")
            self.assertEqual(guard.sanitize_filename("test\x00file.txt"), "testfile.txt")


class ShellGuardTestCase(unittest.TestCase):
    def test_blocks_git_tag_creation(self) -> None:
        result = validate_shell_command("git tag v1.2.3")
        self.assertFalse(result.allowed)
        self.assertEqual(result.risk, CommandRisk.DANGEROUS)

    def test_allows_read_only_git_log(self) -> None:
        result = validate_shell_command("git log --oneline -n 5")
        self.assertTrue(result.allowed)
        self.assertEqual(result.risk, CommandRisk.SAFE)

    def test_blocks_unc_paths(self) -> None:
        result = validate_shell_command(r"git show \\server\share\repo")
        self.assertFalse(result.allowed)
        self.assertEqual(result.risk, CommandRisk.DANGEROUS)


if __name__ == "__main__":
    unittest.main()
