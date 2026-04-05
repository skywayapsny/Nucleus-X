# Copyright (C) 2026 SkyWay.
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shell command policy guard."""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from enum import Enum


class CommandRisk(str, Enum):
    """Risk level for shell command execution."""

    SAFE = "safe"  # Read-only, no side effects
    LOW = "low"  # Minor side effects (e.g., creating dirs)
    MEDIUM = "medium"  # File modifications
    HIGH = "high"  # System changes, network
    DANGEROUS = "dangerous"  # Blocked


@dataclass
class ValidationResult:
    """Validation result for a command."""

    allowed: bool
    risk: CommandRisk
    reason: str | None = None
    sanitized_command: str | None = None
    requires_confirmation: bool = False
    blocked_patterns: list[str] = field(default_factory=list)
    suggested_fix: str | None = None


# Dangerous substitution and expansion patterns.

COMMAND_SUBSTITUTION_PATTERNS = [
    (r"<\(", "process substitution <()"),
    (r">\(", "process substitution >()"),
    (r"=\(", "Zsh process substitution =()"),
    (r"\$\(", "$() command substitution"),
    (r"\$\{", "${} parameter substitution"),
    (r"\$\[", "$[] legacy arithmetic expansion"),
    (r"~\[", "Zsh-style parameter expansion"),
]

# Zsh dangerous commands
ZSH_DANGEROUS_COMMANDS = {
    "zmodload", "emulate", "sysopen", "sysread", "syswrite", "sysseek",
    "zpty", "ztcp", "zsocket", "mapfile",
    "zf_rm", "zf_mv", "zf_ln", "zf_chmod", "zf_chown", "zf_mkdir", "zf_rmdir", "zf_chgrp",
}

# Dangerous bash builtins
DANGEROUS_BUILTINS = {
    "eval", "exec", "source", ".",  # Code execution
    "export", "unset", "readonly",  # Environment manipulation
    "alias", "unalias",  # Alias manipulation
    "trap", "kill", "pkill", "killall",  # Signal handling
    "chown", "chmod", "chgrp",  # Permission changes
    "shutdown", "reboot", "halt", "poweroff",  # System control
    "fdisk", "mkfs", "mount", "umount",  # Disk operations
    "iptables", "ip6tables", "ufw",  # Firewall
    "useradd", "usermod", "userdel", "passwd",  # User management
    "sudo", "su", "doas", "pkexec",  # Privilege escalation
}

# High-risk commands (network, package managers)
HIGH_RISK_COMMANDS = {
    "curl", "wget", "nc", "netcat", "ncat", "socat", "telnet",  # Network
    "ssh", "scp", "sftp", "rsync", "rsh", "rlogin",  # Remote access
    "apt", "apt-get", "aptitude", "yum", "dnf", "rpm", "pacman", "pip", "npm", "yarn", "cargo",  # Package managers
    "docker", "kubectl", "helm", "podman",  # Containers
    "git push", "git reset --hard", "git clean -fdx",  # Git destructive
}


# Command allowlist and flag constraints.

# Safe flags for common read-only commands
SAFE_FLAGS = {
    "ls": {"-l", "-la", "-lh", "-a", "-A", "-R", "-d", "-h", "-v", "--help", "--version"},
    "cat": {"-n", "-b", "-s", "-E", "-T", "-v", "--help", "--version"},
    "head": {"-n", "-c", "-q", "-v", "--help", "--version"},
    "tail": {"-n", "-c", "-f", "-F", "-q", "-v", "--help", "--version"},
    "grep": {"-E", "-F", "-G", "-P", "-i", "-v", "-n", "-l", "-L", "-c", "-w", "-x", "-r", "-R", "--include", "--exclude", "--help", "--version"},
    "find": {"-name", "-iname", "-type", "-size", "-mtime", "-atime", "-ctime", "-maxdepth", "-mindepth", "-print", "-print0", "--help", "--version"},
    "sort": {"-n", "-r", "-u", "-k", "-t", "-b", "-f", "-h", "--help", "--version"},
    "uniq": {"-c", "-d", "-u", "-i", "-f", "-s", "-w", "--help", "--version"},
    "wc": {"-l", "-w", "-c", "-m", "-L", "--help", "--version"},
    "diff": {"-u", "-U", "-c", "-r", "-N", "-p", "-q", "-s", "--help", "--version"},
    "echo": {"-n", "-e", "-E", "--help", "--version"},
    "pwd": {"--help", "--version"},
    "which": {"-a", "--help", "--version"},
    "type": {"-a", "-f", "-p", "-t", "--help"},
    "stat": {"-c", "-f", "-L", "-x", "--help", "--version"},
    "file": {"-b", "-i", "-L", "-z", "--help", "--version"},
    "tree": {"-L", "-d", "-f", "-i", "-p", "-s", "-h", "-u", "-g", "--help", "--version"},
    "du": {"-h", "-s", "-a", "-c", "-d", "--max-depth", "--help", "--version"},
    "df": {"-h", "-i", "-T", "-t", "-x", "--help", "--version"},
    "ps": {"-a", "-e", "-f", "-l", "-u", "-x", "-p", "--help", "--version"},
    "id": {"-u", "-g", "-G", "-n", "-r", "--help", "--version"},
    "date": {"-d", "-r", "-u", "+%", "--help", "--version"},
    "uname": {"-a", "-r", "-s", "-n", "-m", "-p", "-i", "-o", "--help", "--version"},
    "hostname": {"-f", "-s", "-i", "-I", "--help", "--version"},
    "uptime": {"-p", "-s", "--help", "--version"},
    "free": {"-h", "-b", "-k", "-m", "-g", "-t", "--help", "--version"},
    "whoami": {"--help", "--version"},
    "env": {"--help"},
    "printenv": {"--help"},
    "true": {"--help", "--version"},
    "false": {"--help", "--version"},
    "basename": {"-a", "-s", "--help", "--version"},
    "dirname": {"--help", "--version"},
    "realpath": {"-m", "-e", "-q", "--help", "--version"},
    "readlink": {"-f", "-e", "-m", "-n", "-v", "--help", "--version"},
    "tr": {"-c", "-C", "-d", "-s", "-t", "--help", "--version"},
    "cut": {"-b", "-c", "-d", "-f", "-s", "--complement", "--help", "--version"},
    "sed": {"-n", "-e", "-f", "--help", "--version"},  # Note: -i is NOT safe
    "awk": {"-F", "-f", "-v", "--help", "--version"},
    "xargs": {"-I", "-n", "-P", "-r", "-t", "--help", "--version"},  # Note: no -i/-e
}

# Read-only commands (safe to auto-allow)
READ_ONLY_COMMANDS = {
    "ls", "cat", "head", "tail", "grep", "find", "sort", "uniq", "wc",
    "diff", "echo", "pwd", "which", "type", "stat", "file", "tree",
    "du", "df", "ps", "id", "date", "uname", "hostname", "uptime",
    "free", "whoami", "env", "printenv", "true", "false",
    "basename", "dirname", "realpath", "readlink", "tr", "cut", "sed", "awk",
    "git", "gh", "docker", "kubectl",  # With restrictions
}

UNC_PATH_PATTERN = re.compile(r"(?:^|[=\s])(?:\\\\|//)[^\\/\s]+[\\/][^\\/\s]+", re.IGNORECASE)

GIT_FLAG_TYPES: dict[str, str] = {
    "-n": "number",
    "--max-count": "number",
    "--skip": "number",
    "--pretty": "string",
    "--format": "string",
    "--grep": "string",
    "--author": "string",
    "--committer": "string",
    "--since": "string",
    "--until": "string",
    "--date": "string",
    "--color": "string",
    "--word-diff": "string",
    "--name-only": "none",
    "--name-status": "none",
    "--stat": "none",
    "--cached": "none",
    "--staged": "none",
    "--patch": "none",
    "-p": "none",
    "--no-patch": "none",
    "--oneline": "none",
    "--decorate": "none",
    "--graph": "none",
    "--all": "none",
    "--branches": "none",
    "--remotes": "none",
    "--tags": "none",
    "--reverse": "none",
    "--follow": "none",
    "--no-merges": "none",
    "--merges": "none",
    "-s": "none",
    "--short": "none",
    "--porcelain": "none",
    "--ignored": "none",
    "--recursive": "none",
    "-r": "none",
    "-l": "none",
    "--list": "none",
    "--abbrev-ref": "none",
    "--is-inside-work-tree": "none",
    "--is-bare-repository": "none",
    "--show-toplevel": "none",
    "--show-prefix": "none",
    "--show-cdup": "none",
    "--count": "none",
    "--left-right": "none",
    "--ancestry-path": "none",
    "--cherry-pick": "none",
    "--cherry-mark": "none",
    "--first-parent": "none",
    "--topo-order": "none",
    "--date-order": "none",
    "--objects": "none",
    "--contains": "string",
    "--contains=": "string",
    "--merged": "string",
    "--merged=": "string",
    "--no-merged": "string",
    "--no-merged=": "string",
    "--points-at": "string",
    "--points-at=": "string",
    "--sort": "string",
    "--sort=": "string",
    "--format=": "string",
    "-e": "none",
    "--stdin": "none",
    "--batch": "none",
    "--batch-check": "none",
    "--batch-command": "none",
    "--batch-all-objects": "none",
}

GIT_READ_ONLY_SUBCOMMANDS = {
    "diff",
    "log",
    "show",
    "status",
    "blame",
    "ls-files",
    "rev-parse",
    "rev-list",
    "grep",
    "reflog",
    "ls-remote",
    "merge-base",
    "describe",
    "cat-file",
    "for-each-ref",
    "worktree",
    "stash",
    "tag",
    "branch",
}

GH_READ_ONLY_SUBCOMMANDS = {
    "repo",
    "pr",
    "issue",
    "run",
    "release",
    "search",
    "api",
}

GIT_RESTRICTED_REFS = {"expire", "delete", "exists"}


class ShellGuard:
    """Shell command validator."""

    def __init__(
        self,
        allowed_commands: set[str] | None = None,
        blocked_commands: set[str] | None = None,
        require_confirmation_for_high_risk: bool = True,
    ):
        self.allowed_commands = allowed_commands or READ_ONLY_COMMANDS
        self.blocked_commands = blocked_commands or DANGEROUS_BUILTINS | ZSH_DANGEROUS_COMMANDS
        self.require_confirmation_for_high_risk = require_confirmation_for_high_risk

    def validate(self, command: str) -> ValidationResult:
        """
        Validate a shell command.

        Returns:
            ValidationResult with decision and reason
        """
        # Parse command
        base_cmd, args = self._parse_command(command)

        if UNC_PATH_PATTERN.search(command.strip()):
            return ValidationResult(
                allowed=False,
                risk=CommandRisk.DANGEROUS,
                reason="UNC path blocked: credential leak risk",
                blocked_patterns=["unc_path"],
                suggested_fix="Use a local path instead of UNC/network path",
            )

        # Check blocked commands
        if base_cmd in self.blocked_commands:
            suggestion = self._get_suggested_fix(base_cmd, args)
            return ValidationResult(
                allowed=False,
                risk=CommandRisk.DANGEROUS,
                reason=f"Blocked command: {base_cmd}",
                blocked_patterns=[base_cmd],
                suggested_fix=suggestion,
            )

        # Check dangerous patterns
        blocked_patterns = self._check_dangerous_patterns(command)
        if blocked_patterns:
            suggestion = "Remove dangerous constructs and use explicit arguments"
            if "rm" in command and "-rf" in command:
                suggestion = "Use 'rm -i' for interactive deletion or delete files individually"
            return ValidationResult(
                allowed=False,
                risk=CommandRisk.DANGEROUS,
                reason=f"Dangerous patterns detected: {', '.join(blocked_patterns)}",
                blocked_patterns=blocked_patterns,
                suggested_fix=suggestion,
            )

        # Check Zsh dangerous commands in args
        for arg in args:
            if arg in ZSH_DANGEROUS_COMMANDS:
                return ValidationResult(
                    allowed=False,
                    risk=CommandRisk.DANGEROUS,
                    reason=f"Zsh dangerous command: {arg}",
                    blocked_patterns=[arg],
                )

        if base_cmd == "git":
            git_result = self._validate_git(args)
            if git_result is not None:
                return git_result

        if base_cmd == "gh":
            gh_result = self._validate_gh(args)
            if gh_result is not None:
                return gh_result

        # Check high-risk commands
        if base_cmd in HIGH_RISK_COMMANDS or any(hr in command for hr in HIGH_RISK_COMMANDS):
            return ValidationResult(
                allowed=True,  # Allowed but requires confirmation
                risk=CommandRisk.HIGH,
                reason=f"High-risk command: {base_cmd}",
                requires_confirmation=self.require_confirmation_for_high_risk,
            )

        # Check if read-only
        if base_cmd in READ_ONLY_COMMANDS:
            # Validate flags
            unsafe_flags = self._check_unsafe_flags(base_cmd, args)
            if unsafe_flags:
                return ValidationResult(
                    allowed=True,
                    risk=CommandRisk.MEDIUM,
                    reason=f"Unsafe flags for {base_cmd}: {', '.join(unsafe_flags)}",
                    requires_confirmation=True,
                )

            return ValidationResult(
                allowed=True,
                risk=CommandRisk.SAFE,
                reason=None,
            )

        # Unknown command - medium risk
        return ValidationResult(
            allowed=True,
            risk=CommandRisk.MEDIUM,
            reason=f"Unknown command: {base_cmd}",
            requires_confirmation=True,
        )

    def _parse_command(self, command: str) -> tuple[str, list[str]]:
        """Parse command into base command and args."""
        # Simple split (doesn't handle quotes properly)
        try:
            parts = shlex.split(command.strip(), posix=False)
        except ValueError:
            parts = command.strip().split()
        if not parts:
            return ("", [])

        base_cmd = parts[0].strip('"').strip("'")
        if "/" in base_cmd or "\\" in base_cmd:
            base_cmd = re.split(r"[\\/]", base_cmd)[-1]
        base_cmd = base_cmd.lower()

        return (base_cmd, parts[1:])

    def _validate_git(self, args: list[str]) -> ValidationResult | None:
        if not args:
            return ValidationResult(
                allowed=True,
                risk=CommandRisk.MEDIUM,
                reason="git without subcommand",
                requires_confirmation=True,
            )
        sub_idx = self._find_git_subcommand_index(args)
        if sub_idx is None:
            return ValidationResult(
                allowed=True,
                risk=CommandRisk.MEDIUM,
                reason="Unable to determine git subcommand",
                requires_confirmation=True,
            )
        sub = args[sub_idx].lower()
        sub_args = args[sub_idx + 1:]
        if sub not in GIT_READ_ONLY_SUBCOMMANDS:
            return ValidationResult(
                allowed=True,
                risk=CommandRisk.HIGH,
                reason=f"High-risk git subcommand: {sub}",
                requires_confirmation=True,
            )
        if sub == "ls-remote":
            for arg in sub_args:
                arg_key = arg.split("=", 1)[0]
                if arg_key in {"--server-option", "-o"}:
                    return ValidationResult(
                        allowed=False,
                        risk=CommandRisk.DANGEROUS,
                        reason="git ls-remote: --server-option/-o blocked",
                        blocked_patterns=[arg_key],
                    )
        if sub == "reflog":
            lower_sub_args = [a.lower() for a in sub_args if not a.startswith("-")]
            if any(a in GIT_RESTRICTED_REFS for a in lower_sub_args):
                return ValidationResult(
                    allowed=False,
                    risk=CommandRisk.DANGEROUS,
                    reason="git reflog dangerous subcommand blocked",
                    blocked_patterns=[a for a in lower_sub_args if a in GIT_RESTRICTED_REFS],
                )
        if sub == "tag":
            if not any(a in {"-l", "--list"} for a in sub_args):
                return ValidationResult(
                    allowed=False,
                    risk=CommandRisk.DANGEROUS,
                    reason="git tag creation blocked (use -l/--list only)",
                    blocked_patterns=["tag_create"],
                )
        if sub == "branch":
            if not any(a in {"-l", "--list"} for a in sub_args):
                return ValidationResult(
                    allowed=False,
                    risk=CommandRisk.DANGEROUS,
                    reason="git branch creation blocked (use -l/--list only)",
                    blocked_patterns=["branch_create"],
                )
        if sub == "stash":
            if not sub_args or sub_args[0].lower() != "list":
                return ValidationResult(
                    allowed=False,
                    risk=CommandRisk.DANGEROUS,
                    reason="Only git stash list is allowed in read-only mode",
                    blocked_patterns=["stash_write"],
                )
        if sub == "worktree":
            if not sub_args or sub_args[0].lower() != "list":
                return ValidationResult(
                    allowed=False,
                    risk=CommandRisk.DANGEROUS,
                    reason="Only git worktree list is allowed in read-only mode",
                    blocked_patterns=["worktree_write"],
                )
        unsafe_flags = self._check_git_unsafe_flags(sub_args)
        if unsafe_flags:
            return ValidationResult(
                allowed=False,
                risk=CommandRisk.DANGEROUS,
                reason=f"Unsafe git flags: {', '.join(unsafe_flags)}",
                blocked_patterns=unsafe_flags,
            )
        return ValidationResult(
            allowed=True,
            risk=CommandRisk.SAFE,
            reason=None,
        )

    def _find_git_subcommand_index(self, args: list[str]) -> int | None:
        idx = 0
        while idx < len(args):
            token = args[idx]
            if token == "--":
                idx += 1
                break
            if token.startswith("-"):
                if token in {"-C", "-c", "--git-dir", "--work-tree", "--namespace"}:
                    idx += 2
                else:
                    idx += 1
                continue
            return idx
        return idx if idx < len(args) else None

    def _check_git_unsafe_flags(self, args: list[str]) -> list[str]:
        unsafe: list[str] = []
        skip_next = False
        for i, arg in enumerate(args):
            if skip_next:
                skip_next = False
                continue
            if arg == "--":
                break
            if not arg.startswith("-"):
                continue
            flag = arg.split("=", 1)[0]
            if flag in {"--exec-path", "--html-path", "--man-path", "--info-path"}:
                unsafe.append(flag)
                continue
            if flag in GIT_FLAG_TYPES:
                flag_type = GIT_FLAG_TYPES[flag]
                if flag_type != "none" and "=" not in arg:
                    if i + 1 >= len(args):
                        unsafe.append(flag)
                    else:
                        skip_next = True
                continue
            if flag.startswith("--format=") or flag.startswith("--contains=") or flag.startswith("--sort="):
                continue
            if re.fullmatch(r"-[0-9]+", flag):
                continue
            if re.fullmatch(r"-n[0-9]+", flag):
                continue
            unsafe.append(flag)
        return unsafe

    def _validate_gh(self, args: list[str]) -> ValidationResult | None:
        if not args:
            return ValidationResult(
                allowed=True,
                risk=CommandRisk.MEDIUM,
                reason="gh without subcommand",
                requires_confirmation=True,
            )
        sub = args[0].lower()
        if sub not in GH_READ_ONLY_SUBCOMMANDS:
            return ValidationResult(
                allowed=True,
                risk=CommandRisk.HIGH,
                reason=f"High-risk gh subcommand: {sub}",
                requires_confirmation=True,
            )
        if sub in {"repo", "pr", "issue", "run", "release"}:
            if len(args) < 2 or args[1].lower() not in {"list", "view"}:
                return ValidationResult(
                    allowed=True,
                    risk=CommandRisk.MEDIUM,
                    reason=f"gh {sub} subcommand requires confirmation",
                    requires_confirmation=True,
                )
        return ValidationResult(
            allowed=True,
            risk=CommandRisk.SAFE,
            reason=None,
        )

    def _check_dangerous_patterns(self, command: str) -> list[str]:
        """Check dangerous command patterns."""
        blocked = []

        for pattern, description in COMMAND_SUBSTITUTION_PATTERNS:
            if re.search(pattern, command):
                blocked.append(description)

        # Check for backticks (unescaped)
        if re.search(r"(?<!\\)`", command):
            blocked.append("backtick command substitution")

        # Check for semicolon (command chaining)
        if ";" in command and not re.search(r"\\;", command):
            blocked.append("semicolon command chaining")

        # Check for pipe to dangerous commands
        if "|" in command:
            pipe_parts = command.split("|")
            for part in pipe_parts:
                part_cmd = part.strip().split()[0] if part.strip().split() else ""
                if part_cmd in self.blocked_commands:
                    blocked.append(f"pipe to blocked command: {part_cmd}")

        # Check for redirection to files (potential write)
        if re.search(r">\s*\S", command) and not re.search(r">\s*/dev/null", command):
            blocked.append("output redirection to file")

        # Check for input redirection from dangerous sources
        if re.search(r"<\s*\S", command) and not re.search(r"<\s*/dev/null", command):
            # Input redirection is generally safer
            pass

        return blocked

    def _check_unsafe_flags(self, command: str, args: list[str]) -> list[str]:
        """Check unsafe flags."""
        safe_flags = SAFE_FLAGS.get(command, set())
        unsafe = []

        for arg in args:
            # Check if arg is a flag
            if arg.startswith("-"):
                # Extract flag name (handle --flag=value)
                flag = arg.split("=")[0]
                if flag not in safe_flags:
                    # Check if it's a known unsafe flag
                    if flag in ("-i", "-e", "-exec", "-exec-batch", "-x", "-X") or flag in ("-w", "--write", "-o", "--output"):
                        unsafe.append(flag)

        return unsafe

    def _get_suggested_fix(self, blocked_cmd: str, args: list[str]) -> str | None:
        """Suggest a safe alternative for blocked commands."""
        suggestions: dict[str, str] = {
            "rm": "Use 'trash' or move files to a temporary directory instead",
            "chmod": "Check current permissions with 'ls -la' first",
            "chown": "Consult system administrator for ownership changes",
            "dd": "Use 'cp' for simple file copying",
            "shutdown": "Use 'systemctl poweroff' with proper authorization",
            "reboot": "Use 'systemctl reboot' with proper authorization",
            "eval": "Avoid dynamic code execution; use explicit commands",
            "exec": "Run commands directly without exec",
            "source": "Use explicit script path or check script contents first",
            "sudo": "Run with appropriate user permissions instead",
            "su": "Use 'sudo -u <user>' for specific user context",
            "kill": "Use 'kill -l' to list signals, then specify signal explicitly",
            "iptables": "Use 'ufw' for simpler firewall management",
        }
        return suggestions.get(blocked_cmd)

    def is_read_only(self, command: str) -> bool:
        """Return True when command is read-only."""
        result = self.validate(command)
        return result.risk == CommandRisk.SAFE

    def requires_confirmation(self, command: str) -> bool:
        """Return True when command requires confirmation."""
        result = self.validate(command)
        return result.requires_confirmation or result.risk in (CommandRisk.HIGH, CommandRisk.DANGEROUS)


# Permission rule parsing helpers.

def escape_rule_content(content: str) -> str:
    """Escape special characters in rule content."""
    return (
        content
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def unescape_rule_content(content: str) -> str:
    """Unescape special characters in rule content."""
    return (
        content
        .replace("\\(", "(")
        .replace("\\)", ")")
        .replace("\\\\", "\\")
    )


def parse_permission_rule(rule: str) -> tuple[str, str | None]:
    """
    Parse permission rule.

    Format: "ToolName" or "ToolName(content)"

    Returns:
        (tool_name, content or None)
    """
    # Find first unescaped (
    open_idx = -1
    for i, char in enumerate(rule):
        if char == "(" and (i == 0 or rule[i - 1] != "\\"):
            open_idx = i
            break

    if open_idx == -1:
        return (rule, None)

    # Find last unescaped )
    close_idx = -1
    for i in range(len(rule) - 1, -1, -1):
        if rule[i] == ")" and (i == 0 or rule[i - 1] != "\\"):
            close_idx = i
            break

    if close_idx == -1 or close_idx <= open_idx:
        return (rule, None)

    tool_name = rule[:open_idx]
    raw_content = rule[open_idx + 1:close_idx]

    if not tool_name or raw_content in ("", "*"):
        return (tool_name or rule, None)

    content = unescape_rule_content(raw_content)
    return (tool_name, content)


def permission_rule_to_string(tool_name: str, content: str | None) -> str:
    """Convert permission rule to string."""
    if not content:
        return tool_name
    escaped = escape_rule_content(content)
    return f"{tool_name}({escaped})"


_shell_guard: ShellGuard | None = None


def get_shell_guard() -> ShellGuard:
    """Return global ShellGuard instance."""
    global _shell_guard
    if _shell_guard is None:
        _shell_guard = ShellGuard()
    return _shell_guard


def validate_shell_command(command: str) -> ValidationResult:
    """Validate shell command."""
    return get_shell_guard().validate(command)


def shell_guards_for_user_commands_enabled() -> bool:
    """Return whether shell guard checks are enabled for user commands."""
    v = (os.getenv("CORE_SHELL_GUARDS_CMD_RUN") or "1").strip().lower()
    return v not in {"0", "false", "no", "off", "n"}


def shell_command_blocked_reason(command: str) -> str | None:
    """Return block reason when command is denied, otherwise None."""
    if not shell_guards_for_user_commands_enabled():
        return None
    vr = validate_shell_command(command.strip())
    if vr.allowed:
        return None
    return vr.reason or "Shell command blocked"
