"""tools/shell.py — run a shell command under workspace cwd."""

from __future__ import annotations

import subprocess
from typing import Any, Dict

from tools.base import ToolContext, ToolResult, ToolSpec


def _shell(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    command = args.get("command")
    if not command or not isinstance(command, str):
        return ToolResult(ok=False, content="Missing required string argument: command")

    timeout_s = args.get("timeout_s", 60)
    if not isinstance(timeout_s, (int, float)) or timeout_s <= 0:
        timeout_s = 60

    try:
        completed = subprocess.run(
            command,
            shell=True,
            cwd=str(ctx.cwd),
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            ok=False,
            content=f"Command timed out after {timeout_s}s: {command}",
            meta={"error_type": "timeout", "timeout_s": timeout_s},
        )
    except Exception as e:
        return ToolResult(ok=False, content=f"Failed to run command: {e}")

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    body_parts = []
    if stdout:
        body_parts.append(f"stdout:\n{stdout}")
    if stderr:
        body_parts.append(f"stderr:\n{stderr}")
    if not body_parts:
        body_parts.append("(no stdout/stderr)")
    body = "\n".join(body_parts)
    content = f"exit_code={completed.returncode}\n{body}"
    return ToolResult(
        ok=completed.returncode == 0,
        content=ctx.truncate(content),
        meta={
            "exit_code": completed.returncode,
            "cwd": str(ctx.cwd),
        },
    )


SHELL_TOOL = ToolSpec(
    name="shell",
    description=(
        "Run a shell command with cwd set to the stage workspace. "
        "Returns exit_code, stdout, and stderr. Use sparingly; prefer "
        "read/write for file edits."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to execute"},
            "timeout_s": {
                "type": "number",
                "description": "Timeout in seconds (default 60)",
            },
        },
        "required": ["command"],
    },
    handler=_shell,
)
