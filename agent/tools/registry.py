"""tools/registry.py — tool registration and dispatch."""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tools.ask_user import ASK_USER_TOOL
from tools.base import ToolContext, ToolResult, ToolSpec
from tools.read_file import READ_TOOL
from tools.shell import SHELL_TOOL
from tools.task import TASK_TOOL
from tools.write_file import WRITE_TOOL


ALL_TOOLS: Dict[str, ToolSpec] = {
    READ_TOOL.name: READ_TOOL,
    WRITE_TOOL.name: WRITE_TOOL,
    SHELL_TOOL.name: SHELL_TOOL,
    TASK_TOOL.name: TASK_TOOL,
    ASK_USER_TOOL.name: ASK_USER_TOOL,
}

DEFAULT_TOOL_NAMES: List[str] = list(ALL_TOOLS.keys())


class ToolRegistry:
    def __init__(self, names: Optional[Sequence[str]] = None) -> None:
        if names is None:
            selected = DEFAULT_TOOL_NAMES
        else:
            selected = list(names)
        unknown = [n for n in selected if n not in ALL_TOOLS]
        if unknown:
            raise ValueError(
                f"Unknown tool name(s): {unknown}. "
                f"Available: {sorted(ALL_TOOLS)}"
            )
        self._tools: Dict[str, ToolSpec] = {n: ALL_TOOLS[n] for n in selected}

    @property
    def names(self) -> List[str]:
        return list(self._tools.keys())

    def openai_tools(self) -> List[Dict[str, Any]]:
        return [spec.openai_schema() for spec in self._tools.values()]

    def dispatch(self, ctx: ToolContext, name: str, arguments: Any) -> ToolResult:
        if name not in self._tools:
            return ToolResult(
                ok=False,
                content=f"Unknown or disabled tool: {name!r}. Enabled: {self.names}",
                meta={"error_type": "unknown_tool"},
            )
        args = _normalize_arguments(arguments)
        if isinstance(args, ToolResult):
            return args
        try:
            return self._tools[name].handler(ctx, args)
        except Exception as e:
            return ToolResult(
                ok=False,
                content=f"Tool '{name}' raised {type(e).__name__}: {e}",
                meta={"error_type": "exception"},
            )


def _normalize_arguments(arguments: Any) -> Dict[str, Any] | ToolResult:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        text = arguments.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            return ToolResult(
                ok=False,
                content=f"Invalid tool arguments JSON: {e}; raw={arguments!r}",
                meta={"error_type": "bad_arguments"},
            )
        if not isinstance(parsed, dict):
            return ToolResult(
                ok=False,
                content=f"Tool arguments must be a JSON object, got {type(parsed).__name__}",
                meta={"error_type": "bad_arguments"},
            )
        return parsed
    return ToolResult(
        ok=False,
        content=f"Unsupported arguments type: {type(arguments).__name__}",
        meta={"error_type": "bad_arguments"},
    )


def validate_tool_names(names: Iterable[str]) -> List[str]:
    errors: List[str] = []
    for n in names:
        if n not in ALL_TOOLS:
            errors.append(
                f"Unknown tool '{n}'. Available: {sorted(ALL_TOOLS.keys())}"
            )
    return errors
