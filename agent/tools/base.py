"""
tools/base.py — agent stage 工具的公共类型与路径边界。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional


@dataclass
class ToolResult:
    ok: bool
    content: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_tool_message_content(self) -> str:
        if self.ok:
            return self.content
        prefix = self.meta.get("error_type", "error")
        return f"[{prefix}] {self.content}"


@dataclass
class ToolContext:
    """单次 agent stage 运行时的共享上下文。"""

    workdir: Path
    workspace_root: Path
    cwd: Path
    stage_name: str
    max_tool_output_chars: int = 32_000

    def resolve_path(self, path: str) -> Path:
        """将相对路径解析到 workspace_root 下，拒绝越界。"""
        raw = Path(path)
        if raw.is_absolute():
            candidate = raw.resolve()
        else:
            candidate = (self.workspace_root / raw).resolve()
        root = self.workspace_root.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as e:
            raise PermissionError(
                f"Path '{path}' resolves to '{candidate}', which is outside "
                f"workspace_root '{root}'"
            ) from e
        return candidate

    def truncate(self, text: str) -> str:
        if len(text) <= self.max_tool_output_chars:
            return text
        head = self.max_tool_output_chars // 2
        tail = self.max_tool_output_chars - head
        omitted = len(text) - self.max_tool_output_chars
        return (
            text[:head]
            + f"\n\n...[truncated {omitted} chars]...\n\n"
            + text[-tail:]
        )


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: Dict[str, Any]
    handler: Callable[[ToolContext, Dict[str, Any]], ToolResult]

    def openai_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
