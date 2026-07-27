"""tools/write_file.py — write/overwrite a file under workspace_root."""

from __future__ import annotations

from typing import Any, Dict

from tools.base import ToolContext, ToolResult, ToolSpec


def _write(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = args.get("path")
    if not path or not isinstance(path, str):
        return ToolResult(ok=False, content="Missing required string argument: path")
    if "content" not in args:
        return ToolResult(ok=False, content="Missing required argument: content")
    content = args.get("content")
    if not isinstance(content, str):
        content = str(content)

    try:
        abs_path = ctx.resolve_path(path)
    except PermissionError as e:
        return ToolResult(ok=False, content=str(e), meta={"error_type": "path_error"})

    try:
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_text(content, encoding="utf-8")
    except Exception as e:
        return ToolResult(ok=False, content=f"Failed to write '{path}': {e}")

    rel = str(abs_path.relative_to(ctx.workspace_root))
    return ToolResult(
        ok=True,
        content=f"Wrote {len(content)} chars to {rel}",
        meta={"path": rel, "bytes": len(content.encode("utf-8"))},
    )


WRITE_TOOL = ToolSpec(
    name="write",
    description=(
        "Write or overwrite a text file under the workspace. Parent directories "
        "are created automatically. Paths are relative to workspace_root."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to write"},
            "content": {"type": "string", "description": "Full file content"},
        },
        "required": ["path", "content"],
    },
    handler=_write,
)
