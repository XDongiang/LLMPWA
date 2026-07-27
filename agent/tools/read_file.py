"""tools/read_file.py — read a file under workspace_root."""

from __future__ import annotations

from typing import Any, Dict

from tools.base import ToolContext, ToolResult, ToolSpec


def _read(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    path = args.get("path")
    if not path or not isinstance(path, str):
        return ToolResult(ok=False, content="Missing required string argument: path")

    try:
        abs_path = ctx.resolve_path(path)
    except PermissionError as e:
        return ToolResult(ok=False, content=str(e), meta={"error_type": "path_error"})

    if not abs_path.exists():
        return ToolResult(ok=False, content=f"File not found: {path}")
    if not abs_path.is_file():
        return ToolResult(ok=False, content=f"Not a file: {path}")

    try:
        text = abs_path.read_text(encoding="utf-8")
    except Exception as e:
        return ToolResult(ok=False, content=f"Failed to read '{path}': {e}")

    offset = args.get("offset")
    limit = args.get("limit")
    lines = text.splitlines(keepends=True)
    start = 0
    if isinstance(offset, int) and offset > 0:
        # 1-based line offset, like common editor conventions
        start = max(offset - 1, 0)
    end = len(lines)
    if isinstance(limit, int) and limit >= 0:
        end = min(start + limit, len(lines))
    sliced = "".join(lines[start:end])
    meta = {
        "path": str(abs_path.relative_to(ctx.workspace_root)),
        "total_lines": len(lines),
        "start_line": start + 1 if lines else 0,
        "end_line": end,
    }
    return ToolResult(ok=True, content=ctx.truncate(sliced), meta=meta)


READ_TOOL = ToolSpec(
    name="read",
    description=(
        "Read a text file under the workspace. Paths are relative to workspace_root. "
        "Optional 1-based line offset and limit control the slice returned."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Relative path to the file"},
            "offset": {
                "type": "integer",
                "description": "Optional 1-based starting line number",
            },
            "limit": {
                "type": "integer",
                "description": "Optional maximum number of lines to return",
            },
        },
        "required": ["path"],
    },
    handler=_read,
)
