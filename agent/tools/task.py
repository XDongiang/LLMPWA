"""tools/task.py — agent meta-control: finish / note."""

from __future__ import annotations

from typing import Any, Dict

from tools.base import ToolContext, ToolResult, ToolSpec


def _task(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    action = args.get("action")
    if action not in ("finish", "note"):
        return ToolResult(
            ok=False,
            content="action must be one of: finish, note",
            meta={"error_type": "invalid_action"},
        )

    if action == "finish":
        result = args.get("result", "")
        if result is None:
            result = ""
        if not isinstance(result, str):
            result = str(result)
        return ToolResult(
            ok=True,
            content="Stage finished.",
            meta={
                "action": "finish",
                "result": result,
                "finished": True,
            },
        )

    # note
    message = args.get("message") or args.get("result") or ""
    if not isinstance(message, str):
        message = str(message)
    return ToolResult(
        ok=True,
        content=f"Noted: {message}" if message else "Noted.",
        meta={"action": "note", "message": message},
    )


TASK_TOOL = ToolSpec(
    name="task",
    description=(
        "Agent meta-control. Use action='finish' with the final result string "
        "to submit the primary stage output for (optional) human approval; "
        "if rejected, revise and finish again. Use action='note' to record "
        "intermediate progress for the transcript (does not finish)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["finish", "note"],
                "description": "finish ends the stage; note records a progress note",
            },
            "result": {
                "type": "string",
                "description": "Final output when action=finish (becomes stage output.all)",
            },
            "message": {
                "type": "string",
                "description": "Progress note when action=note",
            },
        },
        "required": ["action"],
    },
    handler=_task,
)
