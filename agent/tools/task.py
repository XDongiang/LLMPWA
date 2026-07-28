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

        # Optional checklist: {output_field: relative_path_written}
        # Used by AgentStageRunner to verify multi-file stage.output decls.
        outputs = args.get("outputs")
        if outputs is None:
            outputs_map: Dict[str, str] = {}
        elif isinstance(outputs, dict):
            outputs_map = {}
            for k, v in outputs.items():
                if not isinstance(k, str):
                    return ToolResult(
                        ok=False,
                        content="outputs keys must be strings (stage output field names)",
                        meta={"error_type": "invalid_outputs"},
                    )
                if not isinstance(v, str):
                    return ToolResult(
                        ok=False,
                        content=(
                            f"outputs[{k!r}] must be a relative path string, "
                            f"got {type(v).__name__}"
                        ),
                        meta={"error_type": "invalid_outputs"},
                    )
                outputs_map[k] = v
        else:
            return ToolResult(
                ok=False,
                content="outputs must be an object mapping field name → relative path",
                meta={"error_type": "invalid_outputs"},
            )

        return ToolResult(
            ok=True,
            content="Stage finished.",
            meta={
                "action": "finish",
                "result": result,
                "outputs": outputs_map,
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
        "if rejected, revise and finish again. When the stage declares multiple "
        "output.xxx file fields, write each file first, then pass "
        "outputs={field: relative_path, ...} as a checklist. Use action='note' "
        "to record intermediate progress for the transcript (does not finish)."
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
                "description": (
                    "Final primary text when action=finish (becomes stage "
                    "output.all when that field is not harvested from a written file)"
                ),
            },
            "outputs": {
                "type": "object",
                "description": (
                    "Optional checklist mapping stage output field names to the "
                    "relative paths you wrote (must match config output.*.path). "
                    "Required in practice for multi-file stages."
                ),
                "additionalProperties": {"type": "string"},
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
