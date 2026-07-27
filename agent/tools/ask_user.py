"""tools/ask_user.py — human-in-the-loop stdin prompt."""

from __future__ import annotations

import sys
from typing import Any, Dict, List

from tools.base import ToolContext, ToolResult, ToolSpec


def _ask_user(ctx: ToolContext, args: Dict[str, Any]) -> ToolResult:
    question = args.get("question")
    if not question or not isinstance(question, str):
        return ToolResult(ok=False, content="Missing required string argument: question")

    options = args.get("options")
    option_lines: List[str] = []
    if isinstance(options, list) and options:
        for i, opt in enumerate(options, 1):
            option_lines.append(f"  {i}. {opt}")

    if not sys.stdin.isatty():
        return ToolResult(
            ok=False,
            content=(
                "ask_user requires an interactive TTY stdin. "
                "Re-run in a terminal, or provide non-interactive inputs in a future config."
            ),
            meta={"error_type": "no_tty"},
        )

    print("\n" + "=" * 60)
    print(f"[agent:{ctx.stage_name}] ask_user")
    print(question)
    if option_lines:
        print("Options:")
        print("\n".join(option_lines))
    print("=" * 60)
    try:
        answer = input("Your reply> ").strip()
    except EOFError:
        return ToolResult(
            ok=False,
            content="stdin closed before answer was provided",
            meta={"error_type": "eof"},
        )

    return ToolResult(
        ok=True,
        content=answer,
        meta={"question": question, "options": options or []},
    )


ASK_USER_TOOL = ToolSpec(
    name="ask_user",
    description=(
        "Ask the human operator a question and wait for a reply on stdin. "
        "Use this when you need LaTeX, clarifications, or confirmation. "
        "Requires an interactive terminal."
    ),
    parameters={
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Question shown to the human",
            },
            "options": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional list of suggested answers",
            },
        },
        "required": ["question"],
    },
    handler=_ask_user,
)
