"""
agent_stage.py — interactive multi-turn agent stage runner (kind=agent).

OpenAI tools/tool_calls loop with read/write/shell/task/ask_user.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from config_loader import StageConfig
from tools.base import ToolContext
from tools.registry import DEFAULT_TOOL_NAMES, ToolRegistry


def _compute_hash(*parts: str) -> str:
    combined = "\n---\n".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


def _strip_fences(code: str) -> str:
    code = re.sub(r"^```\w*\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code.strip())
    return code


def _read_prompt_decl(decl: Any, workdir: Path, stage_name: str, field: str) -> str:
    if decl is None:
        return ""
    if isinstance(decl, dict) and decl.get("type") == "file":
        path = workdir / decl["path"]
        if not path.exists():
            raise ValueError(
                f"Stage '{stage_name}': {field} file not found: {path}"
            )
        return path.read_text(encoding="utf-8")
    if isinstance(decl, str):
        return decl
    raise ValueError(
        f"Stage '{stage_name}': invalid {field} declaration: {decl!r}"
    )


class AgentStageRunner:
    """Run one kind=agent stage to completion and write manifest outputs."""

    def __init__(self, name: str, stage_cfg: StageConfig, engine: Any) -> None:
        self.name = name
        self.stage_cfg = stage_cfg
        self.engine = engine
        self.workdir: Path = engine.workdir
        self.manifest = engine.manifest
        self.resolver = engine.resolver

    def execute(self) -> None:
        cfg = self.stage_cfg
        system_template = _read_prompt_decl(
            cfg.system_prompt, self.workdir, self.name, "system_prompt"
        )
        user_template = _read_prompt_decl(
            cfg.prompt, self.workdir, self.name, "prompt"
        )
        if not system_template and not user_template and not cfg.human_prompt:
            raise ValueError(
                f"Stage '{self.name}': agent stage requires system_prompt, "
                "prompt, and/or human_prompt"
            )

        system_text = self.resolver.render(system_template) if system_template else ""
        user_text = self.resolver.render(user_template) if user_template else ""
        if cfg.human_prompt:
            human = self.resolver.render(cfg.human_prompt)
            user_text = (user_text + "\n\n" + human).strip() if user_text else human

        if not system_text:
            system_text = (
                "You are a coding agent with tools. Use tools to inspect and "
                "edit files under the workspace. When the task is complete, "
                "call task with action='finish' and put the final result in "
                "the result field. Prefer ask_user when you need human input."
            )

        tool_names = cfg.tools if cfg.tools is not None else DEFAULT_TOOL_NAMES
        registry = ToolRegistry(tool_names)
        tools_schema = registry.openai_tools()

        workspace_root = (self.workdir / (cfg.workspace_root or ".")).resolve()
        cwd = (self.workdir / (cfg.cwd or ".")).resolve()
        cwd.mkdir(parents=True, exist_ok=True)
        workspace_root.mkdir(parents=True, exist_ok=True)

        ctx = ToolContext(
            workdir=self.workdir,
            workspace_root=workspace_root,
            cwd=cwd,
            stage_name=self.name,
        )

        prompt_hash = _compute_hash(
            system_text,
            user_text,
            ",".join(registry.names),
            str(cfg.max_turns),
            cfg.cwd or ".",
            cfg.workspace_root or ".",
        )

        if cfg.cache and self.manifest.is_cached(self.name, prompt_hash, self.workdir):
            print(f"[agent] {self.name} — cache hit")
            return

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text or "Begin."},
        ]
        transcript: List[Dict[str, Any]] = [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text or "Begin."},
        ]

        print(
            f"[agent] {self.name} — start "
            f"(tools={registry.names}, max_turns={cfg.max_turns})"
        )

        final_result: Optional[str] = None
        used_ask_user = False
        last_assistant_text = ""

        for turn in range(1, cfg.max_turns + 1):
            print(f"[agent] {self.name} — turn {turn}/{cfg.max_turns}")
            message, tool_calls, _raw = self.engine.llm_client.chat_with_tools(
                messages=messages,
                tools=tools_schema,
            )
            # Normalize assistant message for history (drop None content noise)
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": message.get("content"),
            }
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            # Some providers require content to be string when tool_calls present
            if assistant_msg["content"] is None and tool_calls:
                assistant_msg["content"] = ""

            messages.append(assistant_msg)
            transcript.append(
                {
                    "role": "assistant",
                    "content": message.get("content"),
                    "tool_calls": tool_calls or None,
                    "turn": turn,
                }
            )

            content = message.get("content") or ""
            if isinstance(content, str) and content.strip():
                last_assistant_text = content.strip()

            if not tool_calls:
                if cfg.finish_on_message and last_assistant_text:
                    final_result = _strip_fences(last_assistant_text)
                    print(f"[agent] {self.name} — finished via plain message")
                    break
                # Nudge once toward explicit finish rather than hanging
                nudge = (
                    "No tool call received. If the task is done, call the "
                    "`task` tool with action='finish' and put the final "
                    "output in `result`. Otherwise continue using tools."
                )
                messages.append({"role": "user", "content": nudge})
                transcript.append({"role": "user", "content": nudge, "turn": turn})
                time.sleep(0.2)
                continue

            finished_this_turn = False
            for tc in tool_calls:
                tc_id = tc.get("id") or f"call_{turn}"
                fn = tc.get("function") or {}
                name = fn.get("name") or ""
                raw_args = fn.get("arguments")
                print(f"[agent] {self.name} — tool {name}({_preview_args(raw_args)})")

                result = registry.dispatch(ctx, name, raw_args)
                if name == "ask_user":
                    used_ask_user = True
                if result.meta.get("finished"):
                    final_result = result.meta.get("result", "")
                    if not isinstance(final_result, str):
                        final_result = str(final_result)
                    finished_this_turn = True

                tool_content = result.as_tool_message_content()
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": tool_content,
                }
                # Some gateways also want name on tool messages
                if name:
                    tool_msg["name"] = name
                messages.append(tool_msg)
                transcript.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "name": name,
                        "content": tool_content,
                        "ok": result.ok,
                        "meta": result.meta,
                        "turn": turn,
                    }
                )

            if finished_this_turn:
                print(f"[agent] {self.name} — finished via task.finish")
                break

            time.sleep(0.2)
        else:
            # max_turns exhausted without finish
            if cfg.on_max_turns == "use_last_message" and last_assistant_text:
                final_result = _strip_fences(last_assistant_text)
                print(
                    f"[agent] {self.name} — max_turns reached; "
                    "using last assistant message"
                )
            else:
                raise RuntimeError(
                    f"Stage '{self.name}': agent exceeded max_turns={cfg.max_turns} "
                    "without task.finish"
                )

        if final_result is None:
            final_result = last_assistant_text or ""

        raw_outputs: Dict[str, Any] = {"all": final_result}
        if self.stage_cfg.output_type == "json":
            try:
                parsed = json.loads(final_result)
                if isinstance(parsed, dict):
                    raw_outputs["all"] = parsed
                    raw_outputs.update(parsed)
            except json.JSONDecodeError:
                print(
                    f"[agent] {self.name} — output_type=json but result is not "
                    "valid JSON; storing raw string under all"
                )

        # Always attach transcript value if declared (or for log convenience)
        raw_outputs["transcript"] = {
            "stage": self.name,
            "turns": len({t.get("turn") for t in transcript if "turn" in t}),
            "used_ask_user": used_ask_user,
            "messages": transcript,
            "final_result": final_result,
        }

        decls = self.stage_cfg.output or {}
        rendered_decls = {}
        for k, v in decls.items():
            if isinstance(v, dict) and "path" in v:
                rendered_decls[k] = {
                    **v,
                    "path": self.resolver.render(v["path"]),
                }
            else:
                rendered_decls[k] = v

        # If transcript was produced but not declared, still save llm-style log
        if "transcript" not in rendered_decls:
            self._save_transcript_fallback(raw_outputs["transcript"])

        outputs = (
            {k: v for k, v in raw_outputs.items() if k in decls}
            if decls
            else {"all": raw_outputs["all"]}
        )

        # HITL conversations should not poison cache unless explicitly allowed
        # and no ask_user occurred
        effective_hash = prompt_hash
        if used_ask_user or not cfg.cache:
            # Still store hash for debugging, but is_cached requires cfg.cache
            # which is false by default — keep prompt_hash for reproducibility notes
            pass

        self.manifest.write_stage_output(
            stage=self.name,
            outputs=outputs,
            output_decls=rendered_decls if rendered_decls else {
                "all": {"type": "direct"}
            },
            workdir=self.workdir,
            prompt_hash=effective_hash if cfg.cache and not used_ask_user else "",
        )

        # Also write a human-readable md log like llm stages
        self.manifest.save_llm_log(
            self.name,
            f"[system]\n{system_text}\n\n[user]\n{user_text}",
            final_result if isinstance(final_result, str) else json.dumps(final_result, ensure_ascii=False),
            self.workdir,
        )
        print(f"[agent] {self.name} — done")

    def _save_transcript_fallback(self, transcript: dict) -> None:
        log_dir = self.workdir / "gen" / "llm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{self.name.replace('.', '_')}_transcript.json"
        path.write_text(
            json.dumps(transcript, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def _preview_args(raw_args: Any, limit: int = 120) -> str:
    if raw_args is None:
        return ""
    text = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
    text = text.replace("\n", " ")
    if len(text) > limit:
        return text[:limit] + "..."
    return text
