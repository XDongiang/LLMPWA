"""
agent_stage.py — interactive multi-turn agent stage runner (kind=agent).

OpenAI tools/tool_calls loop with read/write/shell/task/ask_user.
After task.finish (or finish_on_message), optional human approval gate
(require_approval, default true for agent stages).
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config_loader import StageConfig
from tools.base import ToolContext, ToolResult
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


def _render_output_decls(decls: dict, resolver: Any) -> Dict[str, Any]:
    """Render <<...>> placeholders in declared output paths."""
    rendered: Dict[str, Any] = {}
    for k, v in (decls or {}).items():
        if isinstance(v, dict) and "path" in v:
            rendered[k] = {**v, "path": resolver.render(v["path"])}
        else:
            rendered[k] = v
    return rendered


def _file_output_contract(rendered_decls: dict) -> str:
    """Machine-readable list of required file outputs for the agent prompt."""
    lines = []
    for field, decl in rendered_decls.items():
        if not isinstance(decl, dict):
            continue
        if decl.get("type") != "file":
            continue
        if field == "transcript":
            # framework-owned; agent need not write it
            continue
        path = decl.get("path")
        if path:
            lines.append(f"- {field} → {path}")
    if not lines:
        return ""
    return (
        "## Required stage outputs (write these exact paths before task.finish)\n"
        + "\n".join(lines)
        + "\n\nWhen finishing multi-file stages, call:\n"
        "task(action='finish', result=<summary or primary text>, "
        "outputs={field: relative_path, ...})\n"
        "Paths in outputs must match the list above. The framework will harvest "
        "file contents into the corresponding stage output fields."
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

        decls = cfg.output or {}
        rendered_decls = _render_output_decls(decls, self.resolver)
        contract = _file_output_contract(rendered_decls)
        if contract:
            user_text = (user_text + "\n\n" + contract).strip() if user_text else contract

        if not system_text:
            system_text = (
                "You are a coding agent with tools. Use tools to inspect and "
                "edit files under the workspace. When the task is complete, "
                "call task with action='finish' and put the final result in "
                "the result field. For multi-file stages, write each declared "
                "output path first, then pass outputs={field: path} on finish. "
                "A human may review and reject the submission — if rejected, "
                "revise using the feedback and call task.finish again. Prefer "
                "ask_user when you need human input."
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
            json.dumps(
                {k: v for k, v in rendered_decls.items() if isinstance(v, dict)},
                sort_keys=True,
                ensure_ascii=False,
            ),
        )

        if cfg.cache and self.manifest.is_cached(
            self.name, prompt_hash, self.workdir, rendered_decls
        ):
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
        final_outputs_map: Dict[str, str] = {}
        used_ask_user = False
        last_assistant_text = ""
        approval_rounds = 0

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

            pending_result: Optional[str] = None
            pending_outputs_map: Dict[str, str] = {}

            if not tool_calls:
                if cfg.finish_on_message and last_assistant_text:
                    pending_result = _strip_fences(last_assistant_text)
                    print(f"[agent] {self.name} — submitted via plain message")
                else:
                    # Nudge once toward explicit finish rather than hanging
                    nudge = (
                        "No tool call received. If the task is done, call the "
                        "`task` tool with action='finish' and put the final "
                        "output in `result` (and `outputs={field:path}` for "
                        "multi-file stages). Otherwise continue using tools."
                    )
                    messages.append({"role": "user", "content": nudge})
                    transcript.append({"role": "user", "content": nudge, "turn": turn})
                    time.sleep(0.2)
                    continue
            else:
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
                        candidate = result.meta.get("result", "")
                        if not isinstance(candidate, str):
                            candidate = str(candidate)
                        pending_result = candidate
                        out_map = result.meta.get("outputs") or {}
                        if isinstance(out_map, dict):
                            pending_outputs_map = {
                                str(k): str(v) for k, v in out_map.items()
                            }
                        # Tool message: submission received; approval is separate
                        if cfg.require_approval:
                            result = ToolResult(
                                ok=True,
                                content=(
                                    "Submission received and queued for human review. "
                                    "Wait for approve/reject feedback before treating "
                                    "the stage as complete."
                                ),
                                meta={
                                    **result.meta,
                                    "finished": False,
                                    "pending_approval": True,
                                },
                            )
                        else:
                            print(f"[agent] {self.name} — finished via task.finish")

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

            if pending_result is not None:
                # Pre-check declared file outputs before accepting the finish
                harvest_err = self._check_file_outputs(
                    rendered_decls, pending_outputs_map
                )
                if harvest_err is not None:
                    used_ask_user = True  # interactive correction path
                    fix_msg = (
                        "Your task.finish submission is incomplete — required "
                        "output files are missing or mismatched:\n"
                        f"{harvest_err}\n\n"
                        "Write the missing files with the write tool (exact "
                        "paths), then call task.finish again with "
                        "outputs={field: relative_path, ...}."
                    )
                    messages.append({"role": "user", "content": fix_msg})
                    transcript.append(
                        {
                            "role": "user",
                            "content": fix_msg,
                            "turn": turn,
                            "kind": "outputs_incomplete",
                        }
                    )
                    print(
                        f"[agent] {self.name} — finish rejected: incomplete outputs"
                    )
                    time.sleep(0.2)
                    continue

                approval_preview = self._approval_preview(
                    pending_result, rendered_decls, pending_outputs_map
                )

                if not cfg.require_approval:
                    final_result = pending_result
                    final_outputs_map = pending_outputs_map
                    break

                approved, feedback = self._request_human_approval(approval_preview)
                approval_rounds += 1
                used_ask_user = True  # HITL → do not cache
                transcript.append(
                    {
                        "role": "human_approval",
                        "approved": approved,
                        "feedback": feedback,
                        "submitted_result": pending_result,
                        "submitted_outputs": pending_outputs_map,
                        "turn": turn,
                        "approval_round": approval_rounds,
                    }
                )
                if approved:
                    final_result = pending_result
                    final_outputs_map = pending_outputs_map
                    print(
                        f"[agent] {self.name} — approved "
                        f"(round {approval_rounds})"
                    )
                    break

                reject_msg = (
                    "Human reviewer REJECTED your submission.\n"
                    f"Feedback:\n{feedback or '(no additional feedback)'}\n\n"
                    "Revise the work using tools as needed, then call "
                    "`task` with action='finish', an updated `result`, and "
                    "`outputs={field: path}` for multi-file stages."
                )
                messages.append({"role": "user", "content": reject_msg})
                transcript.append(
                    {
                        "role": "user",
                        "content": reject_msg,
                        "turn": turn,
                        "kind": "approval_reject",
                    }
                )
                print(
                    f"[agent] {self.name} — rejected "
                    f"(round {approval_rounds}); continuing"
                )
                time.sleep(0.2)
                continue

            time.sleep(0.2)
        else:
            # max_turns exhausted without approved finish
            if cfg.on_max_turns == "use_last_message" and last_assistant_text:
                final_result = _strip_fences(last_assistant_text)
                print(
                    f"[agent] {self.name} — max_turns reached; "
                    "using last assistant message"
                )
            else:
                raise RuntimeError(
                    f"Stage '{self.name}': agent exceeded max_turns={cfg.max_turns} "
                    "without approved task.finish"
                    + (
                        f" (had {approval_rounds} rejection(s))"
                        if approval_rounds
                        else ""
                    )
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

        # Harvest declared file outputs (except transcript, filled below).
        # File content wins over task.finish(result) for any harvested field,
        # including `all` when the agent already wrote the report path.
        harvested = self._harvest_file_outputs(
            rendered_decls, final_outputs_map, require_all=True
        )
        raw_outputs.update(harvested)
        if "all" in harvested:
            print(
                f"[agent] {self.name} — output.all taken from written file "
                f"(not task.finish result)"
            )

        # Always attach transcript value if declared (or for log convenience)
        raw_outputs["transcript"] = {
            "stage": self.name,
            "turns": len({t.get("turn") for t in transcript if "turn" in t}),
            "used_ask_user": used_ask_user,
            "require_approval": cfg.require_approval,
            "approval_rounds": approval_rounds,
            "messages": transcript,
            "final_result": final_result,
            "outputs_map": final_outputs_map,
            "harvested_fields": sorted(harvested.keys()),
        }

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

    def _declared_file_fields(self, rendered_decls: dict) -> Dict[str, str]:
        """field → relative path for agent-visible type=file outputs.

        ``transcript`` is framework-owned and excluded.
        """
        fields: Dict[str, str] = {}
        for field, decl in (rendered_decls or {}).items():
            if not isinstance(decl, dict):
                continue
            if decl.get("type") != "file":
                continue
            if field == "transcript":
                continue
            path = decl.get("path")
            if path:
                fields[field] = str(path)
        return fields

    def _strict_file_fields(self, rendered_decls: dict) -> Dict[str, str]:
        """Fields that must already exist on disk before task.finish is accepted.

        ``all`` is special-cased: legacy single-output agents may only put the
        primary text in task.finish(result=...) and let the framework write
        output.all. Extra fragment fields (load_data, ...) must be written.
        """
        fields = self._declared_file_fields(rendered_decls)
        return {k: v for k, v in fields.items() if k != "all"}

    def _check_file_outputs(
        self,
        rendered_decls: dict,
        outputs_map: Dict[str, str],
    ) -> Optional[str]:
        """Return error text if required fragment files are missing/mismatched.

        ``outputs_map`` is optional; when provided, claimed paths must match
        the declared config paths for strict fields.
        """
        required = self._strict_file_fields(rendered_decls)
        if not required:
            return None

        problems: List[str] = []
        for field, expected_path in required.items():
            claimed = (outputs_map or {}).get(field)
            if claimed is not None and claimed.replace("\\", "/") != expected_path.replace(
                "\\", "/"
            ):
                problems.append(
                    f"- {field}: outputs map path '{claimed}' != declared '{expected_path}'"
                )
            abs_path = self.workdir / expected_path
            if not abs_path.exists():
                problems.append(
                    f"- {field}: file not found at '{expected_path}' "
                    "(write it before task.finish)"
                )
            elif abs_path.is_file() and abs_path.stat().st_size == 0:
                problems.append(f"- {field}: file '{expected_path}' is empty")

        if not problems:
            return None
        return "\n".join(problems)

    def _harvest_file_outputs(
        self,
        rendered_decls: dict,
        outputs_map: Dict[str, str],
        require_all: bool = True,
    ) -> Dict[str, str]:
        """Read declared output files from disk into raw_outputs values.

        Strict fragment fields must exist when require_all=True. Field ``all``
        is harvested only if the file already exists (otherwise final_result
        remains the primary text and write_stage_output creates the file).
        """
        declared = self._declared_file_fields(rendered_decls)
        if not declared:
            return {}

        if require_all:
            err = self._check_file_outputs(rendered_decls, outputs_map or {})
            if err is not None:
                raise RuntimeError(
                    f"Stage '{self.name}': cannot harvest declared file outputs:\n{err}"
                )

        harvested: Dict[str, str] = {}
        for field, rel_path in declared.items():
            abs_path = self.workdir / rel_path
            if not abs_path.exists():
                # ``all`` may be filled from task.finish result only.
                if field == "all" or not require_all:
                    continue
                raise RuntimeError(
                    f"Stage '{self.name}': missing harvested file for "
                    f"'{field}': {abs_path}"
                )
            harvested[field] = abs_path.read_text(encoding="utf-8")
        return harvested

    def _approval_preview(
        self,
        submitted: str,
        rendered_decls: dict,
        outputs_map: Dict[str, str],
    ) -> str:
        """Build human-review text including multi-file snippets."""
        parts = [submitted if isinstance(submitted, str) else str(submitted)]
        declared = self._declared_file_fields(rendered_decls)
        # Prefer showing strict fragments; include all if it was written.
        show = {
            k: v
            for k, v in declared.items()
            if k != "all" or (self.workdir / v).exists()
        }
        if not show:
            return parts[0]

        parts.append("\n\n--- declared file outputs ---")
        for field, rel_path in show.items():
            abs_path = self.workdir / rel_path
            claimed = outputs_map.get(field, rel_path)
            if not abs_path.exists():
                parts.append(f"\n[{field}] {claimed}  MISSING")
                continue
            text = abs_path.read_text(encoding="utf-8")
            preview = text if len(text) <= 1200 else (
                text[:600] + f"\n...[{len(text) - 1200} chars omitted]...\n" + text[-600:]
            )
            parts.append(
                f"\n[{field}] {rel_path} ({len(text)} chars)\n{preview}"
            )
        return "".join(parts)

    def _save_transcript_fallback(self, transcript: dict) -> None:
        log_dir = self.workdir / "gen" / "llm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / f"{self.name.replace('.', '_')}_transcript.json"
        path.write_text(
            json.dumps(transcript, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _request_human_approval(self, submitted: str) -> Tuple[bool, str]:
        """Block for human approve/reject of a task.finish submission.

        Returns (approved, feedback). feedback is empty on approve (unless
        the operator typed more than the keyword); on reject it is the
        remaining text after 'reject' or the full free-form reply.
        """
        preview = submitted if len(submitted) <= 4000 else (
            submitted[:2000]
            + f"\n\n...[{len(submitted) - 4000} chars omitted]...\n\n"
            + submitted[-2000:]
        )
        print("\n" + "=" * 60)
        print(f"[agent:{self.name}] human approval required")
        print("The agent submitted the following result for review:")
        print("-" * 60)
        print(preview)
        print("-" * 60)
        print(
            "Reply with: approve | a | yes | y   to accept and end the stage\n"
            "            reject | r | no | n [feedback...]  to send back\n"
            "Or type free-form feedback (treated as reject)."
        )
        print("=" * 60)

        if not sys.stdin.isatty():
            raise RuntimeError(
                f"Stage '{self.name}': require_approval=true needs an interactive "
                "TTY stdin to approve/reject the agent submission. Re-run in a "
                "terminal, or set require_approval=false for non-interactive runs."
            )

        try:
            answer = input("Approval> ").strip()
        except EOFError as e:
            raise RuntimeError(
                f"Stage '{self.name}': stdin closed during human approval"
            ) from e

        if not answer:
            # empty → re-prompt once, then treat as reject with no feedback
            try:
                answer = input("Approval (approve/reject)> ").strip()
            except EOFError as e:
                raise RuntimeError(
                    f"Stage '{self.name}': stdin closed during human approval"
                ) from e
            if not answer:
                print(f"[agent] {self.name} — empty approval reply; treating as reject")
                return False, "(empty reply)"

        lower = answer.lower()
        tokens = lower.split(None, 1)
        head = tokens[0] if tokens else ""
        rest = tokens[1] if len(tokens) > 1 else ""

        approve_words = {"approve", "a", "yes", "y", "ok", "accept"}
        reject_words = {"reject", "r", "no", "n", "deny", "revise"}

        if head in approve_words:
            return True, rest
        if head in reject_words:
            return False, rest or answer
        # free-form text = reject with that feedback
        return False, answer


def _preview_args(raw_args: Any, limit: int = 120) -> str:
    if raw_args is None:
        return ""
    text = raw_args if isinstance(raw_args, str) else json.dumps(raw_args, ensure_ascii=False)
    text = text.replace("\n", " ")
    if len(text) > limit:
        return text[:limit] + "..."
    return text
