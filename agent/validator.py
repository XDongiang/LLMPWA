"""
validator.py — 静态依赖检查。

第一轮（静态，启动时）：
  - handler / prompt 引用的文件是否存在
  - <<stages.X.Y>> 中 X 是否在 config 中定义、是否在当前 stage 之前
  - <<stages.X.Y>> 中 Y 是否在 stage X 的 output 声明里
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

from config_loader import Config, ConfigError, StageConfig


_REF_RE = re.compile(r"<<([^>]+)>>")
_BRACKET_RE = re.compile(r"\[([^\]]*)\]")


class Validator:
    def __init__(self, config: Config, workdir: Path) -> None:
        self.config = config
        self.workdir = workdir

    def static_check(self) -> None:
        stage_names = self.config.stage_names_in_order()
        errors: List[str] = []

        for i, (name, stage) in enumerate(self.config.items()):
            preceding = set(stage_names[:i])
            self._check_stage_files(name, stage, errors)
            self._check_stage_refs(name, stage, preceding, errors)

        if errors:
            msg = "\n".join(f"  - {e}" for e in errors)
            raise ConfigError(f"Static validation failed:\n{msg}")

    def _check_stage_files(self, name: str, stage: StageConfig, errors: List[str]) -> None:
        if stage.kind == "python" and stage.handler:
            handler_path = self.workdir / stage.handler
            if not handler_path.exists():
                errors.append(f"Stage '{name}': handler file not found: {handler_path}")

        if isinstance(stage.prompt, dict) and stage.prompt.get("type") == "file":
            prompt_path = self.workdir / stage.prompt["path"]
            if not prompt_path.exists():
                errors.append(f"Stage '{name}': prompt file not found: {prompt_path}")

        if isinstance(stage.system_prompt, dict) and stage.system_prompt.get("type") == "file":
            sp_path = self.workdir / stage.system_prompt["path"]
            if not sp_path.exists():
                errors.append(
                    f"Stage '{name}': system_prompt file not found: {sp_path}"
                )

        if stage.kind == "agent" and stage.tools is not None:
            try:
                from tools.registry import validate_tool_names

                for err in validate_tool_names(stage.tools):
                    errors.append(f"Stage '{name}': {err}")
            except Exception as e:
                errors.append(f"Stage '{name}': failed to validate tools: {e}")

    def _check_stage_refs(
        self, name: str, stage: StageConfig, preceding: set, errors: List[str]
    ) -> None:
        texts_to_scan: List[str] = []

        if isinstance(stage.prompt, dict) and stage.prompt.get("type") == "file":
            prompt_path = self.workdir / stage.prompt["path"]
            if prompt_path.exists():
                texts_to_scan.append(prompt_path.read_text(encoding="utf-8"))
        elif isinstance(stage.prompt, str):
            texts_to_scan.append(stage.prompt)

        if isinstance(stage.system_prompt, dict) and stage.system_prompt.get("type") == "file":
            sp_path = self.workdir / stage.system_prompt["path"]
            if sp_path.exists():
                texts_to_scan.append(sp_path.read_text(encoding="utf-8"))
        elif isinstance(stage.system_prompt, str):
            texts_to_scan.append(stage.system_prompt)

        if stage.human_prompt:
            texts_to_scan.append(stage.human_prompt)

        if stage.foreach_source:
            texts_to_scan.append(f"<<{stage.foreach_source}>>")

        for text in texts_to_scan:
            for m in _REF_RE.finditer(text):
                dotpath = m.group(1).strip()
                self._check_ref(name, dotpath, preceding, errors)

    def _check_ref(
        self, stage_name: str, dotpath: str, preceding: set, errors: List[str]
    ) -> None:
        if dotpath == "key":
            return

        parts = dotpath.split(".")
        if not parts or parts[0] not in ("stages", "ref"):
            return

        if parts[0] == "ref":
            ref_key = parts[1] if len(parts) > 1 else ""
            if ref_key and ref_key not in self.config.ref:
                errors.append(
                    f"Stage '{stage_name}': <<ref.{ref_key}>> — "
                    f"'{ref_key}' not found in [ref] table"
                )
            return

        # stages.X.Y
        if len(parts) < 3:
            return

        raw_stage = parts[1]
        bracket_m = _BRACKET_RE.search(raw_stage)
        ref_stage = raw_stage[: bracket_m.start()] if bracket_m else raw_stage
        ref_field = parts[2] if len(parts) > 2 else ""

        if ref_stage not in self.config.stages:
            errors.append(
                f"Stage '{stage_name}': <<{dotpath}>> — "
                f"referenced stage '{ref_stage}' is not defined in config"
            )
            return

        if ref_stage not in preceding:
            errors.append(
                f"Stage '{stage_name}': <<{dotpath}>> depends on '{ref_stage}', "
                f"but '{ref_stage}' is not defined before '{stage_name}' in llm_config.toml"
            )
            return

        if ref_field and ref_field != "all":
            ref_stage_cfg = self.config.get_stage(ref_stage)
            if ref_stage_cfg and ref_field not in ref_stage_cfg.output:
                errors.append(
                    f"Stage '{stage_name}': <<{dotpath}>> — "
                    f"field '{ref_field}' not declared in stage '{ref_stage}' output. "
                    f"Declared: {list(ref_stage_cfg.output.keys())}"
                )
