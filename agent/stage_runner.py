"""
stage_runner.py — 执行单个 stage（python / llm / foreach）。
"""

from __future__ import annotations

import hashlib
import importlib.util
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

from config_loader import StageConfig
from manifest import Manifest
from resolver import Resolver


def _strip_fences(code: str) -> str:
    code = re.sub(r"^```\w*\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code.strip())
    return code


def _compute_hash(*parts: str) -> str:
    combined = "\n---\n".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


class StageRunner:
    def __init__(
        self,
        name: str,
        stage_cfg: StageConfig,
        engine: Any,
        foreach_key: Optional[str] = None,
    ) -> None:
        self.name = name
        self.stage_cfg = stage_cfg
        self.engine = engine
        self.foreach_key = foreach_key

        self.workdir: Path = engine.workdir
        self.manifest: Manifest = engine.manifest
        self.resolver: Resolver = (
            engine.resolver.with_key(foreach_key)
            if foreach_key is not None
            else engine.resolver
        )

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def execute(self) -> None:
        kind = self.stage_cfg.kind
        if kind == "python":
            self._execute_python()
        elif kind == "llm":
            self._execute_llm()
        elif kind == "foreach":
            self._execute_foreach()
        else:
            raise ValueError(f"Unknown stage kind: {kind!r}")

    # ------------------------------------------------------------------
    # python stage
    # ------------------------------------------------------------------

    def _execute_python(self) -> None:
        handler_path = self.workdir / self.stage_cfg.handler
        handler_fn = _load_handler(handler_path)

        from config_accessor import ConfigAccessor
        cfg_accessor = ConfigAccessor(self.stage_cfg, self.resolver, self.workdir)

        print(f"[python] {self.name}")
        result = handler_fn(cfg_accessor, self.resolver)

        self.manifest.write_stage_output(
            stage=self.name,
            outputs=result,
            output_decls=self.stage_cfg.output,
            workdir=self.workdir,
            prompt_hash="",
        )

    # ------------------------------------------------------------------
    # llm stage
    # ------------------------------------------------------------------

    def _execute_llm(self) -> None:
        prompt = self._build_prompt()
        prompt_hash = _compute_hash(prompt)

        if self.manifest.is_cached(self.name, prompt_hash, self.workdir):
            print(f"[llm] {self.name} — cache hit")
            return

        print(f"[llm] {self.name} — calling LLM (prompt {len(prompt)} chars)")
        code = self.engine.llm_client.call(prompt, check=self.stage_cfg.check)
        time.sleep(1)

        self.manifest.write_stage_output(
            stage=self.name,
            outputs={"all": code},
            output_decls=self.stage_cfg.output,
            workdir=self.workdir,
            prompt_hash=prompt_hash,
        )
        self.manifest.save_llm_log(self.name, prompt, code, self.workdir)

    # ------------------------------------------------------------------
    # foreach stage
    # ------------------------------------------------------------------

    def _execute_foreach(self) -> None:
        source = self.stage_cfg.foreach_source
        keys_or_dict = self.resolver.resolve(source)

        if isinstance(keys_or_dict, dict):
            keys = list(keys_or_dict.keys())
        elif isinstance(keys_or_dict, list):
            keys = keys_or_dict
        else:
            raise ValueError(
                f"foreach_source '{source}' resolved to {type(keys_or_dict).__name__}, "
                f"expected dict or list"
            )

        if not keys:
            raise ValueError(f"foreach_source '{source}' is empty")

        print(f"[foreach] {self.name} — keys: {keys}")
        for key in keys:
            sub_runner = StageRunner(
                name=f"{self.name}.{key}",
                stage_cfg=self.stage_cfg,
                engine=self.engine,
                foreach_key=key,
            )
            sub_runner._execute_llm()

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        prompt_decl = self.stage_cfg.prompt
        if isinstance(prompt_decl, dict) and prompt_decl.get("type") == "file":
            prompt_path = self.workdir / prompt_decl["path"]
            template = prompt_path.read_text(encoding="utf-8")
        elif isinstance(prompt_decl, str):
            template = prompt_decl
        else:
            raise ValueError(f"Stage '{self.name}': invalid prompt declaration: {prompt_decl}")

        rendered = self.resolver.render(template)

        if self.stage_cfg.human_prompt:
            human = self.resolver.render(self.stage_cfg.human_prompt)
            rendered = rendered + "\n\n" + human

        return rendered


# ---------------------------------------------------------------------------
# Handler loader
# ---------------------------------------------------------------------------

def _load_handler(path: Path):
    spec = importlib.util.spec_from_file_location("_handler", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn_name = path.stem  # e.g. config_strip_handler
    if not hasattr(mod, fn_name):
        raise AttributeError(
            f"Handler file '{path}' has no function named '{fn_name}'. "
            f"Handler function must match the filename (without .py)."
        )
    return getattr(mod, fn_name)
