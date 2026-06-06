"""
engine.py — 通用主引擎，驱动 llm_config.toml，不包含任何物理分析逻辑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from config_loader import ConfigLoader, Config
from manifest import Manifest
from resolver import Resolver
from validator import Validator
from stage_runner import StageRunner
from llm_client import LLMClient


class Engine:
    def __init__(
        self,
        workdir: str,
        config_name: str = "llm_config_fit.toml",
        model: Optional[str] = None,
        model_check: Optional[str] = None,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.config: Config = ConfigLoader.load(self.workdir / config_name)

        gen_dir = self.workdir / "gen"
        gen_dir.mkdir(parents=True, exist_ok=True)

        # manifest 文件名跟随 config 文件名，例如 llm_config_test.toml → manifest_test.toml
        config_stem = Path(config_name).stem  # e.g. "llm_config_test"
        suffix = config_stem[len("llm_config"):]  # e.g. "_test" or "_fit" or ""
        manifest_name = f"manifest{suffix}.toml"
        self.manifest = Manifest(gen_dir / manifest_name)
        self.resolver = Resolver(self.workdir, self.manifest, self.config)
        self.validator = Validator(self.config, self.workdir)
        self.llm_client = LLMClient(model=model, model_check=model_check)

    # ------------------------------------------------------------------
    # 主流程
    # ------------------------------------------------------------------

    def run(self) -> None:
        print(f"Engine: workdir={self.workdir}")
        print(f"Stages: {self.config.stage_names_in_order()}")

        self.validator.static_check()
        print("Static check passed.")

        visible: set = set()
        for name, stage_cfg in self.config.items():
            self.resolver.current_stage = name
            self.resolver.visible_stages = visible
            runner = StageRunner(name, stage_cfg, self)
            runner.execute()
            # stage 完成后 reload manifest，并将其加入可见集合供后续 stage 引用
            self.manifest.reload()
            visible.add(name)

        print("All stages completed.")

    # ------------------------------------------------------------------
    # check-only 模式
    # ------------------------------------------------------------------

    def check(self) -> None:
        self.validator.static_check()
        print("Static check passed. No stages were executed.")
