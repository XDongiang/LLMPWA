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

        self.manifest = Manifest(gen_dir / "manifest.toml")
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

        for name, stage_cfg in self.config.items():
            runner = StageRunner(name, stage_cfg, self)
            runner.execute()
            # foreach 完成后 manifest 已更新，reload 供后续 stage 使用
            self.manifest.reload()

        print("All stages completed.")

    # ------------------------------------------------------------------
    # check-only 模式
    # ------------------------------------------------------------------

    def check(self) -> None:
        self.validator.static_check()
        print("Static check passed. No stages were executed.")
