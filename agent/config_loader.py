"""
config_loader.py — 解析 llm_config.toml，验证结构，返回 Config 对象。

llm_config.toml 结构示例：
  [ref]
  resonances_config = {type = "file", path = "resonances_config.toml"}

  [stages.config_strip]
  kind = "python"
  handler = "gen/handlers/config_strip_handler.py"
  [stages.config_strip.input]
  resonances_config = {type = "file", path = "resonances_config.toml"}
  [stages.config_strip.output]
  free_params   = {type = "file", path = "run/free_params.toml"}
  stripped_config = {type = "file", path = "gen/fragments/stripped_config.toml"}
  all_sbc       = {type = "direct"}
  all_amp       = {type = "direct"}

  [stages.classification]
  kind = "llm"
  prompt = {type = "file", path = "gen/prompts/classification_prompt.txt"}
  [stages.classification.output]
  all = {type = "file", path = "gen/fragments/classification.json"}

  [stages.resonance_calculation]
  kind = "foreach"
  foreach_source = "stages.classification.all.propagator_classification"
  prompt = {type = "file", path = "gen/prompts/calculate_function.txt"}
  [stages.resonance_calculation.output]
  all = {type = "file", path = "gen/fragments/resonance_<<key>>.py"}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import toml


class ConfigError(Exception):
    pass


def _foreach_to_dotpath(foreach_raw: dict) -> Optional[str]:
    """把 foreach dict 转换为 resolver 可解析的 dotpath。

    支持的形式：
      {type="ref", path="stages.classification.propagator_classification"}
      {type="file", path="...", field="propagator_classification"}
      {type="list", values=["a","b"]}  → 返回 None，由调用方直接用 values
    """
    t = foreach_raw.get("type", "ref")
    if t == "ref":
        return foreach_raw.get("path")
    if t == "file":
        # 用 resolver 的 file-field 穿透语法暂不支持，暂时记为 None
        return None
    if t == "list":
        return None  # values 直接存在 foreach_raw["values"]
    return foreach_raw.get("path")


class StageConfig:
    """单个 stage 的配置。"""

    def __init__(self, name: str, raw: dict) -> None:
        self.name = name
        self.raw = raw
        self.kind: str = raw.get("kind", "")

        # handler 可以是字符串路径，也可以是 {type, path, function} dict
        handler_raw = raw.get("handler")
        if isinstance(handler_raw, dict):
            self.handler: Optional[str] = handler_raw.get("path")
            self.handler_function: Optional[str] = handler_raw.get("function")
        else:
            self.handler = handler_raw
            self.handler_function = None

        self.prompt: Optional[Any] = raw.get("prompt")
        self.human_prompt: Optional[str] = raw.get("human_prompt")

        # foreach 可以是字符串（foreach_source）或 dict（{type, path/values, field?}）
        foreach_raw = raw.get("foreach") or raw.get("foreach_source")
        self.foreach_raw = foreach_raw  # 保留原始值供 stage_runner 用
        if isinstance(foreach_raw, dict):
            self.foreach_source: Optional[str] = _foreach_to_dotpath(foreach_raw)
            self.foreach_values = foreach_raw.get("values")  # list 模式直接取值
        else:
            self.foreach_source = foreach_raw
            self.foreach_values = None

        self.input = raw.get("input", {})
        self.output: dict = raw.get("output", {})
        self.check: bool = raw.get("check", False)

    def __repr__(self) -> str:
        return f"StageConfig({self.name!r}, kind={self.kind!r})"


class Config:
    """解析后的 llm_config.toml。"""

    def __init__(self, raw: dict, path: Path) -> None:
        self.path = path
        self.raw = raw
        self.ref: dict = raw.get("ref", {})
        self._stages_raw: dict = raw.get("stages", {})
        self._stages: Dict[str, StageConfig] = {
            name: StageConfig(name, cfg)
            for name, cfg in self._stages_raw.items()
        }

    @property
    def stages(self) -> Dict[str, StageConfig]:
        return self._stages

    def stage_names_in_order(self) -> List[str]:
        return list(self._stages.keys())

    def get_stage(self, name: str) -> Optional[StageConfig]:
        return self._stages.get(name)

    def items(self) -> Iterator[Tuple[str, StageConfig]]:
        return iter(self._stages.items())

    def __repr__(self) -> str:
        return f"Config(path={self.path}, stages={list(self._stages.keys())})"


class ConfigLoader:
    @staticmethod
    def load(path: Path) -> Config:
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        try:
            raw = toml.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            raise ConfigError(f"Failed to parse {path}: {e}") from e
        ConfigLoader._validate_structure(raw, path)
        return Config(raw, path)

    @staticmethod
    def _validate_structure(raw: dict, path: Path) -> None:
        if "stages" not in raw:
            raise ConfigError(f"{path}: missing top-level [stages] table")
        for name, stage in raw["stages"].items():
            if not isinstance(stage, dict):
                raise ConfigError(f"{path}: stage '{name}' must be a table, got {type(stage)}")
            kind = stage.get("kind", "")
            if kind not in ("python", "llm"):
                raise ConfigError(
                    f"{path}: stage '{name}' has invalid kind={kind!r}. "
                    f"Expected one of: python, llm"
                )
            if kind == "python" and not stage.get("handler"):
                raise ConfigError(f"{path}: stage '{name}' (python) missing 'handler'")
            if kind == "llm" and not stage.get("prompt"):
                raise ConfigError(f"{path}: stage '{name}' (llm) missing 'prompt'")
