"""
config_accessor.py — python stage handler 通过这个对象访问配置，屏蔽 file 穿透细节。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from manifest import _read_file
from resolver import Resolver


class ConfigAccessor:
    def __init__(self, stage_cfg, resolver: Resolver, workdir: Path) -> None:
        self.stage = stage_cfg
        self.resolver = resolver
        self.workdir = workdir

    def read(self, field_descriptor: Any) -> Any:
        """
        field_descriptor 是 llm_config 里的字段定义，例如：
          {"type": "file", "path": "resonances_config.toml"}
          {"type": "ref", "path": "stages.config_strip.free_params"}
          "some_direct_value"
        """
        if not isinstance(field_descriptor, dict):
            return field_descriptor

        fd_type = field_descriptor.get("type", "direct")

        if fd_type == "file":
            path = self.workdir / field_descriptor["path"]
            if not path.exists():
                raise FileNotFoundError(
                    f"ConfigAccessor.read: file not found: {path}"
                )
            return _read_file(path)

        if fd_type == "ref":
            return self.resolver.resolve(field_descriptor["path"])

        return field_descriptor

    def read_input(self, field_name: str) -> Any:
        """从 stage.input 中读取指定字段。"""
        if field_name not in self.stage.input:
            raise KeyError(
                f"Stage '{self.stage.name}': input field '{field_name}' not declared. "
                f"Available: {list(self.stage.input.keys())}"
            )
        return self.read(self.stage.input[field_name])
