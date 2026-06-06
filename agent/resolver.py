"""
resolver.py — <<...>> 引用解析，manifest 路径穿透。

支持的语法：
  <<ref.resonances_config>>                       → config.ref 中的 file 类型字段，返回文件内容
  <<ref.code_template.SECTION_NAME>>              → template 文件中 # SECTION: xxx 片段
  <<stages.config_strip.all_sbc>>                 → manifest 中 direct 类型字段
  <<stages.config_strip.stripped_config>>         → manifest 中 file 类型字段，返回文件全文
  <<stages.config_strip.stripped_config.resonances>> → 读文件后取内部字段
  <<stages.resonance_calculation[*].all>>         → foreach stage 所有 key 的输出，按序拼接
  <<stages.resonance_calculation[BW_BW].all>>     → foreach stage 特定 key 的输出
  <<key>>                                         → foreach 子 stage 注入的当前 key

render(template) 把模板中所有 <<...>> 替换为解析结果。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from manifest import Manifest, _read_file, FieldError


class ResolutionError(Exception):
    pass


# 解析 <<...>> 的正则
_REF_RE = re.compile(r"<<([^>]+)>>")

# 解析路径中的 [key] 或 [*] 片段
_BRACKET_RE = re.compile(r"\[([^\]]*)\]")


class Resolver:
    def __init__(
        self,
        workdir: Path,
        manifest: Manifest,
        config,  # Config 对象
        foreach_key: Optional[str] = None,
    ) -> None:
        self.workdir = workdir
        self.manifest = manifest
        self.config = config
        self.foreach_key = foreach_key  # foreach 子 stage 注入的当前 key

    def with_key(self, key: str) -> "Resolver":
        """返回一个注入了 foreach_key 的新 Resolver。"""
        return Resolver(self.workdir, self.manifest, self.config, foreach_key=key)

    # ------------------------------------------------------------------
    # 对外 API
    # ------------------------------------------------------------------

    def resolve(self, dotpath: str) -> Any:
        """解析单个 <<dotpath>> 表达式，返回值。"""
        # 特殊变量 <<key>>
        if dotpath.strip() == "key":
            if self.foreach_key is None:
                raise ResolutionError("<<key>> used outside a foreach context")
            return self.foreach_key

        parts = _parse_path(dotpath)
        if not parts:
            raise ResolutionError(f"Empty path: '{dotpath}'")

        root_key = parts[0]

        # 处理 stages[*].all 或 stages[key].all 语法（root 本身带 bracket）
        _bracket_in_root = _BRACKET_RE.search(root_key)
        if _bracket_in_root and root_key[: _bracket_in_root.start()] == "stages":
            return self._resolve_stages(parts, dotpath)

        if root_key == "ref":
            return self._resolve_ref(parts[1:], dotpath)
        elif root_key == "stages":
            return self._resolve_stages(parts[1:], dotpath)
        else:
            raise ResolutionError(
                f"Unknown root '{root_key}' in '<<{dotpath}>>'. "
                f"Expected 'ref' or 'stages'."
            )

    def render(self, template: str) -> str:
        """替换模板中所有 <<...>> 占位符。"""
        def _replace(m: re.Match) -> str:
            dotpath = m.group(1).strip()
            try:
                value = self.resolve(dotpath)
            except ResolutionError as e:
                raise ResolutionError(
                    f"Failed to render '<<{dotpath}>>': {e}"
                ) from e
            if isinstance(value, list):
                return "\n\n".join(str(v) for v in value)
            if isinstance(value, dict):
                import json
                return json.dumps(value, indent=2, ensure_ascii=False)
            return str(value) if value is not None else ""

        return _REF_RE.sub(_replace, template)

    # ------------------------------------------------------------------
    # ref 解析
    # ------------------------------------------------------------------

    def _resolve_ref(self, parts: List[str], full_path: str) -> Any:
        if not parts:
            raise ResolutionError(f"'ref' needs at least one more segment in '{full_path}'")

        ref_key = parts[0]
        ref_decl = self.config.ref.get(ref_key)
        if ref_decl is None:
            raise ResolutionError(
                f"'ref.{ref_key}' not found in config [ref] table. "
                f"Available: {list(self.config.ref.keys())}"
            )

        # ref 条目可以是 {type="file", path="..."} / {type="template", path="..."} / 列表 / 字符串
        if isinstance(ref_decl, list):
            # 列表：多个模板文件，合并所有文件的 sections，按 section_name 查找
            if len(parts) < 2:
                raise ResolutionError(
                    f"Template ref '{ref_key}' (list) requires a section name: "
                    f"<<ref.{ref_key}.SECTION_NAME>>"
                )
            section_name = parts[1]
            merged: Dict[str, str] = {}
            for raw_path in ref_decl:
                ref_path = self.workdir / raw_path
                if not ref_path.exists():
                    raise ResolutionError(f"ref template file '{ref_path}' does not exist")
                merged.update(_parse_template_sections(ref_path))
            if section_name not in merged:
                raise ResolutionError(
                    f"Section '{section_name}' not found in any template in ref '{ref_key}'. "
                    f"Available: {list(merged.keys())}"
                )
            return merged[section_name]

        if isinstance(ref_decl, dict):
            ref_type = ref_decl.get("type", "file")
            ref_path = self.workdir / ref_decl["path"]

            if ref_type == "template":
                # 从 template 文件取 # SECTION: xxx 片段
                if len(parts) < 2:
                    raise ResolutionError(
                        f"Template ref '{ref_key}' requires a section name: "
                        f"<<ref.{ref_key}.SECTION_NAME>>"
                    )
                section_name = parts[1]
                sections = _parse_template_sections(ref_path)
                if section_name not in sections:
                    raise ResolutionError(
                        f"Section '{section_name}' not found in template '{ref_path}'. "
                        f"Available: {list(sections.keys())}"
                    )
                return sections[section_name]

            # type == "file"
            if not ref_path.exists():
                raise ResolutionError(f"ref file '{ref_path}' does not exist")
            node = _read_file(ref_path)
            # 继续按剩余 parts 穿透
            for part in parts[1:]:
                if isinstance(node, dict):
                    if part not in node:
                        raise ResolutionError(
                            f"Key '{part}' not found in file '{ref_path}'. "
                            f"Available: {list(node.keys())}"
                        )
                    node = node[part]
                else:
                    raise ResolutionError(
                        f"Cannot descend into non-dict at '{part}' in ref '{ref_key}'"
                    )
            return node

        # ref 条目是字符串（直接值）
        return ref_decl

    # ------------------------------------------------------------------
    # stages 解析
    # ------------------------------------------------------------------

    def _resolve_stages(self, parts: List[str], full_path: str) -> Any:
        if not parts:
            raise ResolutionError(f"'stages' needs at least one more segment in '{full_path}'")

        # 处理 foreach [*] 或 [key] 语法：stages.resonance_calculation[*].all
        # parts 此时可能是 ["resonance_calculation[*]", "all"] 或 ["resonance_calculation[BW_BW]", "all"]
        stage_part = parts[0]
        bracket_m = _BRACKET_RE.search(stage_part)

        if bracket_m:
            stage_name = stage_part[: bracket_m.start()]
            bracket_key = bracket_m.group(1)  # "*" 或具体 key
            remaining = parts[1:]

            # stages[*].field —— 遍历 manifest 中所有 stage
            if stage_name == "stages" and bracket_key == "*":
                all_stages = self.manifest.get("stages") or {}
                results = []
                for sname, snode in all_stages.items():
                    if not isinstance(snode, dict):
                        continue
                    try:
                        val = self._descend_node(snode, remaining, f"stages.{sname}", self.workdir)
                        results.append(val)
                    except ResolutionError:
                        pass
                return results

            stage_node = self.manifest.get("stages", stage_name)
            if stage_node is None:
                raise ResolutionError(
                    f"Stage '{stage_name}' not found in manifest. "
                    f"Make sure it ran before this stage."
                )

            if bracket_key == "*":
                # 返回所有子 key 的 remaining 路径值，按序拼接
                results = []
                for sub_key, sub_node in stage_node.items():
                    if not isinstance(sub_node, dict):
                        continue
                    val = self._descend_node(sub_node, remaining, f"stages.{stage_name}.{sub_key}", self.workdir)
                    results.append(val)
                return results
            else:
                sub_node = stage_node.get(bracket_key)
                if sub_node is None:
                    raise ResolutionError(
                        f"foreach key '{bracket_key}' not found in stage '{stage_name}'. "
                        f"Available keys: {list(stage_node.keys())}"
                )
                return self._descend_node(sub_node, remaining, f"stages.{stage_name}.{bracket_key}", self.workdir)

        # 普通 stage 引用：stages.config_strip.stripped_config 或 stages.config_strip.stripped_config.resonances
        stage_name = stage_part
        if len(parts) < 2:
            raise ResolutionError(
                f"Stage reference needs field name: '<<stages.{stage_name}.field_name>>'"
            )
        field_name = parts[1]
        sub_parts = parts[2:]

        # 从 manifest 读取字段
        try:
            value = self.manifest.read_field(stage_name, field_name, self.workdir)
        except FieldError as e:
            raise ResolutionError(str(e)) from e

        # 继续按剩余 parts 穿透（文件内字段）
        for part in sub_parts:
            if isinstance(value, dict):
                if part not in value:
                    raise ResolutionError(
                        f"Key '{part}' after resolving 'stages.{stage_name}.{field_name}'. "
                        f"Available: {list(value.keys())}"
                    )
                value = value[part]
            else:
                raise ResolutionError(
                    f"Cannot descend into non-dict at '{part}' in 'stages.{stage_name}.{field_name}'"
                )

        return value

    def _descend_node(self, node: dict, parts: List[str], ctx: str, workdir: Path) -> Any:
        """在一个已知的 manifest stage 子节点中按 parts 取值，透明加载 file 类型。"""
        if not parts:
            return node

        field = parts[0]
        remaining = parts[1:]

        if field not in node:
            raise ResolutionError(
                f"Field '{field}' not found in {ctx}. Available: {list(node.keys())}"
            )
        value = node[field]

        # 透明加载 file 类型
        if isinstance(value, dict) and value.get("type") == "file":
            file_path = workdir / value["path"]
            if not file_path.exists():
                raise ResolutionError(
                    f"File '{file_path}' referenced by {ctx}.{field} does not exist."
                )
            value = _read_file(file_path)

        for part in remaining:
            if isinstance(value, dict):
                if part not in value:
                    raise ResolutionError(
                        f"Key '{part}' not found after {ctx}.{field}. "
                        f"Available: {list(value.keys())}"
                    )
                value = value[part]
            else:
                raise ResolutionError(
                    f"Cannot descend into non-dict at '{part}' in {ctx}.{field}"
                )

        return value


# ---------------------------------------------------------------------------
# Path parsing helpers
# ---------------------------------------------------------------------------

def _parse_path(dotpath: str) -> List[str]:
    """
    把 dotpath 拆成段列表，处理 [key] 语法。
    "stages.resonance_calculation[*].all" →
        ["stages", "resonance_calculation[*]", "all"]
    """
    # 先把 [key] 粘在前一段（不拆分），再按 . 分割
    parts = []
    for raw_part in dotpath.split("."):
        raw_part = raw_part.strip()
        if raw_part:
            parts.append(raw_part)
    return parts


# ---------------------------------------------------------------------------
# Template section parser
# ---------------------------------------------------------------------------

_SECTION_CACHE: Dict[str, Dict[str, str]] = {}


def _parse_template_sections(path: Path) -> Dict[str, str]:
    key = str(path)
    if key in _SECTION_CACHE:
        return _SECTION_CACHE[key]

    sections: Dict[str, str] = {}
    current: Optional[str] = None
    buf: List[str] = []

    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if line.strip().startswith("# SECTION:"):
            if current:
                sections[current] = "".join(buf)
            current = line.strip().split("SECTION:")[1].strip()
            buf = []
        elif current:
            buf.append(line)

    if current:
        sections[current] = "".join(buf)

    _SECTION_CACHE[key] = sections
    return sections
