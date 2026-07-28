"""
manifest.py — manifest.toml 读写，原子写入。

manifest 结构：
  [stages.<stage_name>]
  prompt_hash = "..."
  [stages.<stage_name>.all]
  type = "file"
  path = "gen/fragments/xxx.py"
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

import toml


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._data: dict = {}
        self._load()

    # ------------------------------------------------------------------
    # Load / Save
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self.path.exists():
            try:
                self._data = toml.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}
        else:
            self._data = {}

    def reload(self) -> None:
        self._load()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(_dump_manifest(self._data), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------------
    # Generic get / set
    # ------------------------------------------------------------------

    def get(self, *keys: str) -> Any:
        node = self._data
        for k in keys:
            if not isinstance(node, dict):
                return None
            node = node.get(k)
        return node

    def set(self, value: Any, *keys: str) -> None:
        d = self._data
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    # ------------------------------------------------------------------
    # Stage helpers
    # ------------------------------------------------------------------

    def get_stage(self, stage: str) -> Optional[dict]:
        parts = ["stages"] + stage.split(".")
        return self.get(*parts)

    def set_stage(self, stage: str, data: dict) -> None:
        parts = ["stages"] + stage.split(".")
        self.set(data, *parts)

    def update_stage(self, stage: str, data: dict) -> None:
        existing = self.get_stage(stage) or {}
        existing.update(data)
        self.set_stage(stage, existing)

    # ------------------------------------------------------------------
    # Cache check
    # ------------------------------------------------------------------

    def is_cached(
        self,
        stage: str,
        prompt_hash: str,
        workdir: Path,
        output_decls: Optional[dict] = None,
    ) -> bool:
        node = self.get_stage(stage)
        if not node:
            return False
        if node.get("prompt_hash") != prompt_hash:
            return False

        # Prefer validating every declared type=file path (multi-output stages).
        # Falls back to legacy single `all` check when decls not provided.
        file_paths: list = []
        if output_decls:
            for field, decl in output_decls.items():
                if not isinstance(decl, dict):
                    continue
                if decl.get("type") != "file":
                    continue
                path = decl.get("path")
                if path:
                    file_paths.append(str(path))
            if file_paths:
                return all((workdir / p).exists() for p in file_paths)

        # Legacy / partial: check every file pointer recorded in the node
        recorded_files = [
            v["path"]
            for v in node.values()
            if isinstance(v, dict) and v.get("type") == "file" and v.get("path")
        ]
        if recorded_files:
            return all((workdir / p).exists() for p in recorded_files)

        out = node.get("all")
        if isinstance(out, dict) and out.get("type") == "file":
            return (workdir / out["path"]).exists()
        return out is not None

    def load_cached(self, stage: str) -> Any:
        node = self.get_stage(stage)
        if node:
            return node.get("all")
        return None

    # ------------------------------------------------------------------
    # Write stage outputs
    # ------------------------------------------------------------------

    def write_stage_output(
        self,
        stage: str,
        outputs: dict,
        output_decls: dict,
        workdir: Path,
        prompt_hash: str = "",
    ) -> None:
        """
        outputs     — {field: value} from handler / stage runner
        output_decls — {field: {type, path?, ...}} from llm_config stage.output
        workdir     — for resolving relative paths
        """
        node: dict = {"prompt_hash": prompt_hash}

        for field, value in outputs.items():
            decl = output_decls.get(field, {})
            out_type = decl.get("type", "direct")

            if out_type == "file":
                rel_path = decl["path"]
                abs_path = workdir / rel_path
                abs_path.parent.mkdir(parents=True, exist_ok=True)
                tmp = abs_path.with_suffix(abs_path.suffix + ".tmp")
                _write_file(abs_path, tmp, value, rel_path)
                node[field] = {"type": "file", "path": rel_path}
            else:
                node[field] = value

        self.update_stage(stage, node)
        self.save()

    # ------------------------------------------------------------------
    # LLM log
    # ------------------------------------------------------------------

    def save_llm_log(
        self, stage: str, prompt: str, response: str, workdir: Path
    ) -> None:
        log_dir = workdir / "gen" / "llm_logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{stage.replace('.', '_')}.md"
        content = (
            f"# Stage: {stage}\n\n"
            f"## Prompt\n\n```\n{prompt}\n```\n\n"
            f"## Response\n\n```\n{response}\n```\n"
        )
        log_path.write_text(content, encoding="utf-8")

    # ------------------------------------------------------------------
    # Read a stage field (with file transparent loading)
    # ------------------------------------------------------------------

    def read_field(self, stage: str, field: str, workdir: Path) -> Any:
        node = self.get_stage(stage)
        if node is None:
            raise FieldError(
                f"Stage '{stage}' not found in manifest.\n"
                f"  Make sure the stage ran successfully before this one."
            )
        if field not in node:
            raise FieldError(
                f"Stage '{stage}' field '{field}' not in manifest.\n"
                f"  Available fields: {list(node.keys())}\n"
                f"  Check gen/llm_logs/{stage}.md for the LLM response."
            )
        value = node[field]
        if value is None:
            raise FieldError(
                f"Stage '{stage}' field '{field}' is None in manifest.\n"
                f"  Check gen/llm_logs/{stage}.md for the LLM response."
            )
        if isinstance(value, dict) and value.get("type") == "file":
            file_path = workdir / value["path"]
            if not file_path.exists():
                raise FieldError(
                    f"Stage '{stage}' field '{field}' points to '{file_path}' which does not exist."
                )
            return _read_file(file_path)
        return value

    @property
    def data(self) -> dict:
        return self._data


# ---------------------------------------------------------------------------
# File I/O helpers
# ---------------------------------------------------------------------------

def _write_file(abs_path: Path, tmp: Path, value: Any, rel_path: str) -> None:
    suffix = abs_path.suffix
    if suffix == ".toml":
        import toml as _toml
        tmp.write_text(_toml.dumps(value if isinstance(value, dict) else {"data": value}), encoding="utf-8")
    elif suffix == ".json":
        import json
        tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    elif suffix in (".py", ".txt", ".md"):
        tmp.write_text(value if isinstance(value, str) else str(value), encoding="utf-8")
    else:
        tmp.write_text(str(value), encoding="utf-8")
    tmp.replace(abs_path)


def _read_file(path: Path) -> Any:
    suffix = path.suffix
    text = path.read_text(encoding="utf-8")
    if suffix == ".toml":
        return toml.loads(text)
    if suffix == ".json":
        import json
        return json.loads(text)
    return text


# ---------------------------------------------------------------------------
# TOML serialization for manifest
# ---------------------------------------------------------------------------

def _scalar(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(_scalar(i) for i in v) + "]"
    escaped = str(v).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _inline_table(d: dict) -> str:
    pairs = ", ".join(f"{k} = {_scalar(v)}" for k, v in d.items())
    return "{ " + pairs + " }"


def _emit_stage(lines: list, header: str, node: dict) -> None:
    lines.append(f"[{header}]")
    for k, v in node.items():
        if isinstance(v, dict):
            lines.append(f"{k} = {_inline_table(v)}")
        elif isinstance(v, list):
            lines.append(f"{k} = {_scalar(v)}")
        else:
            lines.append(f"{k} = {_scalar(v)}")
    lines.append("")


def _walk(lines: list, prefix: str, node: dict) -> None:
    if "prompt_hash" in node or not any(isinstance(v, dict) for v in node.values()):
        _emit_stage(lines, prefix, node)
    else:
        for k, v in node.items():
            if isinstance(v, dict):
                _walk(lines, f"{prefix}.{k}", v)


def _dump_manifest(data: dict) -> str:
    lines: list = []
    for k, v in data.items():
        if k == "stages":
            continue
        lines.append(f"{k} = {_scalar(v)}")
    lines.append("")
    for stage_name, stage_data in data.get("stages", {}).items():
        if isinstance(stage_data, dict):
            _walk(lines, f"stages.{stage_name}", stage_data)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class FieldError(Exception):
    pass
