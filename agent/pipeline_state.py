"""
pipeline_state.py — LLMPWA 流水线状态导出器（供 DSH client-plugin 可视化消费）。

路线 B 的关键数据资产：把 LLMPWA 流水线的"静态结构 + 动态状态"抽成一份
结构化 JSON 快照，让 DeepSeek Harness 的 host 侧插件直接读取，而无需在
TypeScript 里重写 LLMPWA 的 TOML 解析逻辑。

产出内容（覆盖可视化需要的四类信息）：
  1. stage DAG 依赖图
  2. 每 stage 运行状态（manifest 中的 prompt_hash / cache 命中 / 输出）
  3. 产物代码（gen/fragments, gen/templates, run/fit_script.py, gen/llm_logs）
  4. 参数 / 共振态配置（resonances_config.toml）
  5. 实时增量所需的"可重算快照"：每次调用按当前磁盘状态重算，host 可轮询或 watch。

只读依赖：标准库 tomllib（Python >= 3.11），无需第三方 toml 包。

用法：
  python agent/pipeline_state.py --workdir analyses/kk_new \
      --config llm_config_fit.toml --out gen/pipeline_state.json

  # 作为模块使用：
  from pipeline_state import build_pipeline_state
  snapshot = build_pipeline_state(workdir="analyses/kk_new", config_name="llm_config_fit.toml")
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # pragma: no cover
    tomllib = None  # type: ignore


# ---------------------------------------------------------------------------
# TOML 读取（优先 tomllib，退化到 toml 包）
# ---------------------------------------------------------------------------

def _read_toml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    if tomllib is not None:
        with path.open("rb") as f:
            return tomllib.load(f)
    import toml  # type: ignore
    return toml.loads(path.read_text(encoding="utf-8"))


def _load_ref_file(workdir: Path, decl: Any) -> Optional[Dict[str, Any]]:
    """根据 [ref] 表里 {type="file", path="..."} 形态的声明加载 TOML 文件。

    返回解析后的 dict；声明不存在、不是 file 类型、或文件不存在时返回 None。
    """
    if not isinstance(decl, dict) or decl.get("type") not in ("file",):
        return None
    path = decl.get("path")
    if not isinstance(path, str):
        return None
    return _read_toml(workdir / path)


# ---------------------------------------------------------------------------
# 引用解析：从 <<...>> / stages.* 提取依赖
# ---------------------------------------------------------------------------

_REF_RE = re.compile(r"<<([^>]+)>>")
_BRACKET_RE = re.compile(r"\[([^\]]*)\]")


def _iter_refs(text: str) -> List[str]:
    """提取字符串里所有 <<...>> 引用表达式。"""
    return [m.group(1).strip() for m in _REF_RE.finditer(text)]


def _ref_target_stages(expr: str) -> List[str]:
    """判断一个引用表达式指向哪些 stage，返回 stage 名列表。

    覆盖的语法：
      stages.config_strip.stripped_config        → ['config_strip']
      stages.resonance_calculation[*].all        → ['*']（所有 foreach 子项）
      stages.resonance_calculation[BW_BW].all    -> ['resonance_calculation']
      stages[*].all                              → ['*']（所有 stage）
      其它（<<key>>、ref.* 等）                  → []
    """
    expr = expr.strip()
    if expr == "key":
        return []
    if not expr.startswith("stages"):
        return []
    # 取 stages 后方第一段
    rest = expr[len("stages"):]
    rest = rest.lstrip(".")
    parts = rest.split(".")
    first = parts[0]
    bracket = _BRACKET_RE.search(first)
    if bracket:
        stage_part = first[: bracket.start()]
        bracket_key = bracket.group(1)
        # stages[*] 表示所有 stage；stages.<name>[*] 表示 foreach 子项
        if stage_part == "" and bracket_key == "*":
            return ["*"]
        if stage_part:
            return [stage_part]
        return [bracket_key]
    if first:
        return [first]
    return []


# ---------------------------------------------------------------------------
# 依赖图构建
# ---------------------------------------------------------------------------

def _dependency_for_stage(name: str, stage_raw: Dict[str, Any], workdir: Path) -> Dict[str, Any]:
    """构建单个 stage 的依赖信息。

    收集来源：
      - prompt 文件正文 / prompt 字符串 / human_prompt / system_prompt 中的 <<...>>
      - foreach.source（字符串 dotpath 本身是一种引用）
      - input / output 的 path 字段，以及 list 值里的字符串项
        （如 assemble_final_code 的 input.list = ["ref.code_template.xxx", "stages[*].all"]）
    返回 {refs, deps, deps_all}：
      refs      — 该 stage 引用的原始表达式列表（含 <<...>> 与裸 dotpath）
      deps      — 命中的具体 stage 名（'*' 表示通配，另行处理）
      deps_all  — deps 中是否含 '*'（表示依赖所有 stage / 所有 foreach 子项）
    """
    refs: List[str] = []
    # 收集所有需要扫描的文本（含 <<...>> 包裹的引用，以及裸 dotpath 字段）
    scan_texts: List[str] = []
    bare_dotpaths: List[str] = []

    prompt = stage_raw.get("prompt")
    if isinstance(prompt, str):
        scan_texts.append(prompt)
    elif isinstance(prompt, dict) and prompt.get("type") == "file":
        p = workdir / prompt["path"]
        if p.exists():
            scan_texts.append(p.read_text(encoding="utf-8"))

    for key in ("human_prompt", "system_prompt"):
        v = stage_raw.get(key)
        if isinstance(v, str):
            scan_texts.append(v)

    # foreach source（字符串 dotpath）本身就是一种引用
    foreach_raw = stage_raw.get("foreach") or stage_raw.get("foreach_source")
    if isinstance(foreach_raw, str):
        bare_dotpaths.append(foreach_raw)
    elif isinstance(foreach_raw, dict):
        path = foreach_raw.get("path")
        if isinstance(path, str):
            bare_dotpaths.append(path)

    # input / output 的 path 字段，以及 list 值里的字符串项
    for section in ("input", "output"):
        decl = stage_raw.get(section, {})
        if isinstance(decl, dict):
            for field, d in decl.items():
                if isinstance(d, dict) and isinstance(d.get("path"), str):
                    bare_dotpaths.append(d["path"])
                elif isinstance(d, list):
                    for item in d:
                        if isinstance(item, str):
                            bare_dotpaths.append(item)
                        elif isinstance(item, dict) and isinstance(item.get("path"), str):
                            bare_dotpaths.append(item["path"])

    # 从扫描文本提取 <<...>> 引用
    for t in scan_texts:
        refs.extend(_iter_refs(t))
    # 裸 dotpath 直接作为引用
    refs.extend(bare_dotpaths)

    deps: List[str] = []
    deps_all = False
    for expr in refs:
        for target in _ref_target_stages(expr):
            if target == "*":
                deps_all = True
            elif target != name and target not in deps:
                deps.append(target)

    return {"refs": _dedupe(refs), "deps": deps, "deps_all": deps_all}


def _dedupe(seq: List[str]) -> List[str]:
    seen: List[str] = []
    for v in seq:
        if v not in seen:
            seen.append(v)
    return seen


# ---------------------------------------------------------------------------
# manifest 状态
# ---------------------------------------------------------------------------

def _build_manifest_state(workdir: Path, config_name: str, stage_names: List[str]) -> Dict[str, Any]:
    """定位并解析 manifest，返回每 stage 的运行状态。

    manifest 命名跟随 config：llm_config_fit.toml → manifest_fit.toml。
    也尝试裸 manifest.toml 作为兜底。
    """
    stem = Path(config_name).stem  # e.g. "llm_config_fit"
    suffix = stem[len("llm_config"):] if stem.startswith("llm_config") else ""
    candidates = [
        workdir / "gen" / f"manifest{suffix}.toml",
        workdir / "gen" / "manifest.toml",
        workdir / f"manifest{suffix}.toml",
        workdir / "manifest.toml",
    ]
    manifest_path = next((c for c in candidates if c.exists() and c.name.startswith("manifest")), None)

    info: Dict[str, Any] = {
        "path": str(manifest_path.relative_to(workdir)) if manifest_path else None,
        "present": manifest_path is not None,
        "stages": {},
    }
    if manifest_path is None:
        for name in stage_names:
            info["stages"][name] = _stage_missing()
        return info

    data = _read_toml(manifest_path)
    ms = data.get("stages", {}) or {}

    for name in stage_names:
        stage = ms.get(name)
        if stage is None:
            info["stages"][name] = _stage_missing()
            continue
        # 每个字段若是 {type: file, path} 转为可读形式
        outputs: Dict[str, Any] = {}
        for field, val in stage.items():
            if field in ("prompt_hash",):
                continue
            if isinstance(val, dict) and val.get("type") == "file":
                file_path = workdir / val["path"]
                outputs[field] = {
                    "type": "file",
                    "path": val["path"],
                    "exists": file_path.exists(),
                }
            else:
                outputs[field] = val

        prompt_hash = stage.get("prompt_hash")
        has_fragments = any(
            isinstance(v, dict) and v.get("type") == "file" and (workdir / v["path"]).exists()
            for v in stage.values()
        )
        info["stages"][name] = {
            "present": True,
            "status": "cached" if has_fragments else "ran",
            "prompt_hash": prompt_hash,
            "outputs": outputs,
        }
    return info


def _stage_missing() -> Dict[str, Any]:
    return {"present": False, "status": "missing", "prompt_hash": None, "outputs": {}}


# ---------------------------------------------------------------------------
# 产物文件扫描
# ---------------------------------------------------------------------------

def _scan_files(workdir: Path, rel_dirs: List[str], patterns: List[str]) -> List[str]:
    """收集产物文件（相对路径），按修改时间排序。"""
    result: List[str] = []
    for rel in rel_dirs:
        base = workdir / rel
        if not base.exists() or not base.is_dir():
            continue
        for f in sorted(base.rglob("*")):
            if not f.is_file():
                continue
            if any(f.name.endswith(p) or f.suffix == p for p in patterns):
                result.append(str(f.relative_to(workdir)))
    return result


# ---------------------------------------------------------------------------
# 参数 / 共振态配置（扁平化）
# ---------------------------------------------------------------------------

def _flatten_params(resonances: Dict[str, Any]) -> List[Dict[str, Any]]:
    """把 resonances_config 里的参数 dict 扁平化成可读列表。

    识别 {value, fixed, range, error} 形态的参数字典。
    只做展示用途的简化：resonance / propagator / amplitude 三层。
    """
    params: List[Dict[str, Any]] = []

    def walk(node: Any, path: str, kind: str) -> None:
        if isinstance(node, dict):
            if "value" in node and isinstance(node.get("value"), (int, float)):
                params.append({
                    "path": path,
                    "kind": kind,
                    "value": node.get("value"),
                    "fixed": bool(node.get("fixed", False)),
                    "range": node.get("range"),
                    "error": node.get("error"),
                })
                return
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else k, kind)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]", kind)

    for res_name, res in (resonances or {}).items():
        if not isinstance(res, dict):
            continue
        walk(res, "", f"resonance:{res_name}")

    return params


# ---------------------------------------------------------------------------
# 主构建
# ---------------------------------------------------------------------------

def build_pipeline_state(
    workdir: str,
    config_name: str = "llm_config_fit.toml",
    include_file_text: bool = False,
) -> Dict[str, Any]:
    """构建一份完整的流水线状态快照。

    include_file_text=True 时，把 fragment 文件正文也写入（体积较大，默认关闭，
    供"一次请求带回全部代码"的场景；否则客户端通过 workspace-files 按需读取）。
    """
    workdir = Path(workdir).resolve()
    config_path = workdir / config_name
    config_data = _read_toml(config_path) if config_path.exists() else {}

    ref_table = config_data.get("ref", {}) or {}

    stages_raw = config_data.get("stages", {}) or {}
    stage_names = list(stages_raw.keys())

    manifest_state = _build_manifest_state(workdir, config_name, stage_names)

    # resonances 可能内联在 config，也可能指向 [ref].resonances_config 文件；
    # kk_pipi 这类组合配置可能有多份（_kk / _pipi / _ctrl），全部合并。
    resonances: Dict[str, Any] = config_data.get("resonances", {}) or {}
    if not resonances:
        merged: Dict[str, Any] = {}
        for key, decl in ref_table.items():
            if key.startswith("resonances_config") or key.endswith("resonances_config"):
                val = _load_ref_file(workdir, decl)
                if val:
                    merged[key] = val
        # 若只有一份，直接展开；多份则按 ref 键名分组
        if len(merged) == 1:
            resonances = next(iter(merged.values()))
        else:
            resonances = {"_sources": merged}

    # stage 列表（含依赖、foreach、声明）
    stages: List[Dict[str, Any]] = []
    for name in stage_names:
        raw = stages_raw[name] or {}
        dep = _dependency_for_stage(name, raw, workdir)
        foreach_raw = raw.get("foreach") or raw.get("foreach_source")
        handlers: List[str] = []
        h = raw.get("handler")
        if isinstance(h, str):
            handlers.append(h)
        elif isinstance(h, dict) and h.get("path"):
            handlers.append(h["path"])
            if h.get("function"):
                handlers.append(f"{h['path']}::{h.get('function')}")

        stage_info: Dict[str, Any] = {
            "name": name,
            "kind": raw.get("kind", ""),
            "order": stage_names.index(name),
            "refs": dep["refs"],
            "dependencies": dep["deps"],
            "deps_all": dep["deps_all"],
            "foreach": _describe_foreach(foreach_raw),
            "handlers": handlers,
            "prompt": _describe_prompt(raw.get("prompt"), workdir),
            "output_type": raw.get("output_type"),
            "manifest": manifest_state["stages"].get(name, _stage_missing()),
        }
        stages.append(stage_info)

    draw = config_data.get("draw", {}) or {}
    fit = config_data.get("fit", {}) or {}

    # 产物扫描
    fragments = _scan_files(
        workdir, ["gen/fragments"],
        [".py", ".toml", ".json", ".txt", ".md"],
    )
    templates = _scan_files(workdir, ["gen/templates"], [".py"])
    prompts = _scan_files(workdir, ["gen/prompts"], [".txt", ".md"])
    llm_logs = _scan_files(workdir, ["gen/llm_logs"], [".md", ".json"])
    run_scripts = _scan_files(workdir, ["run"], [".py", ".toml"])

    return {
        "schema_version": 1,
        "config_file": config_name,
        "config_path": str(config_path.relative_to(workdir)) if config_path.exists() else config_name,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "workdir": str(workdir),
        "stage_order": stage_names,
        "stages": stages,
        "manifest": manifest_state,
        "ref": ref_table,
        "resonances": resonances,
        "params": _flatten_params(resonances),
        "draw": draw,
        "fit": fit,
        "artifacts": {
            "fragments": _existing(workdir, fragments),
            "templates": _existing(workdir, templates),
            "prompts": _existing(workdir, prompts),
            "llm_logs": _existing(workdir, llm_logs),
            "run_scripts": _existing(workdir, run_scripts),
        },
    }


def _existing(workdir: Path, rels: List[str]) -> List[str]:
    return [p for p in rels if (workdir / p).exists()]


def _describe_foreach(foreach_raw: Any) -> Optional[Dict[str, Any]]:
    if foreach_raw is None:
        return None
    if isinstance(foreach_raw, str):
        return {"type": "ref", "source": foreach_raw, "values": None}
    if isinstance(foreach_raw, dict):
        t = foreach_raw.get("type", "ref")
        if t == "list":
            return {"type": "list", "source": None, "values": foreach_raw.get("values")}
        if t == "file":
            return {
                "type": "file",
                "source": foreach_raw.get("path"),
                "field": foreach_raw.get("field"),
                "values": None,
            }
        return {"type": "ref", "source": foreach_raw.get("path"), "values": None}
    return None


def _describe_prompt(prompt: Any, workdir: Path) -> Dict[str, Any]:
    if isinstance(prompt, str):
        return {"type": "string", "path": None, "chars": len(prompt)}
    if isinstance(prompt, dict) and prompt.get("type") == "file":
        return {"type": "file", "path": prompt.get("path"), "chars": None}
    return {"type": "unknown", "path": None, "chars": None}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="LLMPWA pipeline state exporter")
    parser.add_argument("--workdir", "-w", required=True)
    parser.add_argument("--config", "-c", default="llm_config_fit.toml")
    parser.add_argument("--out", "-o", default=None, help="输出 json 路径（默认打印到 stdout）")
    parser.add_argument("--include-file-text", action="store_true")
    args = parser.parse_args(argv)

    snapshot = build_pipeline_state(
        args.workdir, args.config, include_file_text=args.include_file_text,
    )
    text = json.dumps(snapshot, indent=2, ensure_ascii=False)

    if args.out:
        out = Path(args.workdir) / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
