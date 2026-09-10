"""
pipeline_viz.py — LLMPWA 流水线可视化报告生成器（路线 A）。

消费 agent/pipeline_state.py 的 build_pipeline_state() —— 也就是把"静态结构 +
动态状态"抽成 JSON 快照的那份数据 —— 渲染成两类可直接展示的报告：

  - Markdown + Mermaid（`graph TD`）：适合在 DSH 对话里作为 artifact 直接看，
    依赖 DSH/GUI 的 markdown 渲染器支持 Mermaid。
  - 自包含 HTML：内嵌 CSS，附带 Mermaid（CDN）+ 纯文本 DAG 兜底 + 可折叠的
    stage / 参数 / 产物区，适合"点开查看"。

本模块不做任何 DSH 改动，只读 pipeline_state 的数据（路线 A 的最小落地）。

用法：
  python agent/pipeline_viz.py -w analyses/kk_dis -c llm_config_fit.toml -o gen/pipeline_report.md
  python agent/pipeline_viz.py -w analyses/kk_dis -c llm_config_fit.toml -o gen/pipeline_report.html
  python agent/pipeline_viz.py -w analyses/kk_dis -c llm_config_fit.toml -o gen/pipeline_report.md --overwrite
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

_KIND_LABEL = {
    "python": "py",
    "llm": "llm",
    "agent": "agent",
}


def _node_id(name: str) -> str:
    """构造合法的 Mermaid 节点 id：非字母数字一律转下划线。"""
    out = "".join(c if c.isalnum() else "_" for c in name)
    if not out or out[0].isdigit():
        out = "n_" + out
    return out


def _mk_edge(names: Set[Tuple[str, str]], a: str, b: str) -> None:
    """记录 a → b 一条边（去重）。"""
    if a == b:
        return
    names.add((a, b))


def _dag_edges(stages: List[Dict[str, Any]]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """从 stage 列表构建 DAG 节点与边。

    边来源：
      - dependencies: 命中的具体上游 stage → 该 stage。
      - deps_all（出现 stages[*]）: 除自身外所有其它 stage → 该 stage。

    foreach 的子项（resonance_calculation.<key>）不在此枚举，仅呈现 stage 级节点；
    foreach 信息在 stage 卡片里单独标注。
    """
    names: List[str] = []
    edges: Set[Tuple[str, str]] = set()
    order = [s["name"] for s in stages]
    for s in stages:
        name = s["name"]
        if name not in names:
            names.append(name)
        for dep in s.get("dependencies", []):
            if dep in order:
                _mk_edge(edges, dep, name)
        if s.get("deps_all"):
            for other in order:
                _mk_edge(edges, other, name)
    return names, sorted(edges)


def _status_color(status: str) -> str:
    return {
        "cached": "#22c55e",
        "ran": "#3b82f6",
        "missing": "#ef4444",
    }.get(status, "#9ca3af")


def _status_label(status: str) -> str:
    return {
        "cached": "cached",
        "ran": "ran",
        "missing": "missing",
    }.get(status, status)


def _params_rows(params: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把扁平参数列表转成表格行，按 resonance 分组。"""
    rows: List[Dict[str, Any]] = []
    for p in params:
        rows.append(
            {
                "res": p.get("kind", ""),
                "path": p.get("path", ""),
                "value": p.get("value"),
                "fixed": p.get("fixed", False),
                "range": p.get("range"),
                "error": p.get("error"),
            }
        )
    return rows


def _format_val(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(str(x) for x in v) + "]"
    return str(v)


# ---------------------------------------------------------------------------
# Markdown + Mermaid
# ---------------------------------------------------------------------------

def _mermaid_diagram(stages: List[Dict[str, Any]]) -> str:
    names, edges = _dag_edges(stages)
    lines = ["```mermaid", "graph TD"]
    for s in stages:
        label = f"{s['name']}<br/>({_KIND_LABEL.get(s.get('kind'), s.get('kind'))})"
        lines.append(f"    {_node_id(s['name'])}[\"{label}\"]")
    for a, b in edges:
        lines.append(f"    {_node_id(a)} --> {_node_id(b)}")
    lines.append("```")
    return "\n".join(lines)


def render_markdown(snapshot: Dict[str, Any]) -> str:
    """渲染一份 Markdown + Mermaid 报告。"""
    stages = snapshot.get("stages", [])
    manifest = snapshot.get("manifest", {})
    mstages = manifest.get("stages", {}) if isinstance(manifest, dict) else {}

    lines: List[str] = []
    lines.append(f"# LLMPWA 流水线报告 — `{snapshot.get('workdir')}`")
    lines.append("")
    lines.append(f"- 配置文件：`{snapshot.get('config_file')}`")
    lines.append(f"- 生成时间：`{snapshot.get('generated_at')}`")
    lines.append(f"- stage 数：`{len(stages)}`；参数数：`{len(snapshot.get('params', []))}`")
    lines.append("")

    # DAG
    lines.append("## Stage DAG")
    lines.append("")
    lines.append(_mermaid_diagram(stages))
    lines.append("")

    # 状态表
    lines.append("## Stage 状态")
    lines.append("")
    lines.append("| # | stage | kind | 状态 | 依赖 | foreach |")
    lines.append("|---|-------|------|------|------|---------|")
    for i, s in enumerate(stages):
        st = (mstages.get(s["name"]) or {})
        status = st.get("status", "missing")
        deps = ", ".join(s.get("dependencies", [])) or "—"
        if s.get("deps_all"):
            deps = "all"
        foreach = s.get("foreach")
        foreach_str = "—"
        if isinstance(foreach, dict):
            foreach_str = foreach.get("source") or foreach.get("type") or "—"
        lines.append(
            f"| {i} | `{s['name']}` | {s.get('kind', '')} | **{_status_label(status)}** "
            f"| {deps} | {foreach_str} |"
        )
    lines.append("")

    # 产物
    artifacts = snapshot.get("artifacts", {})
    lines.append("## 产物代码文件")
    lines.append("")
    if artifacts:
        for category, files in artifacts.items():
            if files:
                lines.append(f"### {category}")
                for f in files:
                    lines.append(f"- `{f}`")
                lines.append("")
    else:
        lines.append("（暂无 `gen/`、`run/` 产物 — 流水线尚未运行）")
        lines.append("")

    # 参数
    params = snapshot.get("params", [])
    lines.append("## 参数 / 共振态配置")
    lines.append("")
    if params:
        lines.append("| 共振态 | 参数路径 | 值 | fixed | range | error |")
        lines.append("|--------|----------|----|-------|-------|-------|")
        for row in _params_rows(params):
            lines.append(
                f"| {row['res']} | `{row['path']}` | {_format_val(row['value'])} "
                f"| {row['fixed']} | {_format_val(row['range'])} | {_format_val(row['error'])} |"
            )
        lines.append("")
    else:
        lines.append("（无参数）")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _render_html_css() -> str:
    return """
* { box-sizing: border-box; }
body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
       padding: 24px; color: #111827; background: #f9fafb; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 28px 0 10px; border-bottom: 1px solid #e5e7eb;
     padding-bottom: 6px; }
.meta { color: #6b7280; font-size: 13px; margin: 0 0 16px; }
.chips { display: flex; gap: 8px; flex-wrap: wrap; margin: 0 0 20px; }
.chip { background: #fff; border: 1px solid #e5e7eb; border-radius: 9999px;
        padding: 4px 12px; font-size: 13px; }
.chip b { margin-left: 2px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; background: #fff; }
th, td { border: 1px solid #e5e7eb; padding: 6px 10px; text-align: left;
         vertical-align: top; }
th { background: #f3f4f6; }
code { background: #f3f4f6; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
.badge { display: inline-block; padding: 1px 8px; border-radius: 9999px;
         color: #fff; font-size: 11px; font-weight: 600; }
.dag-wrap { background: #fff; border: 1px solid #e5e7eb; border-radius: 8px;
            padding: 16px; overflow: auto; }
.dag-list { font-size: 13px; margin-top: 12px; }
details { margin: 8px 0; border: 1px solid #e5e7eb; border-radius: 8px;
          background: #fff; padding: 8px 12px; }
summary { cursor: pointer; font-size: 13px; font-weight: 600; }
pre { background: #f9fafb; padding: 10px; border-radius: 6px; font-size: 12px;
      overflow: auto; max-height: 340px; }
"""


def _html_dag_block(stages: List[Dict[str, Any]]) -> str:
    names, edges = _dag_edges(stages)
    mermaid_lines = ["graph TD"]
    for s in stages:
        label = f"{s['name']}<br/>({_KIND_LABEL.get(s.get('kind'), s.get('kind'))})"
        mermaid_lines.append(f"    {_node_id(s['name'])}[\"{label}\"]")
    for a, b in edges:
        mermaid_lines.append(f"    {_node_id(a)} --> {_node_id(b)}")
    dag_text = html.escape("\n".join(mermaid_lines))
    edge_list = "<br/>".join(
        f"{html.escape(a)} → {html.escape(b)}" for a, b in edges
    ) or "（无依赖边）"
    return f"""
<div class="dag-wrap">
  <pre class="mermaid">{dag_text}</pre>
  <div class="dag-list"><b>依赖边（兜底文本）：</b><br/>{edge_list}</div>
</div>
<script type="module">
  var m = document.querySelector("pre.mermaid");
  if (window.mermaid && m) {{ mermaid.initialize({{ startOnLoad: true }}); }}
</script>
"""


def render_html(snapshot: Dict[str, Any]) -> str:
    """渲染一份自包含的 HTML 报告（含折叠交互）。"""
    stages = snapshot.get("stages", [])
    manifest = snapshot.get("manifest", {})
    mstages = manifest.get("stages", {}) if isinstance(manifest, dict) else {}
    params = snapshot.get("params", [])
    artifacts = snapshot.get("artifacts", {})
    workdir = html.escape(str(snapshot.get("workdir", "")))

    # 汇总 chips
    statuses = [ (mstages.get(s["name"]) or {}).get("status", "missing") for s in stages ]
    cache_count = statuses.count("cached")
    ran_count = statuses.count("ran")
    missing_count = statuses.count("missing")

    chips = f"""
<div class="chips">
  <span class="chip">stage <b>{len(stages)}</b></span>
  <span class="chip">cached <b>{cache_count}</b></span>
  <span class="chip">ran <b>{ran_count}</b></span>
  <span class="chip">missing <b>{missing_count}</b></span>
  <span class="chip">params <b>{len(params)}</b></span>
</div>
"""

    # Stage 卡片
    stage_cards: List[str] = []
    for i, s in enumerate(stages):
        st = (mstages.get(s["name"]) or {})
        status = st.get("status", "missing")
        color = _status_color(status)
        deps = ", ".join(s.get("dependencies", [])) or "—"
        if s.get("deps_all"):
            deps = "all"
        foreach = "—"
        if isinstance(s.get("foreach"), dict):
            f = s["foreach"]
            foreach = f.get("source") or f.get("type") or "—"
        handlers = ", ".join(s.get("handlers", [])) or "—"
        outputs = st.get("outputs", {})
        out_rows = "".join(
            f"<tr><td><code>{html.escape(k)}</code></td>"
            f"<td>{html.escape(json.dumps(v, ensure_ascii=False))}</td></tr>"
            for k, v in outputs.items()
        ) or "<tr><td colspan=2>无</td></tr>"
        stage_cards.append(f"""
<details>
  <summary><span class="badge" style="background:{color}">{_status_label(status)}</span>
    {html.escape(s['name'])} <small>({s.get('kind', '')}, #{i})</small></summary>
  <table>
    <tr><th>依赖</th><td>{html.escape(deps)}</td></tr>
    <tr><th>foreach</th><td>{html.escape(foreach)}</td></tr>
    <tr><th>handlers</th><td>{html.escape(handlers)}</td></tr>
    <tr><th>output_type</th><td>{html.escape(str(s.get('output_type') or '—'))}</td></tr>
    <tr><th>manifest 输出</th><td><table>{out_rows}</table></td></tr>
  </table>
</details>
""")

    # 产物
    artifact_blocks: List[str] = []
    for category, files in artifacts.items():
        if not files:
            continue
        items = "".join(f"<li><code>{html.escape(f)}</code></li>" for f in files)
        artifact_blocks.append(f"<h3>{html.escape(category)}</h3><ul>{items}</ul>")
    if not artifact_blocks:
        artifact_blocks.append("<p>暂无 `gen/`、`run/` 产物 — 流水线尚未运行。</p>")

    # 参数表
    if params:
        param_rows = "".join(
            f"<tr><td>{html.escape(r['res'])}</td><td><code>{html.escape(r['path'])}</code></td>"
            f"<td>{html.escape(_format_val(r['value']))}</td>"
            f"<td>{'✅' if r['fixed'] else '—'}</td>"
            f"<td>{html.escape(_format_val(r['range']))}</td>"
            f"<td>{html.escape(_format_val(r['error']))}</td></tr>"
            for r in _params_rows(params)
        )
        params_block = (
            "<table><thead><tr><th>共振态</th><th>参数路径</th><th>值</th>"
            "<th>fixed</th><th>range</th><th>error</th></tr></thead>"
            f"<tbody>{param_rows}</tbody></table>"
        )
    else:
        params_block = "<p>（无参数）</p>"

    return f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<title>LLMPWA 流水线报告</title>
<style>{_render_html_css()}</style>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
<h1>LLMPWA 流水线报告</h1>
<p class="meta">workdir: {workdir}<br/>config: {html.escape(str(snapshot.get('config_file', '')))}
· 生成于 {html.escape(str(snapshot.get('generated_at', '')))}</p>
{chips}
<h2>Stage DAG</h2>
{_html_dag_block(stages)}
<h2>Stage 详情</h2>
{''.join(stage_cards)}
<h2>产物代码文件</h2>
{''.join(artifact_blocks)}
<h2>参数 / 共振态配置</h2>
{params_block}
</body>
</html>
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    import argparse  # noqa: F811

    parser = argparse.ArgumentParser(
        description="LLMPWA pipeline visualization report (route A)"
    )
    parser.add_argument("--workdir", "-w", required=True)
    parser.add_argument("--config", "-c", default="llm_config_fit.toml")
    parser.add_argument(
        "--format", "-f", default="md", choices=["md", "html"],
        help="Output format (default: md)",
    )
    parser.add_argument(
        "--out", "-o", default=None,
        help="Output path (relative to workdir). Default prints to stdout.",
    )
    parser.add_argument("--include-file-text", action="store_true")
    args = parser.parse_args(argv)

    from pipeline_state import build_pipeline_state

    snapshot = build_pipeline_state(
        args.workdir, args.config, include_file_text=args.include_file_text,
    )
    if args.format == "html":
        text = render_html(snapshot)
    else:
        text = render_markdown(snapshot)

    if args.out:
        out = Path(args.workdir) / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"Wrote {out} ({len(text)} chars)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
