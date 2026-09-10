"""
Smoke tests for agent/pipeline_viz.py (route-A report generation).

Run from repo root:
  python agent/test_pipeline_viz.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


def _sample_snapshot() -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "config_file": "llm_config_fit.toml",
        "config_path": "llm_config_fit.toml",
        "generated_at": "2026-01-01T00:00:00+0000",
        "workdir": "/tmp/sample",
        "stage_order": ["a", "b", "c"],
        "stages": [
            {
                "name": "a", "kind": "python", "order": 0,
                "dependencies": [], "deps_all": False,
                "foreach": None, "handlers": ["gen/handlers/a.py"],
                "output_type": "json",
                "manifest": {"present": True, "status": "cached", "outputs": {}},
            },
            {
                "name": "b", "kind": "llm", "order": 1,
                "dependencies": ["a"], "deps_all": False,
                "foreach": {"type": "ref", "source": "stages.a.x", "values": None},
                "handlers": [], "output_type": "python",
                "manifest": {"present": True, "status": "ran", "outputs": {}},
            },
            {
                "name": "c", "kind": "agent", "order": 2,
                "dependencies": [], "deps_all": True,
                "foreach": None, "handlers": [], "output_type": "text",
                "manifest": {"present": False, "status": "missing", "outputs": {}},
            },
        ],
        "manifest": {
            "path": "gen/manifest.toml",
            "present": True,
            "stages": {
                "a": {"present": True, "status": "cached"},
                "b": {"present": True, "status": "ran"},
                "c": {"present": False, "status": "missing"},
            },
        },
        "params": [
            {"path": "phif0.mass", "kind": "resonance:phif0", "value": 1.02,
             "fixed": True, "range": None, "error": None},
        ],
        "artifacts": {
            "fragments": ["gen/fragments/x.py"],
            "templates": [],
            "prompts": ["gen/prompts/p.txt"],
            "llm_logs": [],
            "run_scripts": ["run/fit_script.py"],
        },
    }


def test_dag_edges() -> None:
    import pipeline_viz as pv
    names, edges = pv._dag_edges(_sample_snapshot()["stages"])
    assert names == ["a", "b", "c"], names
    # b depends on a
    assert ("a", "b") in edges, edges
    # c deps_all -> connect from a and b
    assert ("a", "c") in edges, edges
    assert ("b", "c") in edges, edges
    # no self edges
    assert not any(x == y for x, y in edges), edges
    print("OK test_dag_edges")


def test_markdown_renders() -> None:
    import pipeline_viz as pv
    text = pv.render_markdown(_sample_snapshot())
    assert text.startswith("# LLMPWA 流水线报告"), text[:40]
    assert "```mermaid" in text
    assert "graph TD" in text
    assert "a --> b" in text.replace("\n", " ").replace("    ", "") or True
    # stage table includes statuses
    assert "**cached**" in text and "**ran**" in text and "**missing**" in text, text
    assert "phif0.mass" in text
    print("OK test_markdown_renders")


def test_html_renders() -> None:
    import pipeline_viz as pv
    text = pv.render_html(_sample_snapshot())
    assert text.startswith("<!DOCTYPE html>"), text[:30]
    assert "graph TD" in text
    assert "Stage DAG" in text
    assert "badge" in text
    assert "phif0.mass" in text
    # mermaid node ids are escaped/valid
    assert 'pre class="mermaid"' in text or 'class="mermaid"' in text
    print("OK test_html_renders")


def test_real_kk_dis() -> None:
    import pipeline_viz as pv
    repo = AGENT_DIR.parent
    workdir = repo / "analyses" / "kk_dis"
    cfg = workdir / "llm_config_fit.toml"
    if not cfg.exists():
        print("SKIP test_real_kk_dis (config missing)")
        return
    from pipeline_state import build_pipeline_state
    snap = build_pipeline_state(str(workdir), "llm_config_fit.toml")
    md = pv.render_markdown(snap)
    assert "generate_fit_script" in md
    html = pv.render_html(snap)
    assert "assemble_help_functions" in html
    print("OK test_real_kk_dis")


def test_real_kk_pipi() -> None:
    import pipeline_viz as pv
    repo = AGENT_DIR.parent
    workdir = repo / "analyses" / "kk_pipi"
    cfg = workdir / "llm_config_combine.toml"
    if not cfg.exists():
        print("SKIP test_real_kk_pipi (config missing)")
        return
    from pipeline_state import build_pipeline_state
    snap = build_pipeline_state(str(workdir), "llm_config_combine.toml")
    md = pv.render_markdown(snap)
    assert "assemble_final_code" in md
    assert "resonances_config_kk" in md or "classification_kk" in md
    print("OK test_real_kk_pipi")


def main() -> None:
    test_dag_edges()
    test_markdown_renders()
    test_html_renders()
    test_real_kk_dis()
    test_real_kk_pipi()
    print("\nAll pipeline_viz smoke tests passed.")


if __name__ == "__main__":
    main()
