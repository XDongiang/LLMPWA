"""
Smoke tests for agent/pipeline_state.py (the LLMPWA → DSH data model).

Run from repo root:
  python agent/test_pipeline_state.py
or:
  cd agent && python test_pipeline_state.py

Covers:
  - _ref_target_stages: the <<...>> / stages dotpath → stage-name mapping.
  - _dependency_for_stage: collecting refs/deps/deps_all from a stage decl.
  - _flatten_params: flattening {value, fixed, range, error} param dicts.
  - multi-source resonances merging into resonances._sources.
  - build_pipeline_state on the real analyses (kk_new / kk_dis / kk_pipi).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


def test_ref_target_stages() -> None:
    import pipeline_state as ps

    cases = {
        "stages.config_strip.stripped_config": ["config_strip"],
        "stages.resonance_calculation[*].all": ["resonance_calculation"],
        "stages.resonance_calculation[BW_BW].all": ["resonance_calculation"],
        "stages[*].all": ["*"],
        "stages.some_stage[*]": ["some_stage"],
        "ref.code_template.COMMON_UTILITIES": [],
        "key": [],
        "run/free_params.toml": [],
        "stages.config_strip.all_sbc": ["config_strip"],
    }
    for expr, want in cases.items():
        got = ps._ref_target_stages(expr)
        assert got == want, f"{expr!r}: got {got}, want {want}"
    print("OK test_ref_target_stages")


def test_dependency_for_stage() -> None:
    """Stage prompt + input.list should be scanned for <<...>> and bare dotpaths."""
    import pipeline_state as ps

    workdir = Path("/tmp/ps_test_workdir")
    stage_raw: Dict[str, Any] = {
        "kind": "python",
        "handler": {"type": "file", "path": "gen/handlers/x.py", "function": "x"},
        "prompt": "see <<stages.config_strip.stripped_config>> and <<stages[*].all>>",
        "input": {
            "list": [
                "ref.code_template.COMMON_UTILITIES",
                "stages.resonance_calculation[*].all",
                "stages[*].all",
            ]
        },
        "output_type": "python",
        "output": {
            "all": {"type": "file", "path": "run/fit_script.py"},
        },
    }
    dep = ps._dependency_for_stage("assemble_final_code", stage_raw, workdir)
    assert "config_strip" in dep["deps"], dep
    assert "resonance_calculation" in dep["deps"], dep
    assert dep["deps_all"] is True, dep
    # refs should contain both the <<...>> expressions and the bare dotpaths
    assert "stages[*].all" in dep["refs"], dep
    assert "stages.config_strip.stripped_config" in dep["refs"], dep
    print("OK test_dependency_for_stage")


def test_flatten_params() -> None:
    import pipeline_state as ps

    resonances: Dict[str, Any] = {
        "phif0_980": {
            "propagators": {
                "A_propagator": {
                    "mass": {"value": 1.02, "fixed": True},
                    "width": {"value": 0.004, "fixed": True},
                }
            },
            "Amplitude": {
                "AMP": "phif0_kk",
                "const1": {"value": 0.1, "fixed": True},
                "theta": {"value": 0.3, "fixed": False, "range": [0.0, 1.0]},
            },
            "nested_list": [{"a": {"value": 2.0}}, {"a": {"value": 3.0}}],
        }
    }
    params = ps._flatten_params(resonances)
    assert params, "expected some flattened params"
    # param dicts carry value/fixed/range/error
    for p in params:
        assert "value" in p and "fixed" in p, p
    # nested list traversal
    paths = [p["path"] for p in params]
    assert any("nested_list[0].a" in path for path in paths), paths
    assert any("theta" in path for path in paths), paths
    # fixed flag propagates
    fixed = [p for p in params if p["fixed"]]
    assert any("mass" in p["path"] for p in fixed), params
    # range captures
    theta = [p for p in params if p["path"].endswith("Amplitude.theta")][0]
    assert theta["range"] == [0.0, 1.0], theta
    print("OK test_flatten_params")


def test_multi_source_resonances_merge() -> None:
    """Multiple resonances_config_* refs should bundle under resonances._sources."""
    import pipeline_state as ps

    workdir = Path("/tmp/ps_test_multi")
    for name in ("resonances_config_kk.toml", "resonances_config_pipi.toml"):
        r = workdir / name
        r.parent.mkdir(parents=True, exist_ok=True)
        r.write_text(
            "[resonances]\nphif0 = { value = 1.02, fixed = true }\n",
            encoding="utf-8",
        )

    config_data: Dict[str, Any] = {
        "ref": {
            "resonances_config_kk": {"type": "file", "path": "resonances_config_kk.toml"},
            "resonances_config_pipi": {"type": "file", "path": "resonances_config_pipi.toml"},
        },
        "stages": {},
    }
    merged: Dict[str, Any] = {}
    for key, decl in config_data["ref"].items():
        if key.startswith("resonances_config") or key.endswith("resonances_config"):
            val = ps._load_ref_file(workdir, decl)
            if val:
                merged[key] = val
    assert len(merged) == 2, merged
    assert "_sources" not in merged
    # When >1 source, build_pipeline_state wraps in _sources. Simulate that here.
    wrapped = {"_sources": merged}
    assert wrapped["_sources"]["resonances_config_kk"]["resonances"]["phif0"]["value"] == 1.02
    print("OK test_multi_source_resonances_merge")


def _build_and_check(workdir: str, config_name: str) -> Dict[str, Any]:
    import pipeline_state as ps
    snap = ps.build_pipeline_state(workdir=workdir, config_name=config_name)
    assert snap["schema_version"] == 1
    # every stage has a manifest sub-object keyed as `manifest`
    for st in snap["stages"]:
        assert "manifest" in st, f"{st['name']} missing manifest key"
        assert "manifes" not in st, f"{st['name']} still has typos key"
    return snap


def test_real_kk_new() -> None:
    repo = AGENT_DIR.parent
    workdir = repo / "analyses" / "kk_new"
    if not (workdir / "llm_config_fit.toml").exists():
        print("SKIP test_real_kk_new (kk_new config missing)")
        return
    snap = _build_and_check(str(workdir), "llm_config_fit.toml")
    assert len(snap["stages"]) == 12, len(snap["stages"])
    assert snap["resonances"], "expected inline resonances"
    print("OK test_real_kk_new")


def test_real_kk_dis() -> None:
    repo = AGENT_DIR.parent
    workdir = repo / "analyses" / "kk_dis"
    if not (workdir / "llm_config_fit.toml").exists():
        print("SKIP test_real_kk_dis (kk_dis config missing)")
        return
    snap = _build_and_check(str(workdir), "llm_config_fit.toml")
    stages = [s["name"] for s in snap["stages"]]
    assert "inspect_data_shapes" in stages, stages
    # agent-kind stage must carry 4 output fields as declared
    idsp = next(s for s in snap["stages"] if s["name"] == "inspect_data_shapes")
    assert idsp["kind"] == "agent", idsp["kind"]
    print("OK test_real_kk_dis")


def test_real_kk_pipi() -> None:
    repo = AGENT_DIR.parent
    workdir = repo / "analyses" / "kk_pipi"
    config = workdir / "llm_config_combine.toml"
    if not config.exists():
        print("SKIP test_real_kk_pipi (kk_pipi config missing)")
        return
    snap = _build_and_check(str(workdir), "llm_config_combine.toml")
    assert len(snap["stages"]) == 20, len(snap["stages"])
    assert sorted(snap["resonances"]) == ["_sources"], snap["resonances"].keys()
    # assemble_final_code should depend on all stages via stages[*]
    afc = next(s for s in snap["stages"] if s["name"] == "assemble_final_code")
    assert afc["deps_all"] is True, afc
    assert "resonance_calculation" in afc["dependencies"], afc
    print("OK test_real_kk_pipi")


def main() -> None:
    test_ref_target_stages()
    test_dependency_for_stage()
    test_flatten_params()
    test_multi_source_resonances_merge()
    test_real_kk_new()
    test_real_kk_dis()
    test_real_kk_pipi()
    print("\nAll pipeline_state smoke tests passed.")


if __name__ == "__main__":
    main()
