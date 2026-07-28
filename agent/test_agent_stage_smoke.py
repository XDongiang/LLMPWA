"""
Lightweight smoke tests for kind=agent (no real LLM API).

Run from repo root:
  python agent/test_agent_stage_smoke.py
or:
  cd agent && python test_agent_stage_smoke.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure agent/ is on path whether run from repo root or agent/
AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


def test_tool_registry_and_path_guard() -> None:
    from tools.base import ToolContext
    from tools.registry import ToolRegistry

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "hello.txt").write_text("line1\nline2\n", encoding="utf-8")
        ctx = ToolContext(
            workdir=root,
            workspace_root=root,
            cwd=root,
            stage_name="t",
        )
        reg = ToolRegistry(["read", "write", "task"])
        schemas = reg.openai_tools()
        assert len(schemas) == 3
        assert schemas[0]["type"] == "function"

        r = reg.dispatch(ctx, "read", {"path": "hello.txt", "offset": 1, "limit": 1})
        assert r.ok and "line1" in r.content

        bad = reg.dispatch(ctx, "read", {"path": "../outside.txt"})
        # resolve may still be outside → permission error
        assert not bad.ok

        w = reg.dispatch(ctx, "write", {"path": "out.txt", "content": "abc"})
        assert w.ok
        assert (root / "out.txt").read_text(encoding="utf-8") == "abc"

        fin = reg.dispatch(ctx, "task", {"action": "finish", "result": "DONE"})
        assert fin.ok and fin.meta.get("finished") is True
        assert fin.meta.get("result") == "DONE"
    print("OK test_tool_registry_and_path_guard")


def test_config_loader_accepts_agent() -> None:
    from config_loader import ConfigLoader

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "sys.txt").write_text("system", encoding="utf-8")
        cfg_path = root / "llm_config_agent.toml"
        cfg_path.write_text(
            """
[stages.demo]
kind = "agent"
system_prompt = { type = "file", path = "sys.txt" }
prompt = "do stuff"
tools = ["read", "write", "task"]
max_turns = 5
output_type = "text"
output.all = { type = "file", path = "out.txt" }
""",
            encoding="utf-8",
        )
        cfg = ConfigLoader.load(cfg_path)
        stage = cfg.get_stage("demo")
        assert stage is not None
        assert stage.kind == "agent"
        assert stage.tools == ["read", "write", "task"]
        assert stage.max_turns == 5
        assert stage.cache is False
        # agent 默认需要人工审阅
        assert stage.require_approval is True
    print("OK test_config_loader_accepts_agent")


class _FakeLLM:
    """Scripted tool-calling LLM for AgentStageRunner."""

    def __init__(self) -> None:
        self.calls = 0

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model=None,
        tool_choice="auto",
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        self.calls += 1
        if self.calls == 1:
            # ask for phrase — but we skip ask_user in scripted path: write directly
            msg = {"role": "assistant", "content": ""}
            tcs = [{
                "id": "1",
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps({
                        "path": "gen/fragments/agent_demo_note.txt",
                        "content": "hello-from-fake-llm",
                    }),
                },
            }]
            return msg, tcs, {}
        if self.calls == 2:
            msg = {"role": "assistant", "content": ""}
            tcs = [{
                "id": "2",
                "type": "function",
                "function": {
                    "name": "task",
                    "arguments": json.dumps({
                        "action": "finish",
                        "result": "gen/fragments/agent_demo_note.txt :: hello-from-fake-llm",
                    }),
                },
            }]
            return msg, tcs, {}
        raise AssertionError("unexpected extra LLM call")


class _FakeEngine:
    def __init__(self, workdir: Path) -> None:
        from manifest import Manifest
        from resolver import Resolver
        from config_loader import Config

        self.workdir = workdir
        # minimal config object for Resolver
        raw = {"stages": {}, "ref": {}}
        self.config = Config(raw, workdir / "llm_config.toml")
        gen = workdir / "gen"
        gen.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(gen / "manifest_agent.toml")
        self.resolver = Resolver(workdir, self.manifest, self.config)
        self.resolver.current_stage = "agent_demo"
        self.resolver.visible_stages = set()
        self.llm_client = _FakeLLM()


def test_agent_stage_runner_scripted() -> None:
    from agent_stage import AgentStageRunner
    from config_loader import StageConfig

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "gen" / "prompts").mkdir(parents=True)
        (root / "gen" / "fragments").mkdir(parents=True)
        (root / "gen" / "prompts" / "sys.txt").write_text(
            "You are a test agent.", encoding="utf-8"
        )
        stage = StageConfig(
            "agent_demo",
            {
                "kind": "agent",
                "system_prompt": {"type": "file", "path": "gen/prompts/sys.txt"},
                "prompt": "write a note and finish",
                "tools": ["read", "write", "task"],
                "max_turns": 5,
                # non-interactive smoke: skip human gate
                "require_approval": False,
                "output_type": "text",
                "output": {
                    "all": {
                        "type": "file",
                        "path": "gen/fragments/agent_demo_result.txt",
                    },
                    "transcript": {
                        "type": "file",
                        "path": "gen/llm_logs/agent_demo_transcript.json",
                    },
                },
            },
        )
        engine = _FakeEngine(root)
        AgentStageRunner("agent_demo", stage, engine).execute()

        out = root / "gen" / "fragments" / "agent_demo_result.txt"
        note = root / "gen" / "fragments" / "agent_demo_note.txt"
        assert note.exists()
        assert note.read_text(encoding="utf-8") == "hello-from-fake-llm"
        assert out.exists()
        assert "hello-from-fake-llm" in out.read_text(encoding="utf-8")
        tr = root / "gen" / "llm_logs" / "agent_demo_transcript.json"
        assert tr.exists()
        data = json.loads(tr.read_text(encoding="utf-8"))
        assert data["final_result"]
    print("OK test_agent_stage_runner_scripted")


class _FakeLLMRejectThenFinish:
    """finish → (human reject injected by runner) → finish again."""

    def __init__(self) -> None:
        self.calls = 0

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model=None,
        tool_choice="auto",
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        self.calls += 1
        if self.calls == 1:
            msg = {"role": "assistant", "content": "first draft"}
            tcs = [{
                "id": "1",
                "type": "function",
                "function": {
                    "name": "task",
                    "arguments": json.dumps({
                        "action": "finish",
                        "result": "DRAFT_V1",
                    }),
                },
            }]
            return msg, tcs, {}
        if self.calls == 2:
            # After reject feedback, produce revised result
            msg = {"role": "assistant", "content": "revised"}
            tcs = [{
                "id": "2",
                "type": "function",
                "function": {
                    "name": "task",
                    "arguments": json.dumps({
                        "action": "finish",
                        "result": "DRAFT_V2_FIXED",
                    }),
                },
            }]
            return msg, tcs, {}
        raise AssertionError(f"unexpected extra LLM call #{self.calls}")


def test_agent_require_approval_reject_then_approve(monkeypatch_input=None) -> None:
    """task.finish → reject with feedback → finish again → approve."""
    from agent_stage import AgentStageRunner
    from config_loader import StageConfig
    import builtins

    replies = iter(["reject please fix sign error", "approve"])

    def _fake_input(prompt: str = "") -> str:
        try:
            return next(replies)
        except StopIteration as e:
            raise AssertionError(f"unexpected extra input() prompt={prompt!r}") from e

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "gen" / "prompts").mkdir(parents=True)
        (root / "gen" / "fragments").mkdir(parents=True)
        (root / "gen" / "prompts" / "sys.txt").write_text(
            "You are a test agent.", encoding="utf-8"
        )
        stage = StageConfig(
            "agent_review",
            {
                "kind": "agent",
                "system_prompt": {"type": "file", "path": "gen/prompts/sys.txt"},
                "prompt": "produce a result",
                "tools": ["task"],
                "max_turns": 5,
                "require_approval": True,
                "output_type": "text",
                "output": {
                    "all": {
                        "type": "file",
                        "path": "gen/fragments/agent_review_result.txt",
                    },
                    "transcript": {
                        "type": "file",
                        "path": "gen/llm_logs/agent_review_transcript.json",
                    },
                },
            },
        )
        engine = _FakeEngine(root)
        engine.llm_client = _FakeLLMRejectThenFinish()

        real_input = builtins.input
        real_isatty = sys.stdin.isatty
        try:
            builtins.input = _fake_input  # type: ignore[assignment]
            sys.stdin.isatty = lambda: True  # type: ignore[method-assign]
            AgentStageRunner("agent_review", stage, engine).execute()
        finally:
            builtins.input = real_input  # type: ignore[assignment]
            sys.stdin.isatty = real_isatty  # type: ignore[method-assign]

        out = root / "gen" / "fragments" / "agent_review_result.txt"
        assert out.exists()
        assert out.read_text(encoding="utf-8") == "DRAFT_V2_FIXED"
        tr = root / "gen" / "llm_logs" / "agent_review_transcript.json"
        data = json.loads(tr.read_text(encoding="utf-8"))
        assert data["approval_rounds"] == 2
        assert data["final_result"] == "DRAFT_V2_FIXED"
        human_events = [
            m for m in data["messages"] if m.get("role") == "human_approval"
        ]
        assert len(human_events) == 2
        assert human_events[0]["approved"] is False
        assert "sign" in (human_events[0].get("feedback") or "")
        assert human_events[1]["approved"] is True
        assert engine.llm_client.calls == 2
    print("OK test_agent_require_approval_reject_then_approve")


class _FakeLLMMultiFile:
    """Write report + 3 fragments, then finish with outputs checklist."""

    def __init__(self) -> None:
        self.calls = 0

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model=None,
        tool_choice="auto",
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        self.calls += 1
        files = [
            (
                "gen/fragments/data_shapes_report.txt",
                "phi_kk shape=(100,)\nphif0_kk shape=(2, 100, 3)\n",
            ),
            (
                "gen/fragments/load_data_func.py",
                "def load_data():\n    return {'data_phi_kk': None}\n",
            ),
            (
                "gen/fragments/normalize_data_func.py",
                "def normalize_data(data):\n    return data\n",
            ),
            (
                "gen/fragments/shard_data_distributed_func.py",
                "def shard_data_distributed(data, mesh):\n    return data\n",
            ),
        ]
        if 1 <= self.calls <= 4:
            path, content = files[self.calls - 1]
            msg = {"role": "assistant", "content": ""}
            tcs = [{
                "id": str(self.calls),
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps({"path": path, "content": content}),
                },
            }]
            return msg, tcs, {}
        if self.calls == 5:
            msg = {"role": "assistant", "content": ""}
            tcs = [{
                "id": "5",
                "type": "function",
                "function": {
                    "name": "task",
                    "arguments": json.dumps({
                        "action": "finish",
                        "result": "summary: wrote report + 3 preprocess funcs",
                        "outputs": {
                            "all": "gen/fragments/data_shapes_report.txt",
                            "load_data": "gen/fragments/load_data_func.py",
                            "normalize_data": "gen/fragments/normalize_data_func.py",
                            "shard_data_distributed": (
                                "gen/fragments/shard_data_distributed_func.py"
                            ),
                        },
                    }),
                },
            }]
            return msg, tcs, {}
        raise AssertionError(f"unexpected extra LLM call #{self.calls}")


def test_agent_multi_file_outputs_harvest() -> None:
    """Multi-field type=file outputs are harvested into manifest paths."""
    from agent_stage import AgentStageRunner
    from config_loader import StageConfig

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "gen" / "prompts").mkdir(parents=True)
        (root / "gen" / "fragments").mkdir(parents=True)
        (root / "gen" / "prompts" / "sys.txt").write_text(
            "You are a multi-file test agent.", encoding="utf-8"
        )
        stage = StageConfig(
            "inspect_data_shapes",
            {
                "kind": "agent",
                "system_prompt": {"type": "file", "path": "gen/prompts/sys.txt"},
                "prompt": "inspect shapes and write preprocess fragments",
                "tools": ["read", "write", "task"],
                "max_turns": 10,
                "require_approval": False,
                "cache": False,
                "output_type": "text",
                "output": {
                    "all": {
                        "type": "file",
                        "path": "gen/fragments/data_shapes_report.txt",
                    },
                    "load_data": {
                        "type": "file",
                        "path": "gen/fragments/load_data_func.py",
                    },
                    "normalize_data": {
                        "type": "file",
                        "path": "gen/fragments/normalize_data_func.py",
                    },
                    "shard_data_distributed": {
                        "type": "file",
                        "path": "gen/fragments/shard_data_distributed_func.py",
                    },
                    "transcript": {
                        "type": "file",
                        "path": "gen/llm_logs/inspect_data_shapes_transcript.json",
                    },
                },
            },
        )
        engine = _FakeEngine(root)
        engine.llm_client = _FakeLLMMultiFile()
        AgentStageRunner("inspect_data_shapes", stage, engine).execute()

        report = root / "gen" / "fragments" / "data_shapes_report.txt"
        load_f = root / "gen" / "fragments" / "load_data_func.py"
        norm_f = root / "gen" / "fragments" / "normalize_data_func.py"
        shard_f = root / "gen" / "fragments" / "shard_data_distributed_func.py"
        assert report.exists() and "phi_kk" in report.read_text(encoding="utf-8")
        assert "def load_data" in load_f.read_text(encoding="utf-8")
        assert "def normalize_data" in norm_f.read_text(encoding="utf-8")
        assert "def shard_data_distributed" in shard_f.read_text(encoding="utf-8")

        node = engine.manifest.get_stage("inspect_data_shapes")
        assert node is not None
        for field, rel in [
            ("all", "gen/fragments/data_shapes_report.txt"),
            ("load_data", "gen/fragments/load_data_func.py"),
            ("normalize_data", "gen/fragments/normalize_data_func.py"),
            ("shard_data_distributed", "gen/fragments/shard_data_distributed_func.py"),
        ]:
            assert isinstance(node.get(field), dict), field
            assert node[field].get("type") == "file", field
            assert node[field].get("path") == rel, field

        # harvested content is what resolver/manifest would re-read
        assert "def load_data" in engine.manifest.read_field(
            "inspect_data_shapes", "load_data", root
        )
        tr = root / "gen" / "llm_logs" / "inspect_data_shapes_transcript.json"
        assert tr.exists()
        data = json.loads(tr.read_text(encoding="utf-8"))
        assert "load_data" in data.get("harvested_fields", [])
    print("OK test_agent_multi_file_outputs_harvest")


class _FakeLLMMultiFileMissing:
    """Finish without writing fragments → runner should nudge, then write + finish."""

    def __init__(self) -> None:
        self.calls = 0
        self.saw_incomplete = False

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model=None,
        tool_choice="auto",
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        self.calls += 1
        # Detect incomplete-outputs feedback from runner
        for m in messages:
            if m.get("role") == "user" and "incomplete" in str(m.get("content", "")).lower():
                self.saw_incomplete = True

        if self.calls == 1:
            # Premature finish without files
            msg = {"role": "assistant", "content": ""}
            tcs = [{
                "id": "1",
                "type": "function",
                "function": {
                    "name": "task",
                    "arguments": json.dumps({
                        "action": "finish",
                        "result": "oops no files",
                    }),
                },
            }]
            return msg, tcs, {}
        if 2 <= self.calls <= 4:
            names = [
                (
                    "gen/fragments/load_data_func.py",
                    "def load_data():\n    return {}\n",
                ),
                (
                    "gen/fragments/normalize_data_func.py",
                    "def normalize_data(data):\n    return data\n",
                ),
                (
                    "gen/fragments/shard_data_distributed_func.py",
                    "def shard_data_distributed(data, mesh):\n    return data\n",
                ),
            ]
            path, content = names[self.calls - 2]
            msg = {"role": "assistant", "content": ""}
            tcs = [{
                "id": str(self.calls),
                "type": "function",
                "function": {
                    "name": "write",
                    "arguments": json.dumps({"path": path, "content": content}),
                },
            }]
            return msg, tcs, {}
        if self.calls == 5:
            msg = {"role": "assistant", "content": ""}
            tcs = [{
                "id": "5",
                "type": "function",
                "function": {
                    "name": "task",
                    "arguments": json.dumps({
                        "action": "finish",
                        "result": "report body from result only",
                        "outputs": {
                            "load_data": "gen/fragments/load_data_func.py",
                            "normalize_data": "gen/fragments/normalize_data_func.py",
                            "shard_data_distributed": (
                                "gen/fragments/shard_data_distributed_func.py"
                            ),
                        },
                    }),
                },
            }]
            return msg, tcs, {}
        raise AssertionError(f"unexpected extra LLM call #{self.calls}")


def test_agent_multi_file_incomplete_then_recover() -> None:
    from agent_stage import AgentStageRunner
    from config_loader import StageConfig

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "gen" / "prompts").mkdir(parents=True)
        (root / "gen" / "fragments").mkdir(parents=True)
        (root / "gen" / "prompts" / "sys.txt").write_text("sys", encoding="utf-8")
        stage = StageConfig(
            "inspect_data_shapes",
            {
                "kind": "agent",
                "system_prompt": {"type": "file", "path": "gen/prompts/sys.txt"},
                "prompt": "multi file",
                "tools": ["write", "task"],
                "max_turns": 10,
                "require_approval": False,
                "output": {
                    "all": {
                        "type": "file",
                        "path": "gen/fragments/data_shapes_report.txt",
                    },
                    "load_data": {
                        "type": "file",
                        "path": "gen/fragments/load_data_func.py",
                    },
                    "normalize_data": {
                        "type": "file",
                        "path": "gen/fragments/normalize_data_func.py",
                    },
                    "shard_data_distributed": {
                        "type": "file",
                        "path": "gen/fragments/shard_data_distributed_func.py",
                    },
                },
            },
        )
        engine = _FakeEngine(root)
        engine.llm_client = _FakeLLMMultiFileMissing()
        AgentStageRunner("inspect_data_shapes", stage, engine).execute()

        assert engine.llm_client.saw_incomplete
        assert (root / "gen" / "fragments" / "load_data_func.py").exists()
        report = root / "gen" / "fragments" / "data_shapes_report.txt"
        assert report.exists()
        # all filled from result when report file was not written by agent
        assert "report body" in report.read_text(encoding="utf-8")
    print("OK test_agent_multi_file_incomplete_then_recover")


def test_check_only_demo_config() -> None:
    """Static check against real analyses/kk_new demo if present."""
    repo = AGENT_DIR.parent
    workdir = repo / "analyses" / "kk_new"
    cfg = workdir / "llm_config_agent_demo.toml"
    if not cfg.exists():
        print("SKIP test_check_only_demo_config (demo config missing)")
        return
    from engine import Engine

    eng = Engine(workdir=str(workdir), config_name="llm_config_agent_demo.toml")
    eng.check()
    print("OK test_check_only_demo_config")


def test_check_kk_dis_fit_config() -> None:
    """Static parse/check against analyses/kk_dis multi-output agent stage."""
    repo = AGENT_DIR.parent
    workdir = repo / "analyses" / "kk_dis"
    cfg = workdir / "llm_config_fit.toml"
    if not cfg.exists():
        print("SKIP test_check_kk_dis_fit_config")
        return
    from config_loader import ConfigLoader

    loaded = ConfigLoader.load(cfg)
    stage = loaded.get_stage("inspect_data_shapes")
    assert stage is not None
    assert stage.kind == "agent"
    assert "load_data" in stage.output
    assert stage.output["load_data"]["path"].endswith("load_data_func.py")
    assert "normalize_data" in stage.output
    assert "shard_data_distributed" in stage.output
    print("OK test_check_kk_dis_fit_config")


def main() -> None:
    test_tool_registry_and_path_guard()
    test_config_loader_accepts_agent()
    test_agent_stage_runner_scripted()
    test_agent_require_approval_reject_then_approve()
    test_agent_multi_file_outputs_harvest()
    test_agent_multi_file_incomplete_then_recover()
    test_check_only_demo_config()
    test_check_kk_dis_fit_config()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
