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


def main() -> None:
    test_tool_registry_and_path_guard()
    test_config_loader_accepts_agent()
    test_agent_stage_runner_scripted()
    test_check_only_demo_config()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
