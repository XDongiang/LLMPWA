"""
Offline regression tests for reasoning_effort forwarding in the EasyTrans client.

Background: the gateway rejects function tools on /v1/chat/completions for
gpt-6-sol unless reasoning_effort="none" (HTTP 400:
"Function tools with reasoning_effort are not supported for gpt-6-sol").
agent/easytrans_client.py previously dropped the parameter entirely.

No real API calls are made -- the OpenAI client is replaced with a recorder.

Run from the repo root:
  python agent/test_easytrans_client.py
"""

from __future__ import annotations

import os
import sys
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

AGENT_DIR = Path(__file__).resolve().parent
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


class _FakeResponse:
    def model_dump(self) -> Dict[str, Any]:
        return {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ]
        }


class _RecordingCompletions:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse()


class _FakeOpenAI:
    def __init__(self) -> None:
        self.completions = _RecordingCompletions()
        self.chat = types.SimpleNamespace(completions=self.completions)


@contextmanager
def _fake_openai_module():
    """Replace the real OpenAI client so no HTTP client is ever constructed."""
    import easytrans_client as ec

    fake = _FakeOpenAI()
    original = ec.OpenAI
    ec.OpenAI = lambda **kwargs: fake  # type: ignore[assignment]
    try:
        yield fake
    finally:
        ec.OpenAI = original  # type: ignore[assignment]


def _make_client(model: str = "gpt-6-sol"):
    import easytrans_client as ec

    with _fake_openai_module() as fake:
        client = ec.EasyTransClient(api_key="test-key", model=model)
    return client, fake.completions


@contextmanager
def _clean_env():
    """Make EASYTRANS_REASONING_EFFORT deterministic around a test."""
    saved = os.environ.pop("EASYTRANS_REASONING_EFFORT", None)
    try:
        yield
    finally:
        os.environ.pop("EASYTRANS_REASONING_EFFORT", None)
        if saved is not None:
            os.environ["EASYTRANS_REASONING_EFFORT"] = saved


TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {}}}]


def test_resolve_reasoning_effort() -> None:
    from easytrans_client import resolve_reasoning_effort

    with _clean_env():
        # gpt-6-sol needs "none" only when function tools are present
        assert resolve_reasoning_effort("gpt-6-sol", True) == "none"
        assert resolve_reasoning_effort("gpt-6-sol", False) is None
        # other models keep the parameter unset
        assert resolve_reasoning_effort("gpt-5.6-sol", True) is None
        assert resolve_reasoning_effort("gemini-2.5-pro", True) is None
        # explicit argument always wins
        assert resolve_reasoning_effort("gpt-6-sol", True, "high") == "high"
        assert resolve_reasoning_effort("gpt-6-sol", False, "low") == "low"

        # env override wins over the compatibility default
        os.environ["EASYTRANS_REASONING_EFFORT"] = "medium"
        assert resolve_reasoning_effort("gpt-6-sol", True) == "medium"
        assert resolve_reasoning_effort("gemini-2.5-pro", True) == "medium"
        # ... but not over an explicit argument
        assert resolve_reasoning_effort("gpt-6-sol", True, "high") == "high"

    print("OK test_resolve_reasoning_effort")


def test_chat_completion_forwards_reasoning_effort() -> None:
    with _clean_env():
        client, rec = _make_client("gpt-6-sol")
        client.chat_completion(messages=[{"role": "user", "content": "hi"}], tools=TOOLS)
        assert rec.calls[-1]["reasoning_effort"] == "none", rec.calls[-1]

        # tool-free call is untouched
        client.chat_completion(messages=[{"role": "user", "content": "hi"}])
        assert "reasoning_effort" not in rec.calls[-1], rec.calls[-1]

        # a model that does not need the workaround is untouched
        client2, rec2 = _make_client("gpt-5.6-sol")
        client2.chat_completion(messages=[{"role": "user", "content": "hi"}], tools=TOOLS)
        assert "reasoning_effort" not in rec2.calls[-1], rec2.calls[-1]

        # explicit parameter is forwarded verbatim, even without tools
        client.chat_completion(
            messages=[{"role": "user", "content": "hi"}], reasoning_effort="low"
        )
        assert rec.calls[-1]["reasoning_effort"] == "low", rec.calls[-1]

        # legacy `functions` also counts as function tools
        legacy = [{"name": "f", "parameters": {}}]
        client.chat_completion(
            messages=[{"role": "user", "content": "hi"}], functions=legacy
        )
        assert rec.calls[-1]["reasoning_effort"] == "none", rec.calls[-1]

    print("OK test_chat_completion_forwards_reasoning_effort")


def test_llm_client_chat_with_tools() -> None:
    """The agent path (LLMClient.chat_with_tools) reaches the gateway patched."""
    from llm_client import LLMClient

    with _clean_env(), _fake_openai_module() as fake:
        llm = LLMClient(model="gpt-6-sol")
        message, tool_calls, _raw = llm.chat_with_tools(
            messages=[{"role": "user", "content": "hi"}], tools=TOOLS
        )
        assert message.get("content") == "ok"
        assert fake.completions.calls
        assert fake.completions.calls[-1]["reasoning_effort"] == "none", (
            fake.completions.calls
        )

    print("OK test_llm_client_chat_with_tools")


def main() -> None:
    test_resolve_reasoning_effort()
    test_chat_completion_forwards_reasoning_effort()
    test_llm_client_chat_with_tools()
    print("\nAll easytrans_client tests passed.")


if __name__ == "__main__":
    main()
