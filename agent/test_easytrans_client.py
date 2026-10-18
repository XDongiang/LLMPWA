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


# --------------------------------------------------------------- 重试（§3.5）
class _StatusBoom(Exception):
    """带 HTTP 状态码的假异常，形状与 openai.APIStatusError 一致。"""

    def __init__(self, status_code: int) -> None:
        super().__init__("boom")
        self.status_code = status_code


@contextmanager
def _retry_env(max_retries: int = 3, base: float = 0.0):
    """把重试参数设成确定性、不真睡的取值。"""
    keys = ("EASYTRANS_MAX_RETRIES", "EASYTRANS_RETRY_BASE", "EASYTRANS_RETRY_MAX")
    saved = {k: os.environ.get(k) for k in keys}
    os.environ["EASYTRANS_MAX_RETRIES"] = str(max_retries)
    os.environ["EASYTRANS_RETRY_BASE"] = str(base)
    os.environ["EASYTRANS_RETRY_MAX"] = "0"
    try:
        yield
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


def test_is_retryable_classification() -> None:
    """429/5xx/网络错误重试；鉴权/参数错误立刻抛出。"""
    from easytrans_client import is_retryable

    for code in (408, 429, 500, 502, 503, 504):
        assert is_retryable(_StatusBoom(code)), code
    for code in (400, 401, 403, 404, 422):
        assert not is_retryable(_StatusBoom(code)), code
    # 网关业务码（审计 §3.5 实测撞到的那个）
    assert is_retryable(Exception(
        "Error code: 429 - {'code': 'concurrency_limit_exceeded'}"))
    assert not is_retryable(Exception("invalid request payload"))

    print("OK test_is_retryable_classification")


def test_retry_then_success() -> None:
    """429 连续两次后成功：应重试并返回结果。"""
    client, _ = _make_client()
    state = {"n": 0}

    def flaky():
        state["n"] += 1
        if state["n"] < 3:
            raise _StatusBoom(429)
        return {"choices": [{"message": {"content": "hi"}}]}

    with _retry_env(max_retries=3):
        out = client._call_with_retry(flaky, "test")

    assert out["choices"][0]["message"]["content"] == "hi"
    assert state["n"] == 3, state
    assert client.retry_count == 2, client.retry_count
    print("OK test_retry_then_success")


def test_non_retryable_raises_immediately() -> None:
    """401 不该重试：只调用一次就抛出。"""
    from easytrans_client import EasyTransError

    client, _ = _make_client()
    state = {"n": 0}

    def unauthorized():
        state["n"] += 1
        raise _StatusBoom(401)

    with _retry_env(max_retries=3):
        try:
            client._call_with_retry(unauthorized, "test")
        except EasyTransError:
            pass
        else:
            raise AssertionError("401 竟然没有抛出")

    assert state["n"] == 1, state
    assert client.retry_count == 0, client.retry_count
    print("OK test_non_retryable_raises_immediately")


def test_retries_exhausted() -> None:
    """一直 503：共调用 1 + max_retries 次后抛出。"""
    from easytrans_client import EasyTransError

    client, _ = _make_client()
    state = {"n": 0}

    def always_down():
        state["n"] += 1
        raise _StatusBoom(503)

    with _retry_env(max_retries=3):
        try:
            client._call_with_retry(always_down, "test")
        except EasyTransError as e:
            assert "重试 3 次" in str(e), str(e)
        else:
            raise AssertionError("耗尽重试后竟然没有抛出")

    assert state["n"] == 4, state
    assert client.retry_count == 3, client.retry_count
    print("OK test_retries_exhausted")


def test_chat_completion_retries_through_public_api() -> None:
    """公共接口 chat_completion 也要走重试（不只是内部 helper）。"""
    client, completions = _make_client(model="gpt-6-sol")
    state = {"n": 0}
    original = completions.create

    def flaky_create(**kwargs):
        state["n"] += 1
        if state["n"] < 2:
            raise _StatusBoom(429)
        return original(**kwargs)

    completions.create = flaky_create  # type: ignore[assignment]
    with _retry_env(max_retries=3):
        out = client.chat_completion(messages=[{"role": "user", "content": "hi"}])

    assert out["choices"][0]["message"]["content"] == "ok"
    assert state["n"] == 2, state
    print("OK test_chat_completion_retries_through_public_api")


def main() -> None:
    test_resolve_reasoning_effort()
    test_chat_completion_forwards_reasoning_effort()
    test_llm_client_chat_with_tools()
    test_is_retryable_classification()
    test_retry_then_success()
    test_non_retryable_raises_immediately()
    test_retries_exhausted()
    test_chat_completion_retries_through_public_api()
    print("\nAll easytrans_client tests passed.")


if __name__ == "__main__":
    main()
