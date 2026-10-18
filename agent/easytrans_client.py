#!/usr/bin/env python3
# coding: utf-8
"""
极易云开放平台 API 客户端模块
基于 openai 库实现，支持对话补全、响应、嵌入等 API 接口

重试（审计 §3.5）
-----------------
网关在并发高时会直接返回 429：

    Error code: 429 - {'code': 'concurrency_limit_exceeded', ...}

以前这会直接终止整条流水线（实测：并行跑 route1/route4 时 route1 整条失败，
必须手动重跑）。现在 `_call_with_retry` 对**可重试**的错误做指数退避 + 抖动，
默认 5 次；不可重试的错误（401/403/404/参数错）立刻抛出，不浪费时间。

可调的环境变量：
    EASYTRANS_MAX_RETRIES   默认 5
    EASYTRANS_RETRY_BASE    退避基数秒，默认 1.0
    EASYTRANS_RETRY_MAX     单次退避上限秒，默认 30.0
"""

import os
import random
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# Reasoning-capable models for which the EasyTrans gateway refuses function
# tools on /v1/chat/completions unless reasoning_effort="none" (HTTP 400:
# "Function tools with reasoning_effort are not supported for <model>").
# Only these get the automatic default -- other models keep their normal
# reasoning behaviour. Extend the tuple as new restricted models appear.
_TOOL_REASONING_NONE_MODELS = ("gpt-6-sol",)

# Environment override for the reasoning effort sent to the gateway.
# <none|low|medium|high>; unset means "let the model decide" (see below).
_REASONING_EFFORT_ENV = "EASYTRANS_REASONING_EFFORT"

# ---------------------------------------------------------------- 重试策略
# 这些 HTTP 状态码值得重试：限流、网关/服务端临时故障。
_RETRYABLE_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 522, 524})

# 网络层错误（连不上 / 超时）也重试：openai 的 APIConnectionError 基类覆盖
# APITimeoutError。
try:  # pragma: no cover - 取决于 openai 版本
    from openai import APIConnectionError as _APIConnectionError
except Exception:                                        # noqa: BLE001
    class _APIConnectionError(Exception):                # type: ignore
        pass


def resolve_reasoning_effort(
    model: str,
    has_function_tools: bool,
    reasoning_effort: Optional[str] = None,
) -> Optional[str]:
    """Pick the reasoning_effort to send, or None to omit the parameter.

    Precedence:
      1. explicit ``reasoning_effort`` argument
      2. ``EASYTRANS_REASONING_EFFORT`` environment variable
      3. ``"none"`` when the call carries function tools and the model is
         known to reject them otherwise (e.g. gpt-6-sol)
    """
    if reasoning_effort is not None:
        return reasoning_effort

    env_value = os.getenv(_REASONING_EFFORT_ENV)
    if env_value:
        return env_value

    if has_function_tools and any(
        name in model for name in _TOOL_REASONING_NONE_MODELS
    ):
        return "none"

    return None


def _retry_config():
    """从环境读重试配置（每次调用时读，方便测试与运行时调整）。"""
    def _int(name, default):
        try:
            return int(os.getenv(name, "") or default)
        except ValueError:
            return default

    def _float(name, default):
        try:
            return float(os.getenv(name, "") or default)
        except ValueError:
            return default

    return (
        max(0, _int("EASYTRANS_MAX_RETRIES", 5)),
        max(0.0, _float("EASYTRANS_RETRY_BASE", 1.0)),
        max(0.0, _float("EASYTRANS_RETRY_MAX", 30.0)),
    )


def _status_of(exc: BaseException) -> Optional[int]:
    """取出 HTTP 状态码，取不到返回 None。"""
    for attr in ("status_code", "status"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    return code if isinstance(code, int) else None


def _code_of(exc: BaseException) -> str:
    """取出网关的业务错误码（如 concurrency_limit_exceeded）。"""
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        c = body.get("code")
        if isinstance(c, str):
            return c
    c = getattr(exc, "code", None)
    return c if isinstance(c, str) else ""


def is_retryable(exc: BaseException) -> bool:
    """这个异常值不值得重试？

    * 网络层错误（连接失败 / 超时）→ 是
    * HTTP 状态码在 `_RETRYABLE_STATUS` 里 → 是
    * 网关业务码里含限流/过载语义 → 是
    * 其余（鉴权、参数、404…）→ 否，立刻抛出
    """
    if isinstance(exc, _APIConnectionError):
        return True
    code = _status_of(exc)
    if code is not None:
        return code in _RETRYABLE_STATUS
    text = ("%s %s" % (_code_of(exc), exc)).lower()
    return any(k in text for k in
               ("concurrency_limit", "rate_limit", "too many requests",
                "overloaded", "temporarily unavailable", "timeout"))


def _sleep_for(attempt: int, base: float, cap: float) -> float:
    """第 attempt 次（从 0 数）失败后的退避秒数：指数 + 全抖动，封顶 cap。"""
    raw = base * (2 ** attempt)
    delay = random.uniform(0.0, min(raw, cap)) if cap > 0 else 0.0
    return delay


class EasyTransClient:
    """极易云开放平台 API 客户端"""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv('EASYTRANS_API_KEY')
        self.base_url = base_url or os.getenv('EASYTRANS_BASE_URL', 'https://api.easytransnote.com/v1')
        self.model = model or os.getenv('EASYTRANS_MODEL', 'gemini-2.5-pro')

        if not self.api_key:
            raise ValueError("API 密钥未设置，请在 .env 中配置 EASYTRANS_API_KEY")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

        self.logger = self._setup_logger()
        # 测试/诊断用：累计的重试次数
        self.retry_count = 0

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _call_with_retry(self, fn: Callable[[], Any], what: str) -> Any:
        """跑 `fn()`，对可重试错误做指数退避。

        `what` 只用于日志（"对话补全" / "响应" / "嵌入"）。
        """
        max_retries, base, cap = _retry_config()
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as e:                          # noqa: BLE001
                retryable = is_retryable(e)
                if not retryable or attempt >= max_retries:
                    self.logger.error(f"{what} API 调用失败: {e}")
                    if retryable:
                        raise EasyTransError(
                            f"{what} API 调用失败（已重试 {attempt} 次仍失败）: {e}"
                        ) from e
                    raise EasyTransError(f"{what} API 调用失败: {e}") from e
                delay = _sleep_for(attempt, base, cap)
                self.retry_count += 1
                self.logger.warning(
                    "%s API 调用失败（第 %d/%d 次重试，%.2fs 后）: %s",
                    what, attempt + 1, max_retries, delay, e,
                )
                if delay > 0:
                    time.sleep(delay)
                attempt += 1

    def chat_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        functions: Optional[List[Dict[str, Any]]] = None,
        function_call: Optional[Union[str, Dict[str, str]]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[Union[str, Dict[str, Any]]] = None,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        对话补全。

        ``reasoning_effort`` is forwarded to the gateway when set. When it is
        omitted, ``EASYTRANS_REASONING_EFFORT`` wins if present, and function
        tool calls to models listed in ``_TOOL_REASONING_NONE_MODELS``
        (e.g. gpt-6-sol) default to ``"none"`` -- the gateway rejects them
        otherwise (HTTP 400).
        """
        model = model or self.model

        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        if functions:
            kwargs["functions"] = functions
        if function_call:
            kwargs["function_call"] = function_call
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice

        effort = resolve_reasoning_effort(
            model=model,
            has_function_tools=bool(tools) or bool(functions),
            reasoning_effort=reasoning_effort,
        )
        if effort is not None:
            kwargs["reasoning_effort"] = effort

        def _do():
            return self.client.chat.completions.create(**kwargs).model_dump()

        result = self._call_with_retry(_do, "对话补全")
        self.logger.info(f"对话补全 API 调用成功，模型: {model}")
        return result

    def responses(
        self,
        input_text: str,
        model: Optional[str] = None,
        stream: bool = False,
        background: bool = False
    ) -> Dict[str, Any]:
        """响应 API - 通过 chat completions 接口模拟"""
        model = model or self.model

        def _do():
            return self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": input_text}],
                stream=stream,
            ).model_dump()

        result = self._call_with_retry(_do, "响应")
        self.logger.info(f"响应 API 调用成功，模型: {model}")
        return result

    def embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str = "text-embedding-3-small",
        encoding_format: str = "float"
    ) -> Dict[str, Any]:
        def _do():
            return self.client.embeddings.create(
                model=model,
                input=input_text,
                encoding_format=encoding_format,
            ).model_dump()

        result = self._call_with_retry(_do, "嵌入")
        self.logger.info(f"嵌入 API 调用成功，模型: {model}")
        return result

    def validate_response(self, response: Dict[str, Any]) -> bool:
        if 'choices' in response:
            choice = response['choices'][0]
            finish = choice.get('finish_reason')
            message = choice.get('message') or {}
            if finish in ('stop', 'tool_calls', 'function_call', None):
                return True
            if message.get('function_call') or message.get('tool_calls'):
                return True
            return False

        if 'status' in response:
            return response['status'] in ['completed', 'queued', 'processing']

        if 'role' in response and response['role'] == 'assistant':
            return 'content' in response

        if 'data' in response and isinstance(response['data'], list):
            return len(response['data']) > 0

        return False

    def extract_content(self, response: Dict[str, Any]) -> str:
        # Messages API 响应格式 (Claude 模型)
        if 'content' in response and isinstance(response['content'], list):
            content_list = response['content']
            if content_list and isinstance(content_list[0], dict):
                text = content_list[0].get('text', '')
                if text:
                    return text

        # 对话补全API响应格式
        if 'choices' in response and response['choices']:
            choice = response['choices'][0]

            finish_reason = choice.get('finish_reason')
            if finish_reason == 'length':
                self.logger.warning("响应因长度限制被截断")
                return ""

            message = choice.get('message', {})
            content = message.get('content', '')

            if not content and finish_reason:
                self.logger.warning(f"响应内容为空，finish_reason: {finish_reason}")

            return content

        # 响应API格式
        if 'output' in response and isinstance(response['output'], list):
            for item in response['output']:
                if item.get('type') == 'message' and 'content' in item:
                    for content in item['content']:
                        if 'text' in content:
                            return content['text']

        # 消息API格式
        if 'content' in response and isinstance(response['content'], list):
            for content in response['content']:
                if content.get('type') == 'text':
                    return content.get('text', '')

        return ""

    def extract_function_call(self, response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if 'choices' in response and response['choices']:
            choice = response['choices'][0]
            message = choice.get('message', {})
            return message.get('function_call')
        return None

    def extract_message(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Return the assistant message dict from a chat.completions response."""
        if 'choices' in response and response['choices']:
            message = response['choices'][0].get('message') or {}
            if isinstance(message, dict):
                return message
        return {}

    def extract_tool_calls(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract modern OpenAI tool_calls from a chat completion response.

        Each item: {"id": str, "type": "function", "function": {"name": str, "arguments": str}}
        Falls back to legacy single function_call when tool_calls is absent.
        """
        message = self.extract_message(response)
        tool_calls = message.get('tool_calls') or []
        if isinstance(tool_calls, list) and tool_calls:
            return [tc for tc in tool_calls if isinstance(tc, dict)]

        # Legacy functions API → normalize to one synthetic tool_call
        fn = message.get('function_call')
        if isinstance(fn, dict) and fn.get('name'):
            return [{
                "id": "function_call_0",
                "type": "function",
                "function": {
                    "name": fn.get("name", ""),
                    "arguments": fn.get("arguments") or "{}",
                },
            }]
        return []


class EasyTransError(Exception):
    """极易云 API 异常"""
    pass

if __name__ == "__main__":
    client = EasyTransClient()
    client.logger.info("极易云客户端初始化成功")

    try:
        response = client.responses(
            input_text="极易云开放平台是一个强大的AI服务平台",
            model="gpt-5-2025-08-07"
        )
        print(response)
        content = client.extract_content(response)
        client.logger.info(f"响应内容: {content}")
    except EasyTransError as e:
        client.logger.error(f"响应 API 调用失败: {e}")
