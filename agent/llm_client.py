"""
llm_client.py — LLM 调用封装，对上层屏蔽 API 细节。
改名自 easytrans_client.py，保持向后兼容。
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()


def _strip_fences(code: str) -> str:
    code = re.sub(r"^```\w*\n?", "", code.strip())
    code = re.sub(r"\n?```$", "", code.strip())
    return code


class LLMClient:
    def __init__(
        self,
        model: Optional[str] = None,
        model_check: Optional[str] = None,
    ) -> None:
        from easytrans_client import EasyTransClient
        self._client = EasyTransClient()
        self.model = model or os.getenv("EASYTRANS_MODEL", "gemini-2.5-pro")
        self.model_check = model_check or os.getenv("EASYTRANS_MODEL_CHECK", self.model)

    def call(self, prompt: str, check: bool = False) -> str:
        from easytrans_client import EasyTransError

        response = self._client.responses(input_text=prompt, model=self.model)
        if not self._client.validate_response(response):
            raise EasyTransError("LLM response validation failed")
        code = self._client.extract_content(response)
        if not code:
            raise EasyTransError("LLM returned empty content")
        code = _strip_fences(code)

        if check:
            check_prompt = (
                f"{prompt}\nPlease strictly check the following code for compliance "
                f"with all the rules mentioned in the prompt. If any rule is violated, "
                f"regenerate the code until it fully complies.\nCode:\n{code}"
            )
            resp2 = self._client.responses(input_text=check_prompt, model=self.model_check)
            code = _strip_fences(self._client.extract_content(resp2))

        return code

    def chat_with_tools(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model: Optional[str] = None,
        tool_choice: Optional[Any] = "auto",
        temperature: float = 1.0,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
        """
        One agent turn: chat.completions with tools.

        Returns:
            (assistant_message, tool_calls, raw_response)
        """
        from easytrans_client import EasyTransError

        used_model = model or self.model
        response = self._client.chat_completion(
            messages=messages,
            model=used_model,
            temperature=temperature,
            tools=tools or None,
            tool_choice=tool_choice if tools else None,
        )
        if not self._client.validate_response(response):
            raise EasyTransError(
                "LLM chat_with_tools response validation failed: "
                f"keys={list(response.keys())}"
            )
        message = self._client.extract_message(response)
        if not message:
            raise EasyTransError(
                "LLM chat_with_tools returned empty assistant message. "
                f"response keys={list(response.keys())}"
            )
        tool_calls = self._client.extract_tool_calls(response)
        return message, tool_calls, response
