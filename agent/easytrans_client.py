#!/usr/bin/env python3
# coding: utf-8
"""
极易云开放平台 API 客户端模块
基于 openai 库实现，支持对话补全、响应、嵌入等 API 接口
"""

import os
import logging
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


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
    ) -> Dict[str, Any]:
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

        try:
            response = self.client.chat.completions.create(**kwargs)
            result = response.model_dump()
            self.logger.info(f"对话补全 API 调用成功，模型: {model}")
            return result
        except Exception as e:
            self.logger.error(f"对话补全 API 调用失败: {e}")
            raise EasyTransError(f"对话补全 API 调用失败: {e}")

    def responses(
        self,
        input_text: str,
        model: Optional[str] = None,
        stream: bool = False,
        background: bool = False
    ) -> Dict[str, Any]:
        """响应 API - 通过 chat completions 接口模拟"""
        model = model or self.model

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": input_text}],
                stream=stream,
            )
            result = response.model_dump()
            self.logger.info(f"响应 API 调用成功，模型: {model}")
            return result
        except Exception as e:
            self.logger.error(f"响应 API 调用失败: {e}")
            raise EasyTransError(f"响应 API 调用失败: {e}")

    def embeddings(
        self,
        input_text: Union[str, List[str]],
        model: str = "text-embedding-3-small",
        encoding_format: str = "float"
    ) -> Dict[str, Any]:
        try:
            response = self.client.embeddings.create(
                model=model,
                input=input_text,
                encoding_format=encoding_format,
            )
            result = response.model_dump()
            self.logger.info(f"嵌入 API 调用成功，模型: {model}")
            return result
        except Exception as e:
            self.logger.error(f"嵌入 API 调用失败: {e}")
            raise EasyTransError(f"嵌入 API 调用失败: {e}")

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
