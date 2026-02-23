import asyncio
import os
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

from client.response import (
    StreamEvent,
    StreamEventType,
    TextDelta,
    TokenUsage,
    ToolCall,
    ToolCallDelta,
    parse_tool_call_arguments,
)

load_dotenv()


class LLMClient:
    def __init__(self):
        self._client: AsyncOpenAI | None = None
        self._max_retries: int = 3

    def get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=os.getenv("API_KEY"),
                base_url="https://api.poe.com/v1",
            )
        return self._client

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None

    def _build_tools(self, tools: list[dict[str, Any]]):
        """这里的 tools 实际上是 Tool 的 to_openai_schema 的输出。
        但是 Tool 的 to_openai_schema 的直接输出还需要一些处理，在这里处理
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get(
                        "parameters", {"type": "object", "properties": {}}
                    ),
                },
            }
            for tool in tools
        ]

    async def chat_completion(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        client = self.get_client()
        kwargs = {
            "model": "poe/glm-5",
            "messages": messages,
            "stream": stream,
        }
        if tools:
            kwargs["tools"] = self._build_tools(tools)
            kwargs["tool_choice"] = "auto"
        for attempt in range(self._max_retries):
            try:
                if stream:
                    async for event in self._stream_response(client, kwargs):
                        yield event
                else:
                    event = await self._non_stream_response(client, kwargs)
                    yield event
                return
            except RateLimitError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR, error=f"Rate limit exceeded {e}"
                    )
                    return
            except APIConnectionError as e:
                if attempt < self._max_retries:
                    wait_time = 2**attempt
                    await asyncio.sleep(wait_time)
                else:
                    yield StreamEvent(
                        type=StreamEventType.ERROR, error=f"Connection error {e}"
                    )
                    return
            except APIError as e:
                yield StreamEvent(type=StreamEventType.ERROR, error=f"API error {e}")
                return

    async def _stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> AsyncGenerator[StreamEvent, None]:
        response = await client.chat.completions.create(**kwargs)
        usage: TokenUsage | None = None
        finish_reason: str | None = None
        # ai 返回结果中定义的 tool_calls 数据结构。作为key的整数是 index
        tool_calls: dict[int, dict[str, Any]] = {}
        async for chunk in response:
            if hasattr(chunk, "usage") and chunk.usage:
                usage = TokenUsage(
                    prompt_tokens=chunk.usage.prompt_tokens,
                    completion_tokens=chunk.usage.completion_tokens,
                    total_tokens=chunk.usage.total_tokens,
                    cached_tokens=chunk.usage.prompt_tokens_details.cached_tokens
                    if chunk.usage.prompt_tokens_details
                    else 0,
                )
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            delta = choice.delta
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            if delta.content:
                yield StreamEvent(
                    type=StreamEventType.TEXT_DELTA,
                    text_delta=TextDelta(delta.content),
                    finish_reason=finish_reason,
                )

            if delta.tool_calls:
                for tool_call_delta in delta.tool_calls:
                    idx = tool_call_delta.index
                    # AI返回的一个完整的 tool call 信息往往分成多个 chunks返回，第一个chunk包含id/name，后续chunks包含arguments
                    # 所以要对chunks进行拼接，才能拿到完整的一个 tool call信息
                    if idx not in tool_calls:
                        tool_calls[idx] = {
                            "id": tool_call_delta.id,
                            "name": "",
                            "arguments": "",  # ai 返回的是转换为字符串的 arguments
                        }

                    if tool_call_delta.id:
                        tool_calls[idx]["id"] = tool_call_delta.id

                    if tool_call_delta.function:
                        fn = tool_call_delta.function
                        if fn.name:
                            previous_name = tool_calls[idx]["name"]
                            tool_calls[idx]["name"] = fn.name
                            # 首次拿到 tool name 时，才对外 emit tool call start 事件
                            if not previous_name:
                                yield StreamEvent(
                                    type=StreamEventType.TOOL_CALL_START,
                                    tool_call_delta=ToolCallDelta(
                                        call_id=tool_calls[idx]["id"],
                                        name=fn.name,
                                    ),
                                )

                        if fn.arguments:
                            tool_calls[idx]["arguments"] += fn.arguments
                            yield StreamEvent(
                                type=StreamEventType.TOOL_CALL_DELTA,
                                tool_call_delta=ToolCallDelta(
                                    call_id=tool_calls[idx]["id"],
                                    name=tool_calls[idx]["name"] or fn.name,
                                    arguments_delta=fn.arguments,
                                ),
                            )
        for idx, tool_call in tool_calls.items():
            yield StreamEvent(
                type=StreamEventType.TOOL_CALL_COMPLETE,
                tool_call=ToolCall(
                    call_id=tool_call["id"],
                    name=tool_call["name"],
                    arguments=parse_tool_call_arguments(tool_call["arguments"]),
                ),
            )

        yield StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE,
            finish_reason=finish_reason,
            usage=usage,
        )

    async def _non_stream_response(
        self, client: AsyncOpenAI, kwargs: dict[str, Any]
    ) -> StreamEvent:
        response = await client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        message = choice.message
        text_delta = None
        if message.content:
            text_delta = TextDelta(content=message.content)

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tool_call in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        call_id=tool_call["id"],
                        name=tool_call["name"],
                        arguments=parse_tool_call_arguments(tool_call["arguments"]),
                    )
                )

        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                cached_tokens=response.usage.prompt_tokens_details.cached_tokens,
            )
        else:
            usage = None
        return StreamEvent(
            type=StreamEventType.MESSAGE_COMPLETE, text_delta=text_delta, usage=usage
        )
