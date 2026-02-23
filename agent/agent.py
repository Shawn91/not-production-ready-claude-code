from pathlib import Path
from typing import AsyncGenerator

from agent.events import AgentEvent, AgentEventType
from client.llm_client import LLMClient
from client.response import StreamEventType, ToolCall, ToolResultMessage
from context.manager import ContextManager
from tools.registry import create_default_tool_registry


class Agent:
    def __init__(self):
        self.client = LLMClient()
        self.context_manager = ContextManager()
        self.tool_registry = create_default_tool_registry()

    async def run(self, message: str):
        """给定消息历史，运行一轮 agent，返回一条消息。此外，还要负责发送事件消息等额外工作"""
        # 向外通知 agent 启动了
        yield AgentEvent.agent_start(message)
        self.context_manager.add_user_message(message)

        final_response = ""
        async for event in self._agentic_loop():
            yield event
            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content", "")

        self.context_manager.add_assistant_message(final_response)
        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        """给定消息历史，运行一轮 agent，返回一条消息"""
        response_text = ""
        tool_schemas = self.tool_registry.get_schemas()
        tool_calls: list[ToolCall] = []

        # client 返回的是 llm client events，这里接收到之后，要转换为 agent event
        async for event in self.client.chat_completion(
            self.context_manager.get_messages(),
            tools=tool_schemas if tool_schemas else None,
            stream=True,
        ):
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content=content)
            elif event.type == StreamEventType.TOOL_CALL_COMPLETE and event.tool_call:
                tool_calls.append(event.tool_call)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or "Unknown error")

        if response_text:
            yield AgentEvent.text_complete(content=response_text)

        # 依次执行所有的 tools
        tool_call_result_messages: list[ToolResultMessage] = []
        for tool_call in tool_calls:
            yield AgentEvent.tool_call_start(
                call_id=tool_call.call_id,
                name=tool_call.name,
                arguments=tool_call.arguments,
            )
            result = await self.tool_registry.invoke(
                name=tool_call.name, params=tool_call.arguments, cwd=Path.cwd()
            )
            yield AgentEvent.tool_call_complete(
                call_id=tool_call.call_id,
                name=tool_call.name,
                result=result,
            )
            tool_call_result_messages.append(
                ToolResultMessage(
                    tool_call_id=tool_call.call_id,
                    content=result.to_model_output(),
                    is_error=not result.success,
                )
            )
        for tool_result_message in tool_call_result_messages:
            self.context_manager.add_tool_result(
                tool_result_message.tool_call_id, tool_result_message.content
            )

    async def __aenter__(self) -> "Agent":
        return self

    # Agent class 退出后，自动清理与 llm client 的连接
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.client:
            await self.client.close()
