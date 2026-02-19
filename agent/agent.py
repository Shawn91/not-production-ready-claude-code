from typing import AsyncGenerator

from agent.events import AgentEvent, AgentEventType
from client.llm_client import LLMClient
from client.response import StreamEventType


class Agent:
    def __init__(self):
        self.client = LLMClient()

    async def run(self, message: str):
        """给定消息历史，运行一轮 agent，返回一条消息。此外，还要负责发送消息等额外工作"""
        # 向外通知 agent 启动了
        yield AgentEvent.agent_start(message)
        final_response = ""
        async for event in self._agentic_loop():
            yield event
            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content")
        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        """给定消息历史，运行一轮 agent，返回一条消息"""
        messages = [{"role": "user", "content": "Hey."}]
        response_text = ""
        # client 返回的是 llm client events，这里接收到之后，要转换为 agent event
        async for event in self.client.chat_completion(messages, True):
            if event.type == StreamEventType.TEXT_DELTA:
                if event.text_delta:
                    content = event.text_delta.content
                    response_text += content
                    yield AgentEvent.text_delta(content=content)
            elif event.type == StreamEventType.ERROR:
                yield AgentEvent.agent_error(event.error or "Unknown error")

        if response_text:
            yield AgentEvent.text_complete(content=response_text)

    async def __aenter__(self) -> "Agent":
        return self

    # Agent class 退出后，自动清理与 llm client 的连接
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.client:
            await self.client.close()
