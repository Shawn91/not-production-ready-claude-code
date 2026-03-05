import json
from typing import AsyncGenerator

from agent.events import AgentEvent, AgentEventType
from agent.session import Session
from client.response import StreamEventType, ToolCall, ToolResultMessage
from config.config import Config


class Agent:
    def __init__(self, config: Config):
        self.config = config
        self.session = Session(config=config)

    async def run(self, message: str):
        """给定消息历史，运行一轮 agent。此外，还要负责发送事件消息等额外工作"""
        # 向外通知 agent 启动了
        yield AgentEvent.agent_start(message)
        self.session.context_manager.add_user_message(message)

        final_response = ""
        async for event in self._agentic_loop():
            yield event
            if event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content", "")

        self.session.context_manager.add_assistant_message(final_response)
        yield AgentEvent.agent_end(final_response)

    async def _agentic_loop(self) -> AsyncGenerator[AgentEvent, None]:
        """给定消息历史，运行一轮 agent，这里的“一轮”指的是AI认为消息历史中，要求的任务都完成了。
        假设最新的一条消息要求 AI 做 3 件事，那么 AI 返回的第1条完整消息可能包含N个 tool calls，用于完成第一个任务。
        将这 N 个 tool calls 执行完毕，执行结果又发给 AI，AI然后反而第2条消息，包含M个tool calls，用于完成第2个任务
        依次下去，最终AI总共返回了多条消息，直到返回的某条消息不再包含 tool calls，说明AI认为要求完成的事都完成了，
        这就是所谓“一轮”
        """
        for turn_num in range(self.config.max_turns):
            # 将 session 内部的 turn 计数 +1
            self.session.increment_turn()
            response_text = ""
            tool_schemas = self.session.tool_registry.get_schemas()
            tool_calls: list[ToolCall] = []

            # client 返回的是 llm client events，这里接收到之后，要转换为 agent event
            async for event in self.session.client.chat_completion(
                self.session.context_manager.get_messages(),
                tools=tool_schemas if tool_schemas else None,
                stream=True,
            ):
                if event.type == StreamEventType.TEXT_DELTA:
                    if event.text_delta:
                        content = event.text_delta.content
                        response_text += content
                        yield AgentEvent.text_delta(content=content)
                elif (
                    event.type == StreamEventType.TOOL_CALL_COMPLETE and event.tool_call
                ):
                    tool_calls.append(event.tool_call)
                elif event.type == StreamEventType.ERROR:
                    yield AgentEvent.agent_error(event.error or "Unknown error")

            self.session.context_manager.add_assistant_message(
                response_text,
                [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in tool_calls
                ]
                if tool_calls
                else None,
            )

            if response_text:
                yield AgentEvent.text_complete(content=response_text)

            if not tool_calls:
                return

            # 依次执行所有的 tools
            tool_call_result_messages: list[ToolResultMessage] = []
            for tool_call in tool_calls:
                yield AgentEvent.tool_call_start(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    arguments=tool_call.arguments,
                )
                result = await self.session.tool_registry.invoke(
                    name=tool_call.name, params=tool_call.arguments, cwd=self.config.cwd
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
                self.session.context_manager.add_tool_result(
                    tool_result_message.tool_call_id, tool_result_message.content
                )

        yield AgentEvent.agent_error(f"Maximum turns ({self.config.max_turns}) reached")

    async def __aenter__(self) -> "Agent":
        return self

    # Agent class 退出后，自动清理与 llm client 的连接
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.session and self.session.client:
            await self.session.client.close()
