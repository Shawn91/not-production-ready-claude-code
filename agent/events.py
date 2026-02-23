from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from client.response import TokenUsage
from tools.base import ToolResult


class AgentEventType(str, Enum):
    # AGENT lifecycle
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    AGENT_ERROR = "agent_error"

    # text streaming
    TEXT_DELTA = "text_delta"
    TEXT_COMPLETE = "text_complete"

    # TOOL CALLS
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_COMPLETE = "tool_call_complete"


@dataclass
class AgentEvent:
    type: AgentEventType
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def agent_start(cls, message: str) -> "AgentEvent":
        return cls(type=AgentEventType.AGENT_START, data={"message": message})

    @classmethod
    def agent_end(
        cls, response: str | None, usage: TokenUsage | None = None
    ) -> "AgentEvent":
        return cls(
            type=AgentEventType.AGENT_END,
            data={"response": response, "usage": usage.__dict__ if usage else None},
        )

    @classmethod
    def agent_error(
        cls, error: str, details: dict[str, Any] | None = None
    ) -> "AgentEvent":
        return cls(
            type=AgentEventType.AGENT_ERROR,
            data={"error": error, "details": details or {}},
        )

    @classmethod
    def text_delta(cls, content: str) -> "AgentEvent":
        return cls(
            type=AgentEventType.TEXT_DELTA,
            data={"content": content},
        )

    @classmethod
    def text_complete(cls, content: str) -> "AgentEvent":
        return cls(
            type=AgentEventType.TEXT_COMPLETE,
            data={"content": content},
        )

    @classmethod
    def tool_call_start(
        cls, call_id: str, name: str, arguments: dict[str, Any]
    ) -> "AgentEvent":
        """表示 agent 准备开始调用某个工具"""
        return cls(
            type=AgentEventType.TOOL_CALL_START,
            data={"call_id": call_id, "name": name, "arguments": arguments},
        )

    @classmethod
    def tool_call_complete(
        cls, call_id: str, name: str, result: ToolResult
    ) -> "AgentEvent":
        """表示 agent 调用某个工具完成"""
        return cls(
            type=AgentEventType.TOOL_CALL_COMPLETE,
            data={
                "call_id": call_id,
                "name": name,
                "success": result.success,
                "error": result.error,
                "output": result.output,
                "metadata": result.metadata,
                "truncated": result.truncated,
            },
        )
