from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from client.response import TokenUsage
from config.config import Config
from prompts.system import get_system_prompt
from tools.base import Tool
from utils.text import count_tokens


@dataclass
class MessageItem:
    role: str
    content: str
    token_count: int | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    pruned_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"role": self.role}
        if self.tool_call_id:
            result["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        if self.content:
            result["content"] = self.content
        return result


class ContextManager:
    # 必须要同时满足以下2个条件，才会 prune tool outputs
    # 1. tool outputs 占用的 token 数量超过 PRUNE_PROTECT_TOKENS
    # 2. 如果 prune，至少能删除 PRUNE_MINIMUM_TOKENS 个 token
    PRUNE_PROTECT_TOKENS = 40000
    PRUNE_MINIMUM_TOKENS = 20000

    def __init__(
        self,
        config: Config,
        user_memory: str | None = None,
        tools: list[Tool] | None = None,
    ):
        self._system_prompt = get_system_prompt(
            config=config, user_memory=user_memory, tools=tools
        )
        self.config = config
        self._model_name = self.config.model_name or ""
        self._messages: list[MessageItem] = []
        # 假设完整聊天记录有 A、B、C三条消息。其中 A 和 C 是 user message，B 是 assistant message。
        # 那么 latest usage 表示 ABC 三条消息的 token 数量之和。
        # 而 total usage 是 A 的 token 数 + AB token 数 + ABC token 数之和。
        # 因为发送消息时，会带上历史消息，所以发送 B 时，A 又计算了一次token数。发送 C 时，AB 又计算了一次token数。
        # total usage 作用是为了统计花费的成本。latest usage 是为了判断是否需要做 context compacting。
        self._latest_usage: TokenUsage | None = None
        self._total_usage: TokenUsage | None = None

    def add_user_message(self, content: str) -> None:
        item = MessageItem(
            role="user",
            content=content,
            token_count=count_tokens(content, self._model_name or ""),
        )
        self._messages.append(item)

    def add_assistant_message(
        self, content: str, tool_calls: list[dict[str, Any]] | None = None
    ) -> None:
        item = MessageItem(
            role="assistant",
            content=content or "",
            token_count=count_tokens(content, self._model_name),
            tool_calls=tool_calls or [],
        )
        self._messages.append(item)

    def add_tool_result(self, tool_call_id: str, content: str) -> None:
        item = MessageItem(
            role="tool",
            content=content,
            tool_call_id=tool_call_id,
            token_count=count_tokens(content, self._model_name),
        )
        self._messages.append(item)

    def get_messages(self) -> list[dict[str, Any]]:
        messages = []
        if self._system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": self._system_prompt,
                }
            )
        messages.extend([item.to_dict() for item in self._messages])
        return messages

    def set_latest_usage(self, usage: TokenUsage) -> None:
        self._latest_usage = usage

    def add_usage(self, usage: TokenUsage) -> None:
        if self._total_usage is None:
            self._total_usage = usage
        else:
            self._total_usage += usage

    def needs_compression(self) -> bool:
        if not self._latest_usage:
            return False
        context_limit = self.config.model.context_window
        current_tokens = self._latest_usage.total_tokens
        return current_tokens > context_limit * 0.8

    def replace_with_summary(self, summary: str) -> None:
        self._messages = []
        continuation_content = f"""# Context Restoration (Previous Session Compacted)

            The previous conversation was compacted due to context length limits. Below is a detailed summary of the work done so far.

            **CRITICAL: Actions listed under "COMPLETED ACTIONS" are already done. DO NOT repeat them.**

            ---

            {summary}

            ---

            Resume work from where we left off. Focus ONLY on the remaining tasks."""

        summary_item = MessageItem(
            role="user",
            content=continuation_content,
            token_count=count_tokens(
                continuation_content, model=self.config.model.name
            ),
        )
        self._messages.append(summary_item)

        # 人工构造一个假的针对上面的 summary 的 AI 回复。这样可以更容易让 AI 按照设定的方案继续工作
        # 理论上来说，可以只要上面的 summary item，不要下面的 ack 和 continue items，
        # 但是 ack_content 和 continue_content 可以提高 AI 的效果
        ack_content = """I've reviewed the context from the previous session. I understand:
        - The original goal and what was requested
        - Which actions are ALREADY COMPLETED (I will NOT repeat these)
        - The current state of the project
        - What still needs to be done

        I'll continue with the REMAINING tasks only, starting from where we left off."""
        ack_item = MessageItem(
            role="assistant",
            content=ack_content,
            token_count=count_tokens(ack_content, self._model_name),
        )
        self._messages.append(ack_item)

        continue_content = (
            "Continue with the REMAINING work only. Do NOT repeat any completed actions. "
            "Proceed with the next step as described in the context above."
        )

        continue_item = MessageItem(
            role="user",
            content=continue_content,
            token_count=count_tokens(continue_content, self._model_name),
        )
        self._messages.append(continue_item)

    def prune_tool_outputs(self) -> int:
        """删除一些 tool 的输出，以减少 context 占用。返回值是删除的 token 数量。
        删除策略是删除旧消息，保留最近的
        """
        if len([m for m in self._messages if m.role == "user"]) < 2:
            return 0
        total_tokens = 0
        pruned_tokens = 0
        to_prune = []
        # 从新到旧遍历消息
        for msg in reversed(self._messages):
            if msg.role == "tool" and msg.tool_call_id:
                # 有一条消息已经 pruned，说明更旧的肯定也已经都 prune 过了，不用再检查了
                if msg.pruned_at:
                    break
                tokens = msg.token_count or count_tokens(msg.content, self._model_name)
                total_tokens += tokens

                if total_tokens > self.PRUNE_PROTECT_TOKENS:
                    pruned_tokens += tokens
                    to_prune.append(msg)
        if pruned_tokens < self.PRUNE_MINIMUM_TOKENS:
            return 0
        for msg in to_prune:
            msg.content = "[Old tool result content cleared]"
            msg.token_count = count_tokens(msg.content, self._model_name)
            msg.pruned_at = datetime.now()
        return pruned_tokens
