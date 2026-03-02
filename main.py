import asyncio
import sys
from pathlib import Path
from typing import Any

import click

from agent.agent import Agent
from agent.events import AgentEvent, AgentEventType
from client.llm_client import LLMClient
from ui.tui import TUI, get_console

console = get_console()


class CLI:
    def __init__(self):
        self.agent: Agent | None = None
        self.tui = TUI(console)

    async def run_single(self, message) -> str | None:
        """启动时就要传入 message"""
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)

    async def run_interactive(self) -> str | None:
        self.tui.print_welcome(
            "AI Agent",
            lines=[
                "model: gpt 5.2",
                f"cwd: {Path.cwd()}",
                "commands: /help /config /approval /model /exit",
            ],
        )
        async with Agent() as agent:
            self.agent = agent
            while True:
                try:
                    user_input = console.input("\n[user]>[/user] ").strip()
                    if not user_input:
                        continue
                    elif user_input == "/exit":
                        break
                    await self._process_message(user_input)
                except KeyboardInterrupt:
                    console.print("\n[dim]Use /exit to quit[/dim]")
                except EOFError:
                    break
        console.print("\n[dim]Goodbye![/dim]")

    def _get_tool_kind(self, tool_name: str) -> str | None:
        if not self.agent:
            return None
        tool = self.agent.tool_registry.get(tool_name)
        if not tool:
            return None
        return tool.kind.value

    async def _process_message(self, message: str) -> str | None:
        """
        正常情况下，返回字符串（AI最终的回答）。如果返回 None，说明出现意料外的问题
        """
        if not self.agent:
            return None

        # 用于标记当前是否已经开始输出 AI 产生的内容，如果没有，则输出一些格式性的内容，提醒用户，AI准备开始输出了。
        # 如果当前已经开始输出，则不需要再输出格式性的内容了。
        assistant_streaming = False
        final_response: str | None = None
        async for event in self.agent.run(message):
            print(event)
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_streaming:
                    self.tui.begin_assistant()
                    assistant_streaming = True
                self.tui.stream_assistant_delta(content)
            elif event.type == AgentEventType.TEXT_COMPLETE:
                final_response = event.data.get("content", "")
                if assistant_streaming:
                    self.tui.end_assistant()
                    assistant_streaming = False
            elif event.type == AgentEventType.AGENT_ERROR:
                error = event.data.get("error", "Unknown error")
                console.print(f"\n[error]Error: {error}")
            elif event.type == AgentEventType.TOOL_CALL_START:
                tool_name = event.data.get("name", "unknown")
                tool_kind = self._get_tool_kind(tool_name)
                self.tui.tool_call_start(
                    call_id=event.data.get("call_id", ""),
                    name=tool_name,
                    tool_kind=tool_kind,
                    arguments=event.data.get("arguments", {}),
                )
            elif event.type == AgentEventType.TOOL_CALL_COMPLETE:
                tool_name = event.data.get("name", "unknown")
                self.tui.tool_call_complete(
                    call_id=event.data.get("call_id", ""),
                    name=tool_name,
                    tool_kind=self._get_tool_kind(tool_name),
                    success=event.data.get("success", False),
                    output=event.data.get("output", ""),
                    error=event.data.get("error", ""),
                    metadata=event.data.get("metadata", {}),
                    truncated=event.data.get("truncated", False),
                )

        return final_response


async def run(messages: list[dict[str, Any]]):
    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]
    async for event in client.chat_completion(messages):
        print(event)


@click.command()
@click.argument("prompt", required=False)
def main(prompt: str | None = None):
    cli = CLI()
    messages = (
        [{"role": "user", "content": prompt}]
        if prompt
        else [{"role": "user", "content": "Hello"}]
    )
    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        # 如果是 None，说明出现意料外的问题，直接退出程序
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()
