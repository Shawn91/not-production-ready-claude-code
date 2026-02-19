import asyncio
import sys
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
        async with Agent() as agent:
            self.agent = agent
            return await self._process_message(message)

    async def _process_message(self, message: str) -> str | None:
        if not self.agent:
            return None

        # 用于标记当前是否已经开始输出 AI 产生的内容，如果没有，则输出一些格式性的内容，提醒用户，AI准备开始输出了。
        # 如果当前已经开始输出，则不需要再输出格式性的内容了。
        assistant_streaming = False
        async for event in self.agent.run(message):
            if event.type == AgentEventType.TEXT_DELTA:
                content = event.data.get("content", "")
                if not assistant_streaming:
                    self.tui.begin_assistant()
                    assistant_streaming = True
                self.tui.stream_assistant_delta(content)


async def run(messages: list[dict[str, Any]]):
    client = LLMClient()
    messages = [{"role": "user", "content": "Hello"}]
    async for event in client.chat_completion(messages, True):
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
        if result is None:
            sys.exit(1)


if __name__ == "__main__":
    main()
