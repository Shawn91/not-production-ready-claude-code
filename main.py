import asyncio
import sys
from pathlib import Path

import click

from agent.agent import Agent
from agent.events import AgentEventType
from config.config import Config
from config.loader import load_config
from ui.tui import TUI, get_console
from utils.errors import ConfigError

console = get_console()


class CLI:
    def __init__(self, config: Config):
        self.agent: Agent | None = None
        self.tui = TUI(config, console)
        self.config = config

    async def run_single(self, message) -> str | None:
        """启动时就要传入 message"""
        async with Agent(config=self.config) as agent:
            self.agent = agent
            return await self._process_message(message)

    async def run_interactive(self) -> str | None:
        self.tui.print_welcome(
            "AI Agent",
            lines=[
                f"model: {self.config.model_name}",
                f"cwd: {self.config.cwd}",
                "commands: /help /config /approval /model /exit",
            ],
        )
        async with Agent(config=self.config) as agent:
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
        tool = self.agent.session.tool_registry.get(tool_name)
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
                console.print(f"\n[error]Error: {error}[/error]")
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
                    diff=event.data.get("diff"),
                    truncated=event.data.get("truncated", False),
                    exit_code=event.data.get("exit_code"),
                )

        return final_response


@click.command()
@click.argument("prompt", required=False)
@click.option(
    "--cwd",
    "-c",
    help="Current working directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def main(prompt: str | None, cwd: Path | None):
    try:
        config = load_config(cwd=cwd)
        config_errors = config.validate_config()
        if config_errors:
            for error in config_errors:
                console.print(f"[error]{error}[/error]")
            sys.exit(1)
    except Exception as e:
        raise ConfigError(f"Configuration Error: {e}")

    cli = CLI(config=config)

    if prompt:
        result = asyncio.run(cli.run_single(prompt))
        # 如果是 None，说明出现意料外的问题，直接退出程序
        if result is None:
            sys.exit(1)
    else:
        asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    main()
