import os
from typing import Any

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from utils.paths import resolve_path

AGENT_THEME = Theme(
    {
        # General
        "info": "cyan",
        "warning": "yellow",
        "error": "bright_red bold",
        "success": "green",
        "dim": "dim",
        "muted": "grey50",
        "border": "grey35",
        "highlight": "bold cyan",
        # Rules
        "user": "bright_blue bold",
        "assistant": "bright_white",
        # Tools
        "tool": "bright_magenta bold",
        "tool.read": "cyan",
        "tool.write": "yellow",
        "tool.shell": "magenta",
        "tool.network": "bright_blue",
        "tool.memory": "green",
        "tool.mcp": "bright_cyan",
        # Code / blocks
        "code": "white",
    }
)

_console: Console | None = None


def get_console() -> Console:
    global _console
    if _console is None:
        _console = Console(theme=AGENT_THEME, highlight=False)
    return _console


class TUI:
    def __init__(self, console: Console | None = None):
        self.console = console or get_console()
        # 用于标记当前是否正在输出AI对话内容
        self._assistant_stream_open = False
        # 方便在命令行展示tool被调用时的参数
        self._tool_args_by_call_id: dict[str, dict[str, Any]] = {}
        # 当前工作目录
        self.cwd = os.getcwd()

    def begin_assistant(self):
        """开始输出AI对话内容前，先输出格式化内容，提醒AI要开始输出了"""
        self.console.print()
        self.console.print(Rule(Text("Assistant", style="assistant")))
        self._assistant_stream_open = True

    def end_assistant(self):
        """结束输出AI对话内容后，UI上进行一些首尾工作"""
        if self._assistant_stream_open:
            self.console.print()
        self._assistant_stream_open = False

    def stream_assistant_delta(self, content: str) -> None:
        """将ai流式返回的一块内容输出到终端"""
        self.console.print(content, end="", markup=False)

    def _ordered_args(self, tool_name: str, args: dict[str, Any]) -> list[tuple]:
        """用于将一个 tool 函数的所有参数按照特定顺序排序"""
        _PREFERED_ORDER = {
            "read_file": [
                "path",
                "offset",
                "limit",
            ],  # 对于 read_file tool，按照这个顺序显示参数
        }
        ordered = []
        prefered = _PREFERED_ORDER.get(tool_name, [])
        for key in prefered:
            if key in args:
                ordered.append((key, args[key]))
        # AI 可能会给出意料外的参数，但是只要AI给了，就也要添加到 ordered中
        remaining_keys = set(args.keys()) - set(prefered)
        for key in remaining_keys:
            ordered.append((key, args[key]))
        return ordered

    def _render_args_table(self, tool_name: str, args: dict[str, Any]) -> Table:
        """将一个 tool 函数的所有参数打印在一个 table 中"""
        table = Table.grid(padding=(0, 1))
        table.add_column(style="muted", justify="right", no_wrap=True)
        table.add_column(style="code", overflow="fold")
        for key, value in self._ordered_args(tool_name, args):
            table.add_row(key, value)
        return table

    def tool_call_start(
        self, call_id: str, name: str, tool_kind: str | None, arguments: dict[str, Any]
    ):
        self._tool_args_by_call_id[call_id] = arguments
        # 根据 tool kind 决定 tool call 在命令行中的边框样式
        border_style = f"tool.{tool_kind}" if tool_kind else "tool"
        title = Text.assemble(
            ("* ", "muted"),  # 第一个元素是内容，第二个元素是基于_THEME的样式名
            (name, "tool"),
            (" ", "muted"),
            (f"#{call_id[:8]}", "muted"),
        )

        # 参数中如果有文件路径，那么和当前的工作目录粘贴在一起展示给用户
        copied_args = dict(arguments)
        for key in ("path", "cws"):
            val = copied_args.get(key)
            if isinstance(val, str) and self.cwd:
                copied_args[key] = str(resolve_path(self.cwd, val))

        panel = Panel(
            self._render_args_table(tool_name=name, args=copied_args)
            if copied_args
            else Text("(no args)", style="muted"),
            title=title,
            subtitle=Text("running", style="muted"),
            title_align="left",
            subtitle_align="right",
            padding=(1, 2),
            box=box.ROUNDED,
            border_style=border_style,
        )
        self.console.print()
        self.console.print(panel)
