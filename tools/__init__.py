from tools.builtin.edit_file import EditTool
from tools.builtin.write_file import WriteFileTool

from .base import Tool
from .builtin.read_file import ReadFileTool

__all__ = ["ReadFileTool"]


def get_all_builtin_tools() -> list[type[Tool]]:
    return [ReadFileTool, WriteFileTool, EditTool]
