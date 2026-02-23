from .base import Tool
from .builtin.read_file import ReadFileTool

__all__ = ["ReadFileTool"]


def get_all_builtin_tools() -> list[type[Tool]]:
    return [ReadFileTool]
