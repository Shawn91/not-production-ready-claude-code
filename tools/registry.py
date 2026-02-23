import logging
from pathlib import Path
from typing import Any

from tools import get_all_builtin_tools
from tools.base import Tool, ToolInvocation, ToolResult

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self._tools:
            logger.warning(f"Overwriting existing tool: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def get_tools(self) -> list[Tool]:
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """返回所有 tools 的 openai schema"""
        return [t.to_openai_schema() for t in self.get_tools()]

    async def invoke(self, name: str, params: dict[str, Any], cwd: Path):
        """调用一个 tool"""
        tool = self.get(name)
        if tool is None:
            return ToolResult.error_result(
                f"Unknown tool: {name}", metadata={"tool_name": name}
            )
        validation_errors = tool.validate_params(params)
        if validation_errors:
            return ToolResult.error_result(
                f"Invalid parameters for tool {name}: {'; '.join(validation_errors)}",
                metadata={"tool_name": name, "validation_errors": validation_errors},
            )
        try:
            invocation = ToolInvocation(params=params, cwd=cwd)
            result = await tool.execute(invocation)
        except Exception as e:
            logger.error(f"Error invoking tool {name}: {e}")
            return ToolResult.error_result(
                f"Error invoking tool {name}: {e}", metadata={"tool_name": name}
            )
        return result


def create_default_tool_registry() -> ToolRegistry:
    """创建一个默认的 tool registry"""
    registry = ToolRegistry()
    for tool_class in get_all_builtin_tools():
        registry.register(tool_class())
    return registry
