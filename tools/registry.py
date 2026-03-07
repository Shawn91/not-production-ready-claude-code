import logging
from pathlib import Path
from typing import Any

from config.config import Config
from hooks.hook_system import HookSystem
from safety.approval import ApprovalContext, ApprovalDecision, ApprovalManager
from tools import get_all_builtin_tools
from tools.base import Tool, ToolInvocation, ToolResult
from tools.subagent import SubagentTool, get_default_subagent_definitions

logger = logging.getLogger(__name__)


class ToolRegistry:
    def __init__(self, config: Config):
        self._tools: dict[str, Tool] = {}
        self.config = config

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
        tools = list(self._tools.values())
        if self.config.allowed_tools:
            tools = [t for t in tools if t.name in self.config.allowed_tools]
        return tools

    def get_schemas(self) -> list[dict[str, Any]]:
        """返回所有 tools 的 openai schema"""
        return [t.to_openai_schema() for t in self.get_tools()]

    async def invoke(
        self,
        name: str,
        params: dict[str, Any],
        cwd: Path,
        approval_manager: ApprovalManager,
        hook_system: HookSystem,
    ) -> ToolResult:
        """调用一个 tool"""
        tool = self.get(name)
        if tool is None:
            result = ToolResult.error_result(
                f"Unknown tool: {name}",
                metadata={"tool_name": name},
            )
            await hook_system.trigger_after_tool(name, params, result)
            return result

        validation_errors = tool.validate_params(params)
        if validation_errors:
            result = ToolResult.error_result(
                f"Invalid parameters: {'; '.join(validation_errors)}",
                metadata={"tool_name": name, "validation_errors": validation_errors},
            )

            await hook_system.trigger_after_tool(name, params, result)

            return result

        await hook_system.trigger_before_tool(name, params)
        invocation = ToolInvocation(
            params=params,
            cwd=cwd,
        )
        if approval_manager:
            confirmation = await tool.get_confirmation(invocation)
            if confirmation:
                context = ApprovalContext(
                    tool_name=name,
                    params=params,
                    is_mutating=tool.is_mutating(params),
                    affected_paths=confirmation.affected_paths,
                    command=confirmation.command,
                    is_dangerous=confirmation.is_dangerous,
                )

                decision = await approval_manager.check_approval(context)
                if decision == ApprovalDecision.REJECTED:
                    result = ToolResult.error_result(
                        "Operation rejected by safety policy"
                    )
                    await hook_system.trigger_after_tool(name, params, result)
                    return result
                elif decision == ApprovalDecision.NEEDS_CONFIRMATION:
                    approved = approval_manager.request_confirmation(confirmation)

                    if not approved:
                        result = ToolResult.error_result("User rejected the operation")
                        await hook_system.trigger_after_tool(name, params, result)
                        return result

        try:
            result = await tool.execute(invocation)
        except Exception as e:
            logger.exception(f"Tool {name} raised unexpected error")
            result = ToolResult.error_result(
                f"Internal error: {str(e)}", metadata={"tool_name", name}
            )

        await hook_system.trigger_after_tool(name, params, result)
        return result


def create_default_tool_registry(config: Config) -> ToolRegistry:
    """创建一个默认的 tool registry"""
    registry = ToolRegistry(config=config)
    for tool_class in get_all_builtin_tools():
        registry.register(tool_class(config=config))
    for subagent_def in get_default_subagent_definitions():
        registry.register(SubagentTool(config=config, definition=subagent_def))
    return registry
