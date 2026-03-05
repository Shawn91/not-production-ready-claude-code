import asyncio
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field

from config.config import Config
from tools.base import Tool, ToolInvocation, ToolResult


class SubagentParams(BaseModel):
    # 一个特定的具体任务。可以理解为类似 todo list 中的一个 item
    goal: str = Field(..., description="The specific goal of the subaget to accomplish")


@dataclass
class SubagentDefinition:
    """一个 definition 对应一个 subagent"""

    name: str
    description: str
    # goal_prompt 不是用户的特定需求prompt。而是事先写好的，针对某个特定的 subagent 的 prompt
    # 用于指定这个 subagent 的总体任务和目标。相当于 system prompt
    goal_prompt: str
    allowed_tools: list[str] | None = None
    max_turns: int = 20
    timeout_seconds: float = 600


class SubagentTool(Tool):
    """每一个 subagent 可以视为一个 tool"""

    def __init__(self, config: Config, definition: SubagentDefinition):
        super().__init__(config)
        self.definition = definition

    @property
    def name(self) -> str:
        return f"subagent_{self.definition.name}"

    @property
    def description(self) -> str:
        return f"subagent_{self.definition.description}"

    @property
    def schema(self) -> type[BaseModel]:
        return SubagentParams

    def is_mutating(self, params: dict[str, Any]) -> bool:
        """大部分情况下Subagent 会修改外部状态，例如修改文件等。因此默认设置为 True"""
        return True

    def _build_prompt(self, params: SubagentParams) -> str:
        return f"""You are a specialized sub-agent with a specific task to complete.

                {self.definition.goal_prompt}

                YOUR TASK:
                {params.goal}

                IMPORTANT:
                - Focus only on completing the specified task
                - Do not engage in unrelated actions
                - Once you have completed the task or have the answer, provide your final response
                - Be concise and direct in your output
                """

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        from agent.agent import Agent
        from agent.events import AgentEventType

        params = SubagentParams(**invocation.params)
        if not params.goal:
            return ToolResult.error_result("goal is required")

        config_dict = self.config.to_dict()
        config_dict["max_turns"] = self.definition.max_turns
        if self.definition.allowed_tools:
            config_dict["allowed_tools"] = self.definition.allowed_tools
        subagent_config = Config(**config_dict)

        prompt = self._build_prompt(params)
        tool_calls: list[str] = []  # 记录下所有被调用的工具名字就行
        final_response = None
        error = None
        terminate_reason = "goal"  # subagent停止运行的原因。默认原因是目标已经实现
        try:
            async with Agent(subagent_config) as agent:
                deadline = (
                    asyncio.get_event_loop().time() + self.definition.timeout_seconds
                )
                async for event in agent.run(message=prompt):
                    if asyncio.get_event_loop().time() > deadline:
                        terminate_reason = "timeout"
                        final_response = (
                            f"Subagent timeout: {self.definition.timeout_seconds}s"
                        )
                        break
                    if event.type == AgentEventType.TOOL_CALL_START:
                        tool_calls.append(event.data["name"])
                    elif event.type == AgentEventType.TEXT_COMPLETE:
                        final_response = event.data.get("content")
                    elif event.type == AgentEventType.AGENT_END:
                        if final_response is None:
                            final_response = event.data.get("response")
                    elif event.type == AgentEventType.AGENT_ERROR:
                        terminate_reason = "error"
                        error = event.data.get("error", "Unknown")
                        final_response = f"Subagent error: {error}"
                        break
        except Exception as e:
            terminate_reason = "error"
            error = str(e)
            final_response = f"Subagent error: {error}"

        result = f"""Subagent `{self.definition.name}` completed.
        Terminated reason: {terminate_reason}
        Tools called: {", ".join(tool_calls) if tool_calls else "None"}

        Result:
            {final_response or "No response"}
        """

        if error:
            return ToolResult.error_result(error=result)
        else:
            return ToolResult.success_result(output=result)


CODEBASE_INVESTIGATOR = SubagentDefinition(
    name="codebase_investigator",
    description="Investigates the codebase to answer questions about code structure, patterns, and implementations",
    goal_prompt="""You are a codebase investigation specialist.
Your job is to explore and understand code to answer questions.
Use read_file, grep, glob, and list_dir to investigate.
Do NOT modify any files.""",
    allowed_tools=["read_file", "grep", "glob", "list_dir"],
)

CODE_REVIEWER = SubagentDefinition(
    name="code_reviewer",
    description="Reviews code changes and provides feedback on quality, bugs, and improvements",
    goal_prompt="""You are a code review specialist.
Your job is to review code and provide constructive feedback.
Look for bugs, code smells, security issues, and improvement opportunities.
Use read_file, list_dir and grep to examine the code.
Do NOT modify any files.""",
    allowed_tools=["read_file", "grep", "list_dir"],
    max_turns=10,
    timeout_seconds=300,
)


def get_default_subagent_definitions() -> list[SubagentDefinition]:
    return [CODEBASE_INVESTIGATOR, CODE_REVIEWER]
