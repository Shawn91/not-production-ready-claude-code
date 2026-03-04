import uuid

from pydantic import BaseModel, Field

from config.config import Config
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult


class TodosParams(BaseModel):
    """表示一项 todo item"""

    # add action 表示添加任务，complete action表示完成一项任务。list 表示返回todo中的所有人物.
    # clear表示清空
    action: str = Field(..., description="Action: 'add', 'complete', 'list, 'clear'")
    id: str | None = Field(None, description="Todo ID (for complete)")
    content: str | None = Field(None, description="Todo content (for add)")


class TodosTool(Tool):
    """每个 session 只能有一个未完成的 todo list"""

    name = "todos"
    description = "Manage a task list for the current session. Use this to track progress on multi-step tasks."
    kind = ToolKind.MEMORY

    def __init__(self, config: Config):
        super().__init__(config)
        # key 是 id，value 是一个 todo 的 content
        self._todos: dict[str, str] = {}

    @property
    def schema(self):
        return TodosParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = TodosParams(**invocation.params)

        if params.action.lower() == "add":
            if not params.content:
                return ToolResult.error_result("`content` required for 'add' action")
            todo_id = str(uuid.uuid4())[
                :8
            ]  # 一般而言，不需要那么完整的uuid作为id，取前8个字符就够了
            self._todos[todo_id] = params.content
            return ToolResult.success_result(
                f"Added todo [{todo_id}]: {params.content}"
            )
        elif params.action.lower() == "complete":
            if not params.id:
                return ToolResult.error_result("`id` required for 'complete' action")
            if params.id not in self._todos:
                return ToolResult.error_result(f"Todo item not found for {params.id}")
            content = self._todos.pop(params.id)
            return ToolResult.success_result(f"Completed todo [{params.id}: {content}]")
        elif params.action.lower() == "list":
            if not self._todos:
                return ToolResult.success_result("No todos left")
            lines = ["Todos:"]
            for todo_id, content in self._todos.items():
                lines.append(f"  [{todo_id}]: {content}")
            return ToolResult.success_result("\n".join(lines))
        elif params.action.lower() == "clear":
            count = len(self._todos)
            self._todos.clear()
            return ToolResult.success_result(f"cleared {count} todos")
        return ToolResult.error_result(f"Unknown action: {params.action}")
