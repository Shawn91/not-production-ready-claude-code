import json

from pydantic import BaseModel, Field

from config.loader import get_data_dir
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult


class MemoryParams(BaseModel):
    action: str = Field(
        ..., description="Action: 'set', 'get', 'delete', 'list', 'clear'"
    )
    key: str | None = Field(
        None, description="Memory key (required for `set`, `get`, `delete`)"
    )
    value: str | None = Field(None, description="Value to store (required for `set`)")


class MemoryTool(Tool):
    """memory会使用 json 格式保存到本地文件系统。文件内容类似：
    {
        "entries": {
            "key1": "value1",
            "key2": "value2"
        }
    }

    """

    name = "memory"
    description = "Store and retrieve persistent memory. Use this to remember user preferences, important context or notes."
    kind = ToolKind.MEMORY

    @property
    def schema(self):
        return MemoryParams

    def _load_memory(self) -> dict:
        data_dir = get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "user_memory.json"
        if not path.exists():
            return {"entries": {}}
        try:
            content = path.read_text(encoding="utf-8")
            return json.loads(content)
        except Exception:
            return {"entries": {}}

    def _save_memory(self, memory: dict):
        data_dir = get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "user_memory.json"
        path.write_text(json.dumps(memory, indent=2), encoding="utf-8")

    async def execute(self, invocation: ToolInvocation):
        params = MemoryParams(**invocation.params)
        if params.action == "set":
            if not params.key or not params.value:
                return ToolResult.error_result(
                    "key and value are required for set action"
                )
            memory = self._load_memory()
            memory["entries"][params.key] = params.value
            self._save_memory(memory)
            return ToolResult.success_result(f"memory updated for key '{params.key}'")
        elif params.action == "get":
            if not params.key:
                return ToolResult.error_result("key is required for get action")
            memory = self._load_memory()
            if params.key not in memory.get("entries", {}):
                return ToolResult.error_result(
                    f"key '{params.key}' not found in memory", metadata={"found": False}
                )
            value = memory["entries"][params.key]
            return ToolResult.success_result(
                f"Memory found: {params.key} = {value}", metadata={"found": True}
            )
        elif params.action == "delete":
            if not params.key:
                return ToolResult.error_result("`key` required for 'get' action")
            memory = self._load_memory()
            if params.key not in memory.get("entries", {}):
                return ToolResult.success_result(f"Memory not found: {params.key}")

            del memory["entries"][params.key]
            self._save_memory(memory)
        elif params.action == "list":
            memory = self._load_memory()
            entries = memory.get("entries", {})
            if not entries:
                return ToolResult.success_result(
                    "No memories stored", metadata={"found": False}
                )
            lines = ["Stored memories:"]
            for key, value in sorted(entries.items()):
                lines.append(f"  {key}: {value}")

            return ToolResult.success_result("\n".join(lines), metadata={"found": True})
        elif params.action == "clear":
            memory = self._load_memory()
            count = len(memory.get("entries", {}))
            memory["entries"] = {}
            self._save_memory(memory)
            return ToolResult.success_result(f"Cleared {count} memory entries")
        return ToolResult.error_result("invalid action")
