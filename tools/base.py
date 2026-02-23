import abc
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# pydantic用于定义 tool 的 params，以及做参数的类型验证
from pydantic import BaseModel, ValidationError


class ToolKind(str, Enum):
    READ = "read"
    WRITE = "write"
    SHELL = "shell"
    NETWORK = "network"
    MEMORY = "memory"
    MCP = "mcp"


@dataclass
class ToolInvocation:
    """定义一个 tool 的所有参数，以及当前工作目录"""

    cwd: Path  # current working directory
    params: dict[str, Any]


@dataclass
class ToolResult:
    """定义一个 tool 的执行结果"""

    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] | None = field(default_factory=dict)
    truncated: bool = False

    @classmethod
    def error_result(cls, error: str, output: str = "", **kwargs: Any):
        return cls(success=False, output=output, error=error, **kwargs)

    @classmethod
    def success_result(cls, output: str, **kwargs: Any):
        return cls(success=True, output=output, error=None, **kwargs)

    def to_model_output(self) -> str:
        """根据工具执行结果决定输出内容"""
        if self.success:
            return self.output
        return f"Error: {self.error}\nOutput: {self.output}"


@dataclass
class ToolConfirmation:
    """当一个 tool 涉及到修改外部状态（如写入文件）时，需要用户确认。
    本 dataclass 用于存储向用户寻求确认时，需要展示的信息"""

    tool_name: str
    params: dict[str, Any]
    description: str


class Tool(abc.ABC):
    name: str = "base_tool"
    description: str = "Base tool"
    kind: ToolKind = ToolKind.READ

    def __init__(self):
        pass

    @property
    def schema(self) -> dict[str, Any] | type[BaseModel]:
        """定义tool的参数的类型
        完全自定义的 tool 应该返回 BaseModel。如果是调用了第三方mcp，则可能返回 dict
        """
        raise NotImplementedError("Tool must define schema property or class attribute")

    @abc.abstractmethod
    async def execute(self, invocation: ToolInvocation) -> ToolResult: ...

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """返回值中的每个字符串对应一个 invalid param 的报错信息，空 list 表示所有参数都有效"""
        schema = self.schema
        # 只有当 schema 是 BaseModel 的子类时才进行手动验证。schema是 dict 时，让 llm client 自行处理即可
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            try:
                schema(**params)
            except ValidationError as e:
                errors = []
                for error in e.errors():
                    field = ".".join(str(x) for x in error.get("loc", []))
                    msg = error.get("msg", "Validation error")
                    errors.append(f"Parameter '{field}' {msg}")
            except Exception as e:
                return [str(e)]
        return []

    def is_mutating(self, params: dict[str, Any]) -> bool:
        """返回 True 表示这个 tool 会修改外部状态，例如修改文件等。纯读取文件不会修改外部状态。
        这里的判断只是做了一个基本的判断，表示这个类型的 tool 有可能修改外部状态。
        特定的 tool 需要 override 本方法
        """
        return self.kind in {
            ToolKind.WRITE,
            ToolKind.SHELL,
            ToolKind.NETWORK,
            ToolKind.MEMORY,
        }

    async def get_confirmation(
        self, invocation: ToolInvocation
    ) -> ToolConfirmation | None:
        """对于可能修改外部状态的 tool，需要用户确认。
        返回 None 表示不需要用户确认
        """
        if not self.is_mutating(invocation.params):
            return None
        return ToolConfirmation(
            tool_name=self.name,
            params=invocation.params,
            description=self.description,
        )

    def to_openai_schema(self) -> dict[str, Any]:
        schema = self.schema
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = schema.model_json_schema(mode="serialization")
            return {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": json_schema.get("properties", {}),
                    "required": json_schema.get("required", []),
                },
            }
        # 此时说明调用的是某个mcp作为tool
        elif isinstance(schema, dict):
            result: dict[str, Any] = {
                "name": self.name,
                "description": self.description,
            }
            if "parameters" in schema:
                result["parameters"] = schema["parameters"]
            else:
                result["parameters"] = schema
            return result
        else:
            raise ValueError(
                f"Unsupported schema type for tool {self.name}: {type(schema)}"
            )
