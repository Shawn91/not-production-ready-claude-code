import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

load_dotenv()


class ShellEnvironmentPolicy(BaseModel):
    ignore_default_excludes: bool = False
    # 包含 key, token 这样关键词的环境变量不应该发送给 AI
    exclude_patterns: list[str] = Field(
        default_factory=lambda: ["*KEY*", "*TOKEN*", "*SECRET*"]
    )
    # 针对AI返回shell command这个场景，可以设置一些环境变量，方便/限制AI生成的 commands
    set_vars: dict[str, str] = Field(default_factory=dict)


class MCPServerConfig(BaseModel):
    enabled: bool = True
    startup_timeout_sec: float = 10
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    cwd: Path | None = None

    url: str | None = None

    @model_validator(mode="after")
    def validate_transport(self) -> "MCPServerConfig":
        has_command = self.command is not None
        has_url = self.url is not None
        if not has_command and not has_url:
            raise ValueError("Either command or url must be set")
        if has_command and has_url:
            raise ValueError("Only one of command or url can be set")
        return self


class ModelConfig(BaseModel):
    name: str = "poe/glm-5"
    temperature: float = Field(default=1, ge=0.0, le=2.0)
    context_window: int = 256000


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    cwd: Path = Field(default_factory=Path.cwd)
    shell_environment: ShellEnvironmentPolicy = Field(
        default_factory=ShellEnvironmentPolicy
    )
    max_turns: int = 100  # 一个聊天记录中，最多可以有多少轮对话
    # 仅限 subagent tool 才需要这个属性
    allowed_tools: list[str] | None = Field(
        None, description="If set, only these tools will be available to the agent"
    )
    # key 是 MCP 服务器的名称，value 是 MCP 服务器的配置
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)
    # 这两个 instructions 会写入 agent.md 中，并最终合并到 system prompt 中
    developer_instructions: str | None = None
    user_instructions: str | None = None

    debug: bool = False  # 是否在开发环境

    @property
    def api_key(self) -> str | None:
        return os.environ.get("API_KEY")

    @property
    def base_url(self) -> str | None:
        return os.environ.get("BASE_URL")

    @property
    def model_name(self) -> str | None:
        return self.model.name

    @model_name.setter
    def model_name(self, value: str):
        self.model.name = value

    @property
    def temperature(self) -> float:
        return self.model.temperature

    @temperature.setter
    def temperature(self, value: float):
        self.model.temperature = value

    def validate_config(self) -> list[str]:
        errors: list[str] = []
        if not self.api_key:
            errors.append("API_KEY is not set")
        if not self.cwd.exists():
            errors.append("WORKING_DIRECTORY is not set")
        if not self.base_url:
            errors.append("BASE URL is not set")
        return errors

    def to_dict(self):
        return self.model_dump(mode="json")
