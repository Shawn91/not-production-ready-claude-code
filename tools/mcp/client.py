from enum import Enum
from pathlib import Path

from fastmcp import Client

from config.config import MCPServerConfig


class MCPServerStatus(str, Enum):
    DISCONNECTED = "disconnected"
    CONNECTED = "connected"
    CONNECTING = "connecting"
    ERROR = "error"


class MCPClient:
    """一个 client 连接一个MCP server"""

    def __init__(self, name: str, config: MCPServerConfig, cwd: Path):
        self.name = name
        self.config = config
        self.cwd = cwd
        self.status = MCPServerStatus.DISCONNECTED
        self._client: Client | None = None

    async def connect(self):
        if self.status == MCPServerStatus.CONNECTED:
            return
        self.status = MCPServerStatus.CONNECTING
        # self._client = Client()
