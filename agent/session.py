import json
import uuid
from datetime import datetime
from typing import Any

from client.llm_client import LLMClient
from config.config import Config
from config.loader import get_data_dir
from context.compaction import ChatCompactor
from context.loop_detector import LoopDetector
from context.manager import ContextManager
from hooks.hook_system import HookSystem
from safety.approval import ApprovalManager
from tools.registry import create_default_tool_registry


class Session:
    def __init__(self, config: Config) -> None:
        self.client = LLMClient(config=config)
        self.tool_registry = create_default_tool_registry(config=config)
        self.context_manager = ContextManager(
            config=config,
            user_memory=self._load_memory(),
            tools=self.tool_registry.get_tools(),
        )
        self.config = config
        self.session_id = str(uuid.uuid4())
        self.chat_compactor = ChatCompactor(client=self.client)
        self.approval_manager = ApprovalManager(self.config.approval, self.config.cwd)
        self.loop_detector = LoopDetector()
        self.hook_system = HookSystem(config=config)
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self._turn_count = 0

    def increment_turn(self) -> int:
        self._turn_count += 1
        self.updated_at = datetime.now()
        return self._turn_count

    def _load_memory(self) -> str | None:
        data_dir = get_data_dir()
        data_dir.mkdir(parents=True, exist_ok=True)
        path = data_dir / "user_memory.json"
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8")
            entries = json.loads(content).get("entries", {})
            if not entries:
                return None
            lines = ["User Preferences:"]
            for key, value in entries.items():
                lines.append(f"- {key}: {value}")
            return "\n".join(lines)
        except Exception:
            return None

    def get_stats(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "turn_count": self._turn_count,
            "message_count": self.context_manager.message_count,
            "token_usage": self.context_manager.total_usage,
            "tools_count": len(self.tool_registry.get_tools()),
            # "mcp_servers": len(self.tool_registry.connected_mcp_servers),
        }
