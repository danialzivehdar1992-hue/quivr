from quivr_core.rag.entities.config import QuivrBaseConfig
from typing import Optional, List, Dict, Any


class ToolConfig(QuivrBaseConfig):
    name: str
    config: Optional[Dict[str, Any]] = None


class ToolsConfig(QuivrBaseConfig):
    max_tool_calls: int = 10
    tools: List[ToolConfig] = []

    def get_tool_config_by_name(self, name: str) -> Optional[ToolConfig]:
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None
