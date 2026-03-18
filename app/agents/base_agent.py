from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.memory.stm import ShortTermMemory
from app.memory.ltm import LongTermMemory
from app.tools.registry import ToolRegistry, ToolResult
from app.intent.classifier import IntentClassifier

class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        role: str,
        ltm: Optional[LongTermMemory] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ):
        self.name = name
        self.role = role
        self.ltm = ltm or LongTermMemory()
        self.tool_registry = tool_registry or ToolRegistry()

    @abstractmethod
    async def run(self, user_input: str, session_id: str, user_id: str, stm: ShortTermMemory) -> Dict[str, Any]:
        """核心执行逻辑，由各专家子类实现"""
        pass

    async def _get_context(self, user_id: str, stm: ShortTermMemory) -> str:
        """整合 STM 和 LTM 获取完整上下文"""
        history = await stm.get_history()
        user_profile = await self.ltm.get_user_context(user_id)
        
        context = f"用户长期画像：\n{user_profile}\n\n当前会话历史：\n{history}"
        return context
