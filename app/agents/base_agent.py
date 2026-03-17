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
        stm: Optional[ShortTermMemory] = None,
        ltm: Optional[LongTermMemory] = None,
        tool_registry: Optional[ToolRegistry] = None,
        intent_classifier: Optional[IntentClassifier] = None
    ):
        self.name = name
        self.role = role
        self.stm = stm or ShortTermMemory()
        self.ltm = ltm or LongTermMemory()
        self.tool_registry = tool_registry or ToolRegistry()
        self.intent_classifier = intent_classifier or IntentClassifier()

    @abstractmethod
    async def think(self, user_input: str, session_id: str) -> str:
        """核心思考逻辑，需由子类实现"""
        pass

    async def _call_tool(self, tool_name: str, params: Dict[str, Any]) -> ToolResult:
        """统一工具调用入口"""
        return await self.tool_registry.call(tool_name, params)

    async def _get_context(self, user_id: str, session_id: str) -> str:
        """整合 STM 和 LTM 获取完整上下文"""
        stm_context = await self.stm.get_history(session_id)
        ltm_context = await self.ltm.get_user_context(user_id)
        
        context = f"用户长期画像：\n{ltm_context}\n\n当前会话历史：\n{stm_context}"
        return context
