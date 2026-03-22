from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.memory.stm import ShortTermMemory
from app.memory.ltm import LongTermMemory
from app.tools.registry import ToolRegistry, ToolResult
from app.intent.classifier import IntentClassifier
from app.llm import chat
import json
import logging
import asyncio
from app.config import settings

logger = logging.getLogger(__name__)

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

    async def think(self, user_input: str, session_id: str, user_id: str = "default_user", stm: Optional[ShortTermMemory] = None) -> Dict[str, Any]:
        """别名方法，适配测试脚本"""
        if stm is None:
            from app.config import get_redis_client
            redis_client = get_redis_client()
            stm = ShortTermMemory(session_id, redis_client)
        return await self.run(user_input, session_id, user_id, stm)

    async def _get_context(self, user_id: str, stm: ShortTermMemory) -> str:
        """整合 STM 和 LTM 获取完整上下文"""
        history = await stm.get_history()
        user_profile = await self.ltm.get_user_context(user_id)
        
        context = f"用户长期画像：\n{user_profile}\n\n当前会话历史：\n{history}"
        return context

    async def autonomous_run(self, 
                             user_input: str, 
                             system_prompt: str, 
                             tool_names: List[str], 
                             session_id: str, 
                             user_id: str, 
                             stm: ShortTermMemory,
                             max_iterations: int = 5) -> Dict[str, Any]:
        """
        基于 Function Calling 的自主循环 (ReAct Loop)。
        1. 组装对话历史与可用工具。
        2. 调用 LLM，若 LLM 返回 tool_calls，则执行对应工具。
        3. 将工具执行结果作为 tool 角色消息追加到历史中，再次请求 LLM。
        4. 重复直至 LLM 返回最终纯文本响应。
        """
        context = await self._get_context(user_id, stm)
        tools = self.tool_registry.get_openai_tools(tool_names)
        
        # 初始系统提示
        base_system_msg = {"role": "system", "content": f"{system_prompt}\n\n[上下文信息]\n{context}"}
        
        # 维护最近的观察结果（Role: tool）作为推理依据
        recent_observations: List[Dict[str, Any]] = []
        
        iterations = 0
        used_tools = []
        successful_tools: List[str] = []
        tool_failures: List[Dict[str, Any]] = []
        
        while iterations < max_iterations:
            iterations += 1
            logger.info(f"[{self.name}] Autonomous Iteration {iterations}/{max_iterations}")
            
            # 模式 1：Think 阶段 - 仅传输 [System, User, *RecentObservations]
            think_messages = [
                base_system_msg,
                {"role": "user", "content": user_input},
                *recent_observations[-2:] # 只保留最近2步的工具反馈，极大缩减上下文
            ]
            
            # 使用轻量级模型进行思考
            response = await chat(
                messages=think_messages, 
                tools=tools,
                model=settings.INTENT_MODEL,
                max_tokens=300
            )
            
            # 检查是否有工具调用
            if "tool_calls" not in response:
                content = response.get("content", "")
                await stm.add_message("user", user_input)
                await stm.add_message("assistant", content)
                return {
                    "role": self.role,
                    "content": content,
                    "used_tools": used_tools,
                    "successful_tools": successful_tools,
                    "tool_failures": tool_failures,
                    "confidence": 0.95
                }
                
            # 处理并行工具调用
            tool_calls = response["tool_calls"]
            
            async def execute_tool_call(tc):
                fn_name = tc["function"]["name"]
                try:
                    fn_args = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                
                logger.debug(f"[{self.name}] 并查调用工具: {fn_name}")
                result: ToolResult = await self.tool_registry.call(fn_name, fn_args)
                
                # 截断工具结果至 300 字符
                observation_text = str(result.data if result.success else {"error": result.error})
                if len(observation_text) > 300:
                    observation_text = observation_text[:300] + "...(truncated)"
                
                return {
                    "name": fn_name,
                    "success": result.success,
                    "error": result.error if not result.success else None,
                    "tool_call_id": tc["id"],
                    "content": observation_text
                }

            # 并行执行单次 iteration 中的工具
            executed_results = await asyncio.gather(*(execute_tool_call(tc) for tc in tool_calls))
            
            for res in executed_results:
                used_tools.append(res["name"])
                if res["success"]:
                    successful_tools.append(res["name"])
                else:
                    tool_failures.append({"name": res["name"], "error": res["error"]})
                
                recent_observations.append({
                    "role": "tool",
                    "tool_call_id": res["tool_call_id"],
                    "name": res["name"],
                    "content": res["content"]
                })
                
        # 超过最大迭代次数时的兜底
        fallback_msg = "我已经收集了足够的信息，但无法得出最终结论，请稍后再试或提供更多细节。"
        await stm.add_message("user", user_input)
        await stm.add_message("assistant", fallback_msg)
        return {
            "role": self.role,
            "content": fallback_msg,
            "used_tools": used_tools,
            "successful_tools": successful_tools,
            "tool_failures": tool_failures,
            "confidence": 0.3
        }
