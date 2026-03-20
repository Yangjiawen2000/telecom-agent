from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from app.memory.stm import ShortTermMemory
from app.memory.ltm import LongTermMemory
from app.tools.registry import ToolRegistry, ToolResult
from app.intent.classifier import IntentClassifier
from app.llm import chat
import json
import logging

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
        
        # 初始消息列表
        messages = [
            {"role": "system", "content": f"{system_prompt}\n\n[上下文信息]\n{context}"},
            {"role": "user", "content": user_input}
        ]
        
        iterations = 0
        used_tools = []
        successful_tools: List[str] = []
        tool_failures: List[Dict[str, Any]] = []
        
        while iterations < max_iterations:
            iterations += 1
            logger.info(f"[{self.name}] Autonomous Iteration {iterations}/{max_iterations}")
            
            # 请求大模型
            response = await chat(messages=messages, tools=tools)
            
            # 记录大模型的回复消息
            messages.append(response)
            
            # 检查是否有工具调用
            if "tool_calls" not in response:
                # 已经是最终文本回复
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
                
            # 处理工具调用
            tool_calls = response["tool_calls"]
            for tool_call in tool_calls:
                fn_name = tool_call["function"]["name"]
                try:
                    fn_args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    fn_args = {}
                    
                logger.info(f"[{self.name}] 决定调用工具: {fn_name} 参数: {fn_args}")
                used_tools.append(fn_name)
                
                # 执行工具
                result: ToolResult = await self.tool_registry.call(fn_name, fn_args)
                if result.success:
                    successful_tools.append(fn_name)
                else:
                    tool_failures.append({"name": fn_name, "error": result.error})
                
                # 记录工具执行结果
                tool_msg = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "name": fn_name,
                    "content": json.dumps(result.data if result.success else {"error": result.error}, ensure_ascii=False)
                }
                messages.append(tool_msg)
                
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
