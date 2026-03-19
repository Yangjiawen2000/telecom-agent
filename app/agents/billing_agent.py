import logging
import re
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.memory.stm import ShortTermMemory
from app.llm import chat

logger = logging.getLogger(__name__)

class BillingAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="Billing_Expert", role="电信账务专家", **kwargs)

    async def run(self, user_input: str, session_id: str, user_id: str, stm: ShortTermMemory) -> Dict[str, Any]:
        """
        账务专家逻辑：
        1. 从上下文提取手机号
        2. 调用账单工具
        3. LLM 分析并返回账单摘要
        """
        # 1. 从历史记录提取手机号
        history = await stm.get_history()
        phone = self._extract_phone(user_input, history)
        
        # 2. 调用账单工具
        bill_res = await self.tool_registry.call("get_bill", {"phone": phone})
        user_res = await self.tool_registry.call("get_user_info", {"phone": phone})
        
        user_info = user_res.data if user_res.success else {}
        bill_data = bill_res.data if bill_res.success else {}
        
        # 3. LLM 分析
        system_prompt = f"""你是一个专业的电信账务专家。基于账单数据和用户信息，清晰地回答用户的账单相关问题。

用户信息：
{user_info}

账单数据：
{bill_data}

要求：
- 用自然语言解释账单情况
- 如有欠费或异常，突出说明并给出建议
- 语气亲切、专业
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response_text = await chat(messages, stream=False)
        return {
            "answer": response_text,
            "sources": ["billing_tool"],
            "confidence": 0.9 if bill_res.success else 0.5
        }

    def _extract_phone(self, user_input: str, history: List[Dict]) -> str:
        """从当前输入或对话历史中提取手机号"""
        phone_pattern = re.compile(r'1[3-9]\d{9}')
        
        # 先从用户当前输入找
        match = phone_pattern.search(user_input)
        if match:
            return match.group()
        
        # 再从历史记录找（优先取用户消息）
        for msg in reversed(history):
            match = phone_pattern.search(msg.get("content", ""))
            if match:
                return match.group()
        
        return "18612345678"  # 测试默认值
