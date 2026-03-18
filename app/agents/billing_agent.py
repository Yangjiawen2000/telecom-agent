import logging
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.memory.stm import ShortTermMemory
from app.llm import chat

logger = logging.getLogger(__name__)

class BillingAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="Billing_Expert", role="电信账务专家", **kwargs)

    async def run(self, user_input: str, stm: ShortTermMemory) -> Dict[str, Any]:
        """
        账务专家逻辑：
        1. 获取用户账户信息和账单
        2. 分析用户问题并分类
        3. 给出详细解释
        """
        # 1. 尝试寻找手机号
        context_str = await self._get_context("", stm)
        # 简单从上下文提取（真实逻辑应用 IntentClassifier 的 entities）
        phone = "18612345678" # 假定
        
        # 2. 调用账单工具
        bill_res = await self.tool_registry.call(
            "get_bill", 
            {"phone": phone}
        )
        
        # 3. LLM 分析
        system_prompt = f"""你是一个账务专家。基于账单数据回答用户问题。
账单数据：
{bill_res.data if bill_res.success else "无法调取账单"}

输出格式：
{{
  "bill_summary": "总体情况总结",
  "items": ["具体费用项1", "费用项2"],
  "action_needed": "建议执行的操作 (如充值/联系人工)"
}}
请返回纯 JSON。
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        response_text = await chat(messages, stream=False)
        try:
            import json
            clean_text = response_text.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"BillingAgent error: {e}")
            return {"bill_summary": "账单分析失败", "items": [], "action_needed": None}
