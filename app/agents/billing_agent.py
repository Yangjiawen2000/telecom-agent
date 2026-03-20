import logging
import re
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.memory.stm import ShortTermMemory
from app.llm import chat
from app.schemas.business import UserInfo, Bill, UsageInfo

logger = logging.getLogger(__name__)

class BillingAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="Billing_Expert", role="电信账务专家", **kwargs)

    async def run(self, user_input: str, session_id: str, user_id: str, stm: ShortTermMemory) -> Dict[str, Any]:
        """
        账务专家逻辑：基于 Function Calling 自主查询账单、用量、办理充值。
        """
        system_prompt = """你是一个专业的电信账务专家。你的目标是帮助用户查询账单、剩余流量/语音，以及办理充值。
你可以使用以下工具：
- get_user_info: 获取用户基本信息
- get_bill: 获取当前最近的一笔账单
- get_usage_info: 获取实时的通用流量、语音使用情况
- get_billing_history: 查询历史月份的账单记录
- get_plans: 获取所有可选套餐的详细资费信息（用于对比或介绍）
- recharge_phone: 为用户充值话费

要求：
1. 必须根据用户的具体问题（如“查流量” vs “查当前套餐”）选择性地调用相关工具。
2. 对于“查信息”、“查套餐”等模糊请求，优先调用 `get_user_info` 获取当前状态。
3. 如果你需要手机号，且用户没有提供，请直接向用户提问。
4. 得到工具结果后，使用专业、亲切的语气综合回答。
        """
        
        # 允许使用的工具池
        allowed_tools = [
            "get_user_info", 
            "get_bill", 
            "get_usage_info", 
            "get_billing_history", 
            "get_plans",
            "recharge_phone"
        ]
        
        result = await self.autonomous_run(
            user_input=user_input,
            system_prompt=system_prompt,
            tool_names=allowed_tools,
            session_id=session_id,
            user_id=user_id,
            stm=stm,
            max_iterations=4
        )
        
        # 兼容外层处理格式
        return {
            "answer": result["content"],
            "sources": result["used_tools"],
            "confidence": result["confidence"]
        }
