import json
import logging
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.memory.stm import ShortTermMemory
from app.llm import chat
from app.tools.clients import get_plans, get_user_info

logger = logging.getLogger(__name__)

class RecommendAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="Recommend_Expert", role="套餐个性化推荐专家", **kwargs)

    async def run(self, user_input: str, stm: ShortTermMemory) -> Dict[str, Any]:
        """
        推荐专家逻辑：
        1. 获取所有可用套餐
        2. 获取用户当前状态及手机号（从上下文或输入提取）
        3. 获取用户长期习惯 (LTM)
        4. LLM 分析并返回 Top-3 推荐
        """
        # 1. 获取工具数据
        plans = await self.tool_registry.call(
            "get_plans", 
            {}
        )
        
        # 尝试从输入或历史提取手机号
        history = await stm.get_history()
        # 简单的手机号辅助提取（项目中已有更强的 IntentClassifier，这里假定已获得或从背景获取）
        # 为了演示，我们假设 context_str 中有足够的线索
        # 为了演示，我们假设 user_id 在这里是可用的（通过 self.name 获取）
        # 实际上 BaseAgent 已经持有 ltm
        context_str = await self._get_context("", stm)
        
        # 2. 构造个性化 Prompt
        system_prompt = f"""你是一个专业的电信套餐推荐专家。
基于用户的上下文和所有可用套餐，为用户推荐 3 个最合适的套餐，并给出推荐理由。

用户上下文：
{context_str}

可用套餐列表：
{plans.data if plans.success else "无法获取套餐列表"}

输出要求：
必须返回纯 JSON 格式，不要包含任何 Markdown 代码块标签，格式如下：
{{
  "plans": [
    {{"id": "...", "name": "...", "price": 0, "reason": "..."}}
  ],
  "primary": "推荐主套餐的 ID"
}}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"用户需求：{user_input}"}
        ]

        # 3. 调用 LLM
        response_text = await chat(messages, stream=False)
        
        # 4. 尝试解析 JSON
        try:
            # 去除可能存在的 markdown 标签
            clean_text = response_text.strip().replace("```json", "").replace("```", "")
            result = json.loads(clean_text)
            return result
        except Exception as e:
            logger.error(f"Failed to parse RecommendAgent JSON: {e}, Response: {response_text}")
            return {
                "plans": [],
                "primary": "",
                "error": "推荐引擎结果解析失败"
            }
