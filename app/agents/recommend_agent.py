import json
import logging
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.memory.stm import ShortTermMemory
from app.llm import chat

logger = logging.getLogger(__name__)

class RecommendAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="Recommend_Expert", role="套餐个性化推荐专家", **kwargs)

    async def run(self, user_input: str, session_id: str, user_id: str, stm: ShortTermMemory) -> Dict[str, Any]:
        """
        推荐专家逻辑：
        1. 获取所有可用套餐
        2. 获取用户上下文
        3. LLM 分析并返回 Top-3 推荐
        """
        # 1. 获取工具数据
        plans = await self.tool_registry.call("get_plans", {})
        context_str = await self._get_context(user_id, stm)
        
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
  "primary": "推荐主套餐的 ID",
  "answer": "对用户的推荐说明文字（自然语言）"
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
            clean_text = response_text.strip().replace("```json", "").replace("```", "")
            result = json.loads(clean_text)
            result.setdefault("sources", ["recommend_tool"])
            result.setdefault("confidence", 0.85)
            # 确保有 answer 字段
            if "answer" not in result:
                plan_names = ", ".join([p.get("name", "") for p in result.get("plans", [])])
                result["answer"] = f"为您推荐以下套餐：{plan_names}"
            return result
        except Exception as e:
            logger.error(f"Failed to parse RecommendAgent JSON: {e}, Response: {response_text}")
            return {
                "plans": [],
                "primary": "",
                "answer": "推荐引擎暂时无法处理，请稍后重试。",
                "error": "推荐引擎结果解析失败",
                "sources": [],
                "confidence": 0.0
            }
