import json
import logging
from typing import Dict, Any, List
from app.agents.base_agent import BaseAgent
from app.memory.stm import ShortTermMemory
from app.llm import chat

logger = logging.getLogger(__name__)

class QAAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(name="QA_Expert", role="电信业务知识问答专家", **kwargs)

    async def run(self, user_input: str, stm: ShortTermMemory) -> Dict[str, Any]:
        """
        问答专家逻辑：
        1. 从 LTM 检索知识片段
        2. 过滤低相关度结果
        3. 拼装 Prompt 调用 LLM
        4. 返回结构化结果
        """
        # 1. 语义检索 (top-5)
        kb_results = await self.ltm.search_knowledge(user_input, top_k=5)
        
        # 2. 过滤相关度 < 0.7 的结果
        valid_docs = [doc for doc in kb_results if doc.get("score", 0) >= 0.7]
        
        if not valid_docs:
            return {
                "answer": "暂无相关资料，建议咨询人工客服。",
                "sources": [],
                "confidence": 0.0
            }

        # 3. 拼装上下文
        context = "\n---\n".join([d["content"] for d in valid_docs])
        sources = list(set([d["source"] for d in valid_docs]))
        avg_score = sum([d["score"] for d in valid_docs]) / len(valid_docs)

        system_prompt = f"""你是一个专业的电信业务知识专家。请基于以下参考资料回答用户问题。
如果资料中没有提到相关信息，请直接回答不知情，不要捏造。

参考资料：
{context}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # 4. 调用 LLM 生成回答
        answer = await chat(messages, stream=False)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": round(avg_score, 2)
        }
