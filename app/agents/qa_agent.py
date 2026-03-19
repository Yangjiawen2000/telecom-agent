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

    async def run(self, user_input: str, session_id: str, user_id: str, stm: ShortTermMemory) -> Dict[str, Any]:
        """
        问答专家逻辑 (GraphRAG 增强型):
        1. 从 LTM 检索基础知识片段 (Baseline RAG)
        2. 识别实体并检索关联关系 (GraphRAG)
        3. 拼装增强 Prompt 并调用 LLM
        """
        # 1. 基础语义检索 (top-5)
        kb_results = await self.ltm.search_knowledge(user_input, top_k=5)
        
        # 2. 实体与关系检索 (GraphRAG)
        # 检索与输入语义最接近的实体节点
        entity_search = await self.ltm.search_knowledge(user_input, top_k=3)
        found_entities = []
        for d in entity_search:
            if d.get("doc_type") == "entity":
                # 解析出实体名称 (格式为 "实体: NAME, 类型: TYPE")
                try:
                    name = d["content"].split(",")[0].replace("实体: ", "").strip()
                    found_entities.append(name)
                except Exception:
                    continue
        
        # 检索相关关系链
        graph_facts = []
        for entity in set(found_entities):
            rels = await self.ltm.search_related_entities(entity)
            for r in rels:
                graph_facts.append(f"关系: {r['source']} --({r['relation']})--> {r['target']}")
        
        # 3. 过滤并筛选参考资料
        valid_docs = [doc for doc in kb_results if doc.get("score", 0) >= 0.7 and doc.get("doc_type") != "entity"]
        
        if not valid_docs and not graph_facts:
            return {
                "answer": "暂无相关资料，建议咨询人工客服。",
                "sources": [],
                "confidence": 0.0
            }

        # 4. 拼装增强上下文
        context = "\n---\n".join([d["content"] for d in valid_docs])
        graph_context = "\n".join(graph_facts)
        sources = list(set([d["source"] for d in valid_docs]))
        avg_score = sum([d["score"] for d in valid_docs]) / len(valid_docs) if valid_docs else 0.8
        
        system_prompt = f"""你是一个经过 GraphRAG 增强的电信业务专家。
请结合提供的【结构化知识图谱关系】和【非结构化参考资料】回答用户问题。

### 结构化知识图谱关系 (更精确的实体依赖):
{graph_context if graph_context else "暂无关联关系"}

### 详细参考资料:
{context if context else "暂无详细资料"}

注意：
1. 优先遵循知识图谱中的逻辑（如要求、冲突、属于）。
2. 如果资料中没有提到相关信息，请直接回答不知情，不要捏造。
3. 结构化关系中的内容具有最高优先级。
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # 5. 调用 LLM 生成回答
        answer = await chat(messages, stream=False)

        return {
            "answer": answer,
            "sources": sources + (["knowledge_graph"] if graph_facts else []),
            "confidence": round(avg_score, 2),
            "graph_hit": len(graph_facts) > 0
        }
