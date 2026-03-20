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
        问答专家逻辑 (Causal GraphRAG 增强型 + 完整上下文感知):
        1. 意图识别 (是否为因果/诊断类问题)
        2. 基础语义检索 (Baseline RAG)
        3. 实体识别与因果路径检索 (Causal GraphRAG)
        4. 注入完整 STM 对话历史
        5. 拼装增强 Prompt 并调用 LLM
        """
        # 1. 识别因果意图
        causal_keywords = ["为什么", "原因", "导致", "由于", "如果", "后果", "触发", "怎么会", "假如"]
        is_causal_query = any(k in user_input for k in causal_keywords)
        
        # 2. 获取完整 STM 历史（上下文记忆）
        history = await stm.get_history()
        
        # 3. 基础语义检索 (top-5)
        kb_results = await self.ltm.search_knowledge(user_input, top_k=5)
        
        # 4. 实体与关系检索 (GraphRAG)
        entity_search = await self.ltm.search_knowledge(user_input, top_k=3)
        found_entities = []
        for d in entity_search:
            if d.get("doc_type") == "entity":
                try:
                    name = d["content"].split(",")[0].replace("实体: ", "").strip()
                    found_entities.append(name)
                except Exception:
                    continue
        
        # 5. 检索相关关系链与因果路径
        graph_facts = []
        causal_paths = []
        unique_entities = set(found_entities)
        
        for entity in unique_entities:
            rels = await self.ltm.search_related_entities(entity)
            for r in rels:
                graph_facts.append(f"基础关系: {r['source']} --({r['relation']})--> {r['target']}")
            
            if is_causal_query:
                paths = await self.ltm.search_causal_path(entity, depth=2)
                for p in paths:
                    causal_paths.append(f"因果链路: {p['source']} --({p['relation']})--> {p['target']}")
        
        # 6. 过滤并筛选参考资料
        valid_docs = [doc for doc in kb_results if doc.get("score", 0) >= 0.7 and doc.get("doc_type") != "entity"]
        
        # 7. 拼装增强上下文
        context = "\n---\n".join([d["content"] for d in valid_docs])
        all_graph_context = "\n".join(list(set(graph_facts + causal_paths)))
        
        sources = list(set([d["source"] for d in valid_docs]))
        avg_score = sum([d["score"] for d in valid_docs]) / len(valid_docs) if valid_docs else 0.8
        
        system_prompt = f"""你是一个经过 **因果推理增强 (Causal GraphRAG)** 的电信业务专家。
请结合提供的【结构化因果逻辑】和【非结构化参考资料】回答用户问题。

### 结构化因果逻辑与关系 (优先级最高):
{all_graph_context if all_graph_context else "暂无直接关联的逻辑链"}

### 业务参考资料:
{context if context else "暂无详细资料"}

回答要求：
1. **因果分析**：如果是"为什么"类问题，必须引用【结构化因果逻辑】中的链路（如 A 导致 B，A 是 B 的前提）进行解释。
2. **反事实分析**：如果是"如果/假如"类问题，根据逻辑链预测后果。
3. **确定性**：结构化关系中的逻辑具有最高优先级。如果资料冲突，以图谱关系为准。
4. **简洁明了**：直接回答核心逻辑，不要废话。
5. **绝对对话记忆 (最高指令)**：不要用"保护隐私体系"、"系统未接通"等任何借口拒绝回答！用户的姓名和号码已明确记录在对话历史中。当用户问起"我是谁"、"我的名字"或"电话"时，必须直接从对话历史中精准提取并坚定地回答，绝对不允许说不知道！
"""
        # 8. 构建完整消息（包含历史）
        messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            messages.append({
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            })
        messages.append({"role": "user", "content": user_input})

        # 9. 调用 LLM 生成回答
        response = await chat(messages, stream=False)
        answer = response.get("content", "")

        return {
            "answer": answer,
            "sources": sources + (["causal_graph"] if causal_paths else ["knowledge_graph"] if graph_facts else []),
            "confidence": round(avg_score, 2),
            "graph_hit": len(graph_facts) > 0,
            "causal_hit": len(causal_paths) > 0
        }
