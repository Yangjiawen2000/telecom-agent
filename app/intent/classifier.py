import json
import logging
from enum import Enum
from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from app.llm import chat
from app.memory.ltm import LongTermMemory

logger = logging.getLogger(__name__)

class Intent(str, Enum):
    QUERY_PLAN    = "query_plan"      # 套餐查询
    HANDLE_BIZ    = "handle_biz"      # 业务办理
    RECOMMEND     = "recommend"       # 套餐推荐
    QUERY_BILL    = "query_bill"      # 账单查询
    COMPLAINT     = "complaint"       # 投诉
    GENERAL_QA    = "general_qa"      # 通用问答
    UNKNOWN       = "unknown"

class IntentResult(BaseModel):
    intent: Intent
    confidence: float           # 0~1
    entities: Dict[str, Any] = Field(default_factory=dict)
    sub_intents: List[Intent] = Field(default_factory=list)
    reasoning: str              # LLM 推理过程

class IntentClassifier:
    def __init__(self):
        self.ltm = LongTermMemory()

    def _get_system_prompt(self, context: Optional[str] = None) -> str:
        prompt = f"""你是一个电信业务意图识别专家。你的任务是分析用户的输入，提取其意图和实体。
目前支持的意图包括：
- query_plan: 套餐查询、资费、流量查询
- handle_biz: 新办卡、变更套餐、销户等业务办理
- recommend: 根据需求推荐套餐
- query_bill: 账单、充值、余额查询
- complaint: 投诉、意见反馈、不满、停机原因咨询
- general_qa: 通用电信知识或闲聊
- unknown: 无法识别的意图

输出必须是一个合法的 JSON 对象，包含以下字段：
- intent: 主意图 (枚举值之一)
- confidence: 置信度 (0.0 到 1.0)
- entities: 提取的实体，如 {{"phone": "186xxx", "plan": "畅越129"}}，如果没有则为空字典 {{}}
- sub_intents: 如果用户有多个意图，请列出所有次要意图，如果没有则为空列表 []
- reasoning: 你的推理简述

"""
        if context:
            prompt += f"\n参考业务上下文信息：\n{context}\n"
        
        prompt += "\n仅输出 JSON，不要有其他解释文字。"
        return prompt

    async def classify(self, text: str, history: List[Dict] = None, user_context: str = None) -> IntentResult:
        """多层意图识别：1. LLM 初步分类 2. 低置信度时 RAG 补强"""
        
        # 1. 初始分类
        result = await self._llm_classify(text, history)
        
        # 2. 如果置信度较低，触发 RAG 补强
        if result.confidence < 0.75:
            logger.info(f"Low confidence ({result.confidence}), triggering RAG fallback for text: {text}")
            knowledge_hits = await self.ltm.search_knowledge(text, top_k=3)
            if knowledge_hits:
                context = "\n".join([h["content"] for h in knowledge_hits])
                result = await self._llm_classify(text, history, context)
        
        return result

    async def _llm_classify(self, text: str, history: List[Dict] = None, context: str = None) -> IntentResult:
        system_prompt = self._get_system_prompt(context)
        
        user_msg = f"用户输入：{text}"
        if history:
            user_msg = f"历史对话背景：{history[-3:]}\n" + user_msg
            
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ]
        
        try:
            response_text = await chat(messages, temperature=0.1)
            # 处理可能的 markdown 标签
            clean_json = response_text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            return IntentResult(**data)
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return IntentResult(
                intent=Intent.UNKNOWN,
                confidence=0.0,
                entities={},
                sub_intents=[],
                reasoning=f"Error parsing LLM response: {str(e)}"
            )

    async def extract_entities(self, text: str) -> Dict[str, Any]:
        """专门提取实体"""
        system_prompt = """你是一个实体提取专家。从文本中提取电信相关实体（手机号、套餐名、身份证号、业务类型等）。
输出格式为 JSON，例如: {"phone": "18600001111", "plan": "畅越129"}。
如果没有实体，返回 {}。"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"提取文本中的实体：{text}"}
        ]
        
        try:
            response_text = await chat(messages, temperature=0.1)
            clean_json = response_text.strip().replace("```json", "").replace("```", "").strip()
            return json.loads(clean_json)
        except:
            return {}
