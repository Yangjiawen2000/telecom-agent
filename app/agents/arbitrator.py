import logging
import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from app.llm import embed, chat
from app.memory.stm import ShortTermMemory

logger = logging.getLogger(__name__)

class ConflictReport(BaseModel):
    has_conflict: bool
    conflict_type: Optional[str] = None
    parties: List[str] = []      # 冲突的专家名称
    description: Optional[str] = None

class ArbitrateResult(BaseModel):
    resolved: bool
    winner: str             # 采用哪个专家的结论
    reason: str
    escalate: bool          # 是否需要人工介入

class ConflictArbitrator:
    def __init__(self):
        self.escalation_count = 0

    async def detect(self, results: List[Dict[str, Any]], stm: ShortTermMemory) -> ConflictReport:
        """
        检测多个专家结果是否存在冲突
        """
        if len(results) < 2:
            return ConflictReport(has_conflict=False)

        # 1. STATUS 冲突检测：检查专家返回的元数据是否与 STM 冲突
        # 假设专家结果中有 "user_status" 字段
        stm_data = await stm.get_history() # 简化处理
        # 实际实现中可能需要更复杂的 STM 状态获取
        
        # 2. SEMANTIC 冲突检测
        for i in range(len(results)):
            for j in range(i + 1, len(results)):
                text1 = self._get_text_content(results[i]["output"])
                text2 = self._get_text_content(results[j]["output"])
                
                if not text1 or not text2:
                    continue
                
                similarity = await self._get_similarity(text1, text2)
                
                # 如果相似度低 (< 0.4) 且逻辑矛盾 (由 LLM 进一步判断)
                if similarity < 0.4:
                    is_contradictory = await self._check_logical_contradiction(text1, text2)
                    if is_contradictory:
                        return ConflictReport(
                            has_conflict=True,
                            conflict_type="SEMANTIC",
                            parties=[results[i]["task_id"], results[j]["task_id"]],
                            description=f"语义低相似度 ({similarity:.2f}) 且逻辑矛盾:\n1: {text1}\n2: {text2}"
                        )

        return ConflictReport(has_conflict=False)

    async def arbitrate(self, conflict: ConflictReport, context: str, results: List[Dict[str, Any]]) -> ArbitrateResult:
        """
        根据优先级和 LLM 裁判进行仲裁
        1. 业务系统实时数据 (Tool Result) > 长期记忆 > 当前 LLM 推理
        """
        # 优先级映射
        # 假设我们能从 task_id 或结果中区分来源
        # 简单规则：handle_agent 和 billing_agent 通常包含 Tool Result
        
        party_results = [r for r in results if r["task_id"] in conflict.parties]
        
        # 优先级规则 1: 检查是否有业务系统数据 (通过 handle/billing 返回且包含具体数据)
        tool_winner = None
        for r in party_results:
            # 简单判断是否来自工具：如果有 specific_data 或来自 handle/billing
            if any(k in r["task_id"] for k in ["handle", "billing"]):
                tool_winner = r
                break
        
        if tool_winner:
            return ArbitrateResult(
                resolved=True,
                winner=tool_winner["task_id"],
                reason="依据优先级规则：业务系统实时数据 (Tool Result) 优先",
                escalate=False
            )

        # 优先级规则无法消解，提交给 LLM 裁判
        prompt = f"""你是一名专业的电信业务仲裁员。当前多个专家给出的结论存在冲突。
上下文: {context}
冲突描述: {conflict.description}

请根据上下文，分析哪份结论更具置信度。
输出格式必须为 JSON: {{"winner": "task_id", "reason": "...", "confidence": 0.0}}
"""
        try:
            res_str = await chat([{"role": "user", "content": prompt}])
            # 简单解析
            import json
            import re
            match = re.search(r"\{.*\}", res_str, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return ArbitrateResult(
                    resolved=True,
                    winner=data["winner"],
                    reason=f"LLM 仲裁结论: {data['reason']}",
                    escalate=False
                )
        except Exception as e:
            logger.error(f"Arbitration LLM error: {e}")

        self.escalation_count += 1
        if self.escalation_count >= 3:
            return ArbitrateResult(resolved=False, winner="", reason="连续 3 次无法消解冲突", escalate=True)
            
        return ArbitrateResult(resolved=False, winner="", reason="仲裁逻辑异常", escalate=False)

    def _get_text_content(self, output: Any) -> str:
        if isinstance(output, dict):
            return output.get("message", output.get("answer", ""))
        return str(output)

    async def _get_similarity(self, t1: str, t2: str) -> float:
        v1 = await embed(t1)
        v2 = await embed(t2)
        
        # 计算余弦相似度
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm1 = math.sqrt(sum(a * a for a in v1))
        norm2 = math.sqrt(sum(a * a for a in v2))
        
        if norm1 * norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    async def _check_logical_contradiction(self, t1: str, t2: str) -> bool:
        prompt = f"请判断以下两条关于电信业务的结论是否逻辑矛盾（例如一个说可行，一个说不可行）：\n1. {t1}\n2. {t2}\n只需回答 '是' 或 '否'。"
        res = await chat([{"role": "user", "content": prompt}])
        return "是" in res
