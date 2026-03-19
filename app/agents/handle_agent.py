import json
import logging
import re
from typing import Dict, Any, List, Optional
from app.agents.base_agent import BaseAgent
from app.memory.stm import ShortTermMemory
from app.llm import chat

logger = logging.getLogger(__name__)

class HandleAgent(BaseAgent):
    """
    业务办理专家：
    维护一个简单的状态机：INIT -> COLLECTING -> CONFIRMING -> SUBMITTING -> DONE
    """
    def __init__(self, **kwargs):
        super().__init__(name="Handle_Expert", role="电信业务办理专家", **kwargs)

    async def run(self, user_input: str, session_id: str, user_id: str, stm: ShortTermMemory) -> Dict[str, Any]:
        # 1. 获取当前状态（优先从结构化元数据中读取）
        history = await stm.get_history()
        current_state = "INIT"
        form_data = {
            "name": None,
            "id_card": None,
            "phone": None,
            "plan_id": None
        }
        
        # 尝试从元数据恢复
        for msg in reversed(history):
            if msg.get("role") == "assistant" and "metadata" in msg:
                state_info = msg["metadata"].get("handle_state")
                if state_info:
                    current_state = state_info.get("state", "INIT")
                    form_data.update(state_info.get("form_data", {}))
                    break
        
        # 2. 鲁棒性增强：如果核心字段缺失，尝试从最近几轮的【对话文字】中通过 LLM 补全提取
        if not all(form_data.values()):
            extracted = await self._extract_info_from_text(user_input, history)
            for k, v in extracted.items():
                if v and not form_data.get(k):  # 仅补全缺失项
                    form_data[k] = v
                    if current_state == "INIT":
                        current_state = "COLLECTING"

        # 3. 调用 LLM 驱动状态转换
        system_prompt = f"""你是一个专业的电信业务办理专家。
当前办理状态：{current_state}
已收集数据：{form_data}

【严格的状态机规则】
1. INIT: 用户表达办卡或宽带办理意图时，必须立即跳转到 COLLECTING。
2. COLLECTING: 你必须收集以下 4 个信息：
   - 姓名 (name)
   - 身份证号 (id_card)
   - 手机号 (phone)
   - 套餐ID (plan_id)
   如果 {form_data} 中缺少上述任何一项，你必须在 message 中礼貌追问缺失项。只有当 4 项全部收集完毕时，才能跳转到 CONFIRMING。
3. CONFIRMING: 4 项信息收齐后，必须在 message 中生成全量确认话术。
4. SUBMITTING: 用户确认后，系统将自动处理。
5. DONE: 办理完成。

【输出格式强制要求】
请返回纯 JSON 格式：
{{
  "state": "COLLECTING | CONFIRMING | SUBMITTING | DONE",
  "form_data": {{ "name": "...", "id_card": "...", "phone": "...", "plan_id": "..." }},
  "message": "回复给用户的文字",
  "done": false
}}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        response_text = await chat(messages, stream=False)
        try:
            clean_text = response_text.strip().replace("```json", "").replace("```", "")
            result = json.loads(clean_text)
            
            # 特殊处理提交逻辑
            if result.get("state") == "SUBMITTING":
                params = result.get("form_data", {})
                order_res = await self.tool_registry.call("create_order", params)
                if order_res.success:
                    result["state"] = "DONE"
                    result["done"] = True
                    order_id = order_res.data.get('order_id') if isinstance(order_res.data, dict) else "N/A"
                    result["message"] = f"办理成功！订单号：{order_id}"
                else:
                    return {
                        "need_switch": "qa_agent",
                        "reason": f"下单失败：{order_res.error}，正在为您转接支持。"
                    }
            
            # 标准化输出，供 Orchestrator 聚合并持久化
            return {
                "answer": result.get("message", ""),
                "handle_state": {
                    "state": result.get("state"),
                    "form_data": result.get("form_data")
                },
                "confidence": 0.9
            }
        except Exception as e:
            logger.error(f"HandleAgent error: {e}")
            return {"answer": "办理逻辑异常，请稍后重试。", "confidence": 0.0}

    async def _extract_info_from_text(self, user_input: str, history: List[Dict]) -> Dict[str, str]:
        """通过 LLM 从对话历史文字中提取潜在的办理信息，作为元数据丢失时的兜底方案"""
        context_slice = history[-6:]  # 取最近3轮完整对话
        prompt = f"""请从以下电信业务对话中提取办理信息。
对话内容：
{context_slice}
当前用户输入：{user_input}

请提取以下字段，如果没提到则设为 null：
- name (姓名)
- id_card (身份证号)
- phone (手机号)
- plan_id (套餐ID，通常是数字或特定名称)

请直接返回 JSON：
{{"name": null, "id_card": null, "phone": null, "plan_id": null}}
"""
        try:
            res = await chat([{"role": "user", "content": prompt}], stream=False)
            clean_res = res.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_res)
        except:
            return {}
