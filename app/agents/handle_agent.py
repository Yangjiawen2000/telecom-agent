import json
import logging
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

    async def run(self, user_input: str, stm: ShortTermMemory) -> Dict[str, Any]:
        # 1. 获取当前状态
        history = await stm.get_history()
        # 简单策略：从历史最后一条助手消息中提取状态
        current_state = "INIT"
        form_data = {}
        
        for msg in reversed(history):
            if msg.get("role") == "assistant" and "metadata" in msg:
                state_info = msg["metadata"].get("handle_state")
                if state_info:
                    current_state = state_info.get("state", "INIT")
                    form_data = state_info.get("form_data", {})
                    break
        
        # 2. 调用 LLM 驱动状态转换
        system_prompt = f"""你是一个专业的电信业务办理专家。
当前办理状态：{current_state}
已收集数据：{form_data}

【严格的状态机规则】
1. INIT: 用户表达办卡或办理业务意图时，必须立即跳转到 COLLECTING。
2. COLLECTING: 你必须收集且仅收集以下 4 个信息：
   - 姓名 (name)
   - 身份证号 (id_card)
   - 手机号 (phone)
   - 套餐ID (plan_id)
   如果 {form_data} 中缺少上述任何一项，你必须在 message 中礼貌追问缺失的一项或多项。只有当 4 项全部收集完毕时，才能跳转到 CONFIRMING 状态。
3. CONFIRMING: 当且仅当 4 项信息收齐后进入此状态。
   - 必须在 message 中生成确认话术，例如：“请核对您的办理信息。确认无误后请回复‘确认’。”
   - 必须将收集到的 4 项信息填入 card 字段中用于前端展示。
   - 如果用户回复“确认/对/没问题”，则将状态跳转为 SUBMITTING。如果用户要求修改，退回 COLLECTING。
4. SUBMITTING: （此状态由系统底层处理，如果你输出此状态，只需在 message 中回复“正在为您提交订单...”）
5. DONE: 办理完成。

【输出格式强制要求】
请返回纯 JSON 格式，不要包含任何 markdown 标记，字段定义如下：
{{
  "state": "COLLECTING | CONFIRMING | SUBMITTING | DONE",
  "form_data": {{ "name": "...", "id_card": "...", "phone": "...", "plan_id": "..." }},
  "card": {{ "title": "订单信息核对", "content": "姓名:xxx, 身份证:xxx, 手机号:xxx, 套餐:xxx" }}, // 仅在 CONFIRMING 状态时生成有内容的卡片，其余状态可为空或 null
  "message": "回复给用户的文字",
  "done": false // 只有在 DONE 状态时为 true
}}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]

        # 3. 执行状态流转
        response_text = await chat(messages, stream=False)
        try:
            clean_text = response_text.strip().replace("```json", "").replace("```", "")
            result = json.loads(clean_text)
            
            # 4. 特殊处理 SUBMITTING
            if result.get("state") == "SUBMITTING":
                # 执行真实下单工具
                params = result.get("form_data", {})
                order_res = await self.tool_registry.call(
                    "create_order", 
                    params
                )
                
                if order_res.success:
                    result["state"] = "DONE"
                    result["done"] = True
                    # order_res.data 是 Any，这里假设它是 Dict 以获取 order_id
                    order_id = order_res.data.get('order_id') if isinstance(order_res.data, dict) else "N/A"
                    result["message"] = f"办理成功！订单号：{order_id}"
                else:
                    return {
                        "need_switch": "qa_agent",
                        "reason": f"下单失败：{order_res.error}，正在为您转接问答支持。"
                    }
                    
            return result
        except Exception as e:
            logger.error(f"HandleAgent error: {e}")
            return {"state": "ERROR", "message": "办理逻辑异常", "done": False}
