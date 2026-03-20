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

        # 3. 调用 LLM 驱动状态转换并自主调用工具
        system_prompt = f"""你是一个专业的电信业务办理专家。
当前办理状态：{current_state}
已收集数据：{form_data}

【任务目标】
你的主要任务是收集办理宽带/办卡所需的 4 个核心信息，并在收集完毕后，调用 `create_order` 工具完成办理，然后告知用户。

【执行规则】
1. 必须收集以下 4 个信息：
   - 姓名 (name)
   - 身份证号 (id_card)
   - 手机号 (phone)
   - 套餐ID (plan_id)
2. 如果上述信息有缺失，你必须向用户礼貌追问缺失的信息。
3. 当 4 项信息全部收集完毕，并且你向用户确认无误后，你必须调用 `create_order` 工具提交订单。
4. 提交订单成功后，告知用户办理成功及订单号。
5. 每次回复时，请在内部维护好状态，并输出专业、亲切的文字。
"""
        
        allowed_tools = ["create_order", "get_plans"]
        
        result = await self.autonomous_run(
            user_input=user_input,
            system_prompt=system_prompt,
            tool_names=allowed_tools,
            session_id=session_id,
            user_id=user_id,
            stm=stm,
            max_iterations=4
        )
        
        # 只有 create_order 工具成功时，才允许进入 DONE
        create_order_attempted = "create_order" in result.get("used_tools", [])
        create_order_success = "create_order" in result.get("successful_tools", [])
        is_done = create_order_success
        new_state = "DONE" if is_done else "COLLECTING"
        answer = result["content"]
        # 防护：LLM 即使“说成功了”，也要以工具调用成功为准
        if create_order_attempted and not create_order_success:
            answer = f"抱歉，订单提交失败（下单接口调用未成功）。\n{answer}"
        
        return {
            "answer": answer,
            "handle_state": {
                "state": new_state,
                "form_data": form_data # 这里为了简化，假设LLM在文本中收集，实际可由LLM输出结构化数据
            },
            "sources": result["used_tools"],
            "confidence": result["confidence"]
        }

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
            response = await chat([{"role": "user", "content": prompt}], stream=False)
            content = response.get("content", "")
            clean_res = content.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_res)
        except:
            return {}
