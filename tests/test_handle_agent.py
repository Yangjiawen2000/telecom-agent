import pytest
import json
from unittest.mock import AsyncMock, patch
from app.agents.handle_agent import HandleAgent
from app.tools.registry import ToolResult

@pytest.mark.asyncio
async def test_handle_agent_collecting():
    agent = HandleAgent()
    mock_stm = AsyncMock()
    # 模拟空历史，初始状态
    mock_stm.get_history.return_value = []
    
    with patch("app.agents.handle_agent.chat", new_callable=AsyncMock) as mock_chat:
        # LLM 发现缺姓名、身份证和套餐ID，正在追问
        mock_chat.return_value = '{"state": "COLLECTING", "form_data": {"phone": "186123"}, "message": "请问您的姓名和身份证号是多少？", "done": false}'
        
        result = await agent.run("我想办业务", mock_stm)
        assert result["state"] == "COLLECTING"
        assert result["form_data"]["phone"] == "186123"

@pytest.mark.asyncio
async def test_handle_agent_submitting():
    mock_reg = AsyncMock()
    mock_reg.call.return_value = ToolResult(success=True, data={"order_id": "ORD123"})
    
    agent = HandleAgent(tool_registry=mock_reg)
    mock_stm = AsyncMock()
    # 模拟上一步是 CONFIRMING
    mock_stm.get_history.return_value = [
        {"role": "assistant", "metadata": {"handle_state": {"state": "CONFIRMING", "form_data": {"phone": "186", "plan_id": "p1"}}}}
    ]
    
    with patch("app.agents.handle_agent.chat", new_callable=AsyncMock) as mock_chat:
        # 4项信息已收齐，LLM 决定跳转到 SUBMITTING
        full_form = {"name": "张三", "id_card": "110101", "phone": "186", "plan_id": "p1"}
        mock_chat.return_value = json.dumps({
            "state": "SUBMITTING", 
            "form_data": full_form, 
            "message": "正在为您提交订单...", 
            "done": False
        })
        
        result = await agent.run("确定办理", mock_stm)
        assert result["state"] == "DONE"
        assert "ORD123" in result["message"]
        mock_reg.call.assert_called_once()
