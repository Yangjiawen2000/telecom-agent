import pytest
from unittest.mock import AsyncMock, patch
from app.agents.billing_agent import BillingAgent
from app.tools.registry import ToolResult

@pytest.mark.asyncio
async def test_billing_agent_success():
    mock_reg = AsyncMock()
    mock_reg.call.return_value = ToolResult(success=True, data={"total": 100, "details": []})
    
    with patch("app.agents.billing_agent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"bill_summary": "正常", "items": ["基础费 100"], "action_needed": null}'
        
        agent = BillingAgent(tool_registry=mock_reg)
        mock_stm = AsyncMock()
        mock_stm.get_history.return_value = []
        
        result = await agent.run("查账单", mock_stm)
        assert "bill_summary" in result
        assert "基础费 100" in result["items"]
        mock_reg.call.assert_called_with("get_bill", {"phone": "18612345678"})
