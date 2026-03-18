import pytest
from unittest.mock import AsyncMock, patch
from app.agents.recommend_agent import RecommendAgent
from app.tools.registry import ToolResult

@pytest.mark.asyncio
async def test_recommend_agent_success():
    mock_reg = AsyncMock()
    mock_reg.call.return_value = ToolResult(
        success=True, 
        data=[{"id": "p1", "name": "Plan 1", "price": 50}]
    )
    
    with patch("app.agents.recommend_agent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = '{"plans": [{"id": "p1", "name": "Plan 1", "price": 50, "reason": "better price"}], "primary": "p1"}'
        
        agent = RecommendAgent(tool_registry=mock_reg)
        mock_stm = AsyncMock()
        mock_stm.get_history.return_value = []
        
        result = await agent.run("我想省钱", mock_stm)
        
        assert "plans" in result
        assert result["primary"] == "p1"
        mock_reg.call.assert_called_with("get_plans", {})
