import pytest
from unittest.mock import AsyncMock, patch
from app.agents.qa_agent import QAAgent
from app.memory.stm import ShortTermMemory

@pytest.mark.asyncio
async def test_qa_agent_success():
    # Mock dependencies
    mock_ltm = AsyncMock()
    mock_ltm.search_knowledge.return_value = [
        {"content": "套餐A包含10GB流量", "source": "知识库01", "score": 0.9},
        {"content": "套餐A月费59元", "source": "知识库01", "score": 0.8}
    ]
    
    # Mock LLM chat
    with patch("app.agents.qa_agent.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "套餐A包含10GB流量，月费59元。"
        
        agent = QAAgent(ltm=mock_ltm)
        mock_stm = AsyncMock(spec=ShortTermMemory)
        
        result = await agent.run(
            user_input="套餐A是什么？",
            stm=mock_stm
        )
        
        assert result["confidence"] >= 0.8
        assert "套餐A" in result["answer"]
        assert "知识库01" in result["sources"]
        mock_ltm.search_knowledge.assert_called_once()

@pytest.mark.asyncio
async def test_qa_agent_low_confidence():
    # Mock dependencies
    mock_ltm = AsyncMock()
    mock_ltm.search_knowledge.return_value = [
        {"content": "无关文档", "source": "未知", "score": 0.5}
    ]
    
    agent = QAAgent(ltm=mock_ltm)
    mock_stm = AsyncMock(spec=ShortTermMemory)
    
    result = await agent.run(
        user_input="火星上有电信营业厅吗？",
        stm=mock_stm
    )
    
    assert "暂无相关资料" in result["answer"]
    assert result["confidence"] == 0.0
