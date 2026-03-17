import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.memory.ltm import LongTermMemory

@pytest.mark.asyncio
async def test_ltm_workflow():
    # 注意：这个测试依赖真实的 Milvus 或需要深度的 Mock
    # 这里我们主要测试逻辑流转
    ltm = LongTermMemory()
    
    # 1. 初始化（即使已存在也不报错）
    await ltm.init_collections()
    
    # 2. 测试写入知识
    sample_docs = [{
        "content": "测试知识点：5G套餐资费为199元",
        "embedding": [0.1] * 1024,
        "source": "test.md",
        "doc_type": "test"
    }]
    await ltm.upsert_knowledge(sample_docs)
    
    # 3. 测试检索 (Mock embedding 保证一致性)
    with patch("app.memory.ltm.embed", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.1] * 1024
        results = await ltm.search_knowledge("199元套餐")
        assert len(results) > 0
        assert "199元" in results[0]["content"]

@pytest.mark.asyncio
async def test_user_profile():
    ltm = LongTermMemory()
    user_id = "user123"
    summary = "该用户偏好大流量套餐，曾咨询过239档。"
    
    with patch("app.memory.ltm.embed", new_callable=AsyncMock) as mock_embed:
        mock_embed.return_value = [0.2] * 1024
        await ltm.update_user_profile(user_id, summary)
        
    context = await ltm.get_user_context(user_id)
    assert context == summary
