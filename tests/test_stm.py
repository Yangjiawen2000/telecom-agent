import pytest
import asyncio
import json
from unittest.mock import AsyncMock, patch
from app.memory.stm import ShortTermMemory

class MockRedis:
    def __init__(self):
        self.data = {}
        self.expirations = {}

    async def hset(self, key, field=None, value=None, mapping=None):
        if key not in self.data:
            self.data[key] = {}
        if mapping:
            self.data[key].update(mapping)
        else:
            self.data[key][field] = value

    async def hgetall(self, key):
        return self.data.get(key, {})

    async def delete(self, key):
        if key in self.data:
            del self.data[key]

    async def expire(self, key, ttl):
        self.expirations[key] = ttl

@pytest.mark.asyncio
async def test_stm_history_limit():
    redis = MockRedis()
    stm = ShortTermMemory("sess_1", redis)
    
    # 写入 25 条消息
    for i in range(25):
        await stm.add_message("user", f"msg {i}")
        await asyncio.sleep(0.001) # 保证时间戳 field 唯一且递增

    history = await stm.get_history(max_turns=20)
    assert len(history) == 20
    assert history[0]["content"] == "msg 5"
    assert history[-1]["content"] == "msg 24"

@pytest.mark.asyncio
async def test_stm_distill_anchors():
    redis = MockRedis()
    stm = ShortTermMemory("sess_2", redis)
    
    # 写入一些普通消息和锚点消息
    await stm.add_message("user", "anchor 1", metadata={"is_anchor": True})
    for i in range(15):
        await stm.add_message("user", f"normal {i}")
        await asyncio.sleep(0.001)

    # Mock chat for distillation
    with patch("app.memory.stm.chat", new_callable=AsyncMock) as mock_chat:
        mock_chat.return_value = "Memory summarized"
        await stm.distill(keep_turns=5)

    history = await stm.get_history(max_turns=50)
    
    # 此时历史应该包含：1个摘要，1个保留的锚点，5个最后的普通消息 = 7条
    assert any(m.get("is_summary") for m in history)
    assert any(m["content"] == "anchor 1" for m in history)
    assert len(history) >= 7

@pytest.mark.asyncio
async def test_stm_snapshot_restore():
    redis = MockRedis()
    stm = ShortTermMemory("sess_3", redis)
    
    await stm.add_message("user", "hi")
    await stm.add_message("assistant", "hello", metadata={"is_anchor": True})
    
    snapshot = await stm.snapshot()
    
    # 模拟在另一个 session 或恢复
    new_stm = ShortTermMemory("sess_4", redis)
    await new_stm.restore(snapshot)
    
    restored_history = await new_stm.get_history()
    assert len(restored_history) == 2
    assert restored_history[0]["content"] == "hi"
    assert restored_history[1]["metadata"]["is_anchor"] is True
