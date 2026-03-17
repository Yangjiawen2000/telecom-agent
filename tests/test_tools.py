import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from app.tools.registry import ToolRegistry, ToolResult

@pytest.mark.asyncio
async def test_tool_retry_logic():
    registry = ToolRegistry()
    
    # 一个模拟函数，前两次调用抛出异常，第三次成功
    call_count = 0
    async def mock_tool_func(x):
        nonlocal call_count
        call_count += 1
        if call_count < 3:
            raise Exception("Simulated 500 Error")
        return {"result": x * 2}

    registry.register(
        name="test_tool",
        func=mock_tool_func,
        description="A test tool",
        params_schema={"x": "number"}
    )

    # 减少重试等待时间以便快速测试
    with patch("tenacity.nap.time.sleep", return_value=None):
        result = await registry.call("test_tool", {"x": 10}, max_retries=3)
        
    assert result.success is True
    assert result.data == {"result": 20}
    assert result.retries == 2  # 前两次失败，重试了两次
    assert call_count == 3

@pytest.mark.asyncio
async def test_tool_fallback_logic():
    registry = ToolRegistry()
    
    async def failing_func():
        raise Exception("Permanent Failure")
        
    async def backup_func():
        return {"msg": "recovered via backup"}

    registry.register(
        name="unreliable_tool",
        func=failing_func,
        description="Always fails",
        params_schema={},
        backup_func=backup_func
    )

    with patch("tenacity.nap.time.sleep", return_value=None):
        result = await registry.call("unreliable_tool", {}, max_retries=2)
    
    assert result.success is True
    assert result.data["msg"] == "recovered via backup"
    assert result.fallback == "Backup Function"

@pytest.mark.asyncio
async def test_tool_not_found():
    registry = ToolRegistry()
    result = await registry.call("non_existent", {})
    assert result.success is False
    assert "not found" in result.error
