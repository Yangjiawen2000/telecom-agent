import pytest
import httpx
import json
import asyncio
from httpx import AsyncClient
from app.main import app
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_chat_streaming_flow():
    """测试三轮对话流式输出"""
    # 清理 Redis 环境
    import redis.asyncio as redis
    from app.config import settings
    r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT)
    await r.flushdb()
    await r.aclose()

    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        session_id = "test_session_123"
        user_id = "user_456"
        
        test_cases = [
            "帮我查一下我的套餐信息",
            "那我的账单呢？",
            "顺便帮我办一个新的 5G 尊享 199 套餐"
        ]

        for i, msg in enumerate(test_cases):
            print(f"\n--- Round {i+1}: {msg} ---")
            
            # 开启流式响应
            async with ac.stream("POST", "/chat/message", json={
                "session_id": session_id,
                "user_id": user_id,
                "message": msg
            }) as response:
                assert response.status_code == 200
                
                types_received = []
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        types_received.append(data["type"])
                        if data["type"] == "token":
                            print(data["content"], end="", flush=True)
                        elif data["type"] == "thinking":
                            print(f"[{data['content']}] ", end="", flush=True)

                assert "thinking" in types_received
                assert "token" in types_received
                assert "done" in types_received
                print("\nRound complete.")

@pytest.mark.asyncio
async def test_history_and_delete():
    """测试历史查询与会话删除"""
    from httpx import ASGITransport
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        session_id = "history_test_session"
        user_id = "history_user"
        
        # 1. 先发一条消息产生历史
        async with ac.stream("POST", "/chat/message", json={
            "session_id": session_id,
            "user_id": user_id,
            "message": "你好"
        }) as response:
            async for line in response.aiter_lines():
                pass # 必须消费掉流，直到结束
        
        # 2. 查询历史
        resp = await ac.get(f"/chat/history/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert len(data["history"]) >= 2 # User + Assistant
        
        # 3. 删除/归档会话
        del_resp = await ac.delete(f"/chat/session/{session_id}?user_id={user_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "success"
        
        # 4. 再次查询历史应为空
        empty_resp = await ac.get(f"/chat/history/{session_id}")
        assert len(empty_resp.json()["history"]) == 0
