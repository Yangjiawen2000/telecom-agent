import asyncio
import traceback
from unittest.mock import patch, AsyncMock, MagicMock

# mock pymilvus
import sys
sys.modules['pymilvus'] = MagicMock()

from app.agents.billing_agent import BillingAgent
from app.memory.stm import ShortTermMemory
from app.tools.registry import ToolRegistry
from app.tools.init_tools import register_all_tools

async def main():
    try:
        registry = ToolRegistry()
        register_all_tools(registry)
        
        agent = BillingAgent(tool_registry=registry)
        
        mock_redis = AsyncMock()
        mock_redis.hgetall.return_value = {}
        
        stm = ShortTermMemory(session_id="test_session", redis_client=mock_redis)
        await stm.add_message("user", "你好")
        await stm.add_message("assistant", "你好，我是电信专家。")
        
        print("User: 我的手机号是 18612345678, 请帮我查一下上个月的账单。")
        print("-" * 50)
        
        res = await agent.run(
            user_input="我的手机号是 18612345678, 请帮我查一下上个月的账单。",
            session_id="test_session",
            user_id="user_123",
            stm=stm
        )
        
        print("\n[AI Response]")
        print(res["answer"])
        print("\n[Used Tools]")
        print(res["sources"])
    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
