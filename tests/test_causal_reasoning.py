import asyncio
import logging
from app.agents.qa_agent import QAAgent
from app.memory.ltm import LongTermMemory
from app.memory.stm import ShortTermMemory

async def test_causal_diagnostic():
    logging.basicConfig(level=logging.INFO)
    
    agent = QAAgent()
    # Mock or dummy stm since it's not actually used for the core logic we're testing
    class DummyRedis:
        async def hset(self, *args, **kwargs): pass
        async def hgetall(self, *args, **kwargs): return {}
        async def expire(self, *args, **kwargs): pass
        async def delete(self, *args, **kwargs): pass

    stm = ShortTermMemory(session_id="test_session", redis_client=DummyRedis())
    session_id = "test_causal_session"
    user_id = "test_user_123"

    test_queries = [
        "为什么我的号会被停机？",
        "如果我没有完成实名审核，会发生什么后果？",
        "由于超量使用导致的费用增加，应该如何避免？"
    ]

    print("\n" + "="*50)
    print("CAUSAL REASONING TEST")
    print("="*50)

    for query in test_queries:
        print(f"\n[QUERY]: {query}")
        result = await agent.run(query, session_id, user_id, stm)
        print(f"[ANSWER]: {result['answer']}")
        print(f"[CAUSAL HIT]: {result.get('causal_hit', False)}")
        print(f"[SOURCES]: {result['sources']}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(test_causal_diagnostic())
