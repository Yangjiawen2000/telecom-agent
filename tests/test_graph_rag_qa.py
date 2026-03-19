import asyncio
import logging
from unittest.mock import MagicMock
from app.agents.qa_agent import QAAgent
from app.memory.stm import ShortTermMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def test_q_a_with_graph():
    # Initialize the agent
    agent = QAAgent()
    
    # Mock Redis client for ShortTermMemory
    mock_redis = MagicMock()
    mock_redis.hgetall.return_value = asyncio.Future()
    mock_redis.hgetall.return_value.set_result({}) # Initial empty history
    
    # Initialize STM with correct signature
    stm = ShortTermMemory(session_id="test_session", redis_client=mock_redis)
    
    # Test Question 1: Complex query involving entity relationships
    # The graph should help identify that "畅越 129 套餐" includes "IPTV" (天翼高清)
    query = "办办 畅越 129 套餐需要什么材料？赠送哪些业务？"
    print(f"\n[QUERY] User: {query}")
    
    try:
        result = await agent.run(
            user_input=query,
            session_id="test_session",
            user_id="test_user",
            stm=stm
        )
        
        print("\n[RESPONSE]")
        print(f"Assistant: {result['answer']}")
        print(f"Graph Hit: {result.get('graph_hit')}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {result['sources']}")
    except Exception as e:
        logger.error(f"Error during test execution: {e}", exc_info=True)

    # Test Question 2: Query involving conditions or hierarchies
    query = "如何查询手机号是否欠费停机？如果已停机怎么复机？"
    print(f"\n[QUERY] User: {query}")
    
    try:
        result = await agent.run(
            user_input=query,
            session_id="test_session",
            user_id="test_user",
            stm=stm
        )
        
        print("\n[RESPONSE]")
        print(f"Assistant: {result['answer']}")
        print(f"Graph Hit: {result.get('graph_hit')}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {result['sources']}")
    except Exception as e:
        logger.error(f"Error during test execution: {e}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_q_a_with_graph())
