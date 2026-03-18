import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

# 确保能导入 app
sys.path.append(os.getcwd())

from app.agents.orchestrator import Orchestrator
from app.intent.classifier import Intent, IntentResult
from app.memory.stm import ShortTermMemory
from app.tools.registry import ToolRegistry

async def manual_test_switching():
    print("🚀 开始场景切换与上下文恢复测试...\n")
    
    # 1. 模拟依赖
    mock_registry = MagicMock(spec=ToolRegistry)
    mock_stm = MagicMock(spec=ShortTermMemory)
    
    orchestrator = Orchestrator(registry=mock_registry)
    
    # 2. 模拟意图解析：用户想要“办业务”
    mock_intent = IntentResult(
        intent=Intent.HANDLE_BIZ,
        confidence=0.98,
        entities={"action": "办理套餐"},
        sub_intents=[],
        reasoning="用户明确要求办理业务"
    )
    orchestrator.classifier.classify = AsyncMock(return_value=mock_intent)
    
    # 3. 模拟 HandleAgent：第一次调用时由于“欠费”触发跳转
    # 第二次调用（恢复）时完成办理
    handle_calls = {"count": 0}
    async def mock_handle_run(user_input, session_id, user_id, stm):
        handle_calls["count"] += 1
        if handle_calls["count"] == 1:
            print("🤖 [HandleAgent] 发现用户欠费，触发转向 QA 解释原因...")
            return {
                "need_switch": "qa_agent", 
                "reason": "解释为何欠费无法办理",
                "message": "由于您账户欠费，暂时无法办理。正在为您查询原因..."
            }
        else:
            print("🤖 [HandleAgent] 收到恢复信号，继续办理完成！")
            return {
                "status": "DONE",
                "message": "办理成功！您的新套餐已生效。"
            }
    
    orchestrator.handle_agent.run = AsyncMock(side_effect=mock_handle_run)
    
    # 4. 模拟 QAAgent：解释原因
    async def mock_qa_run(user_input, session_id, user_id, stm):
        print("🤖 [QAAgent] 正在回答用户关于欠费的问题...")
        return {
            "answer": "您上月话费超支 10 元，充值后即可继续办理。",
            "sources": ["billing_system"],
            "confidence": 1.0
        }
    
    orchestrator.qa_agent.run = AsyncMock(side_effect=mock_qa_run)

    # 5. 执行图
    initial_state = {
        "session_id": "switching_test_001",
        "user_id": "user_001",
        "user_input": "我想办个 5G 套餐",
        "intent_result": None,
        "task_dag": [],
        "current_node": "",
        "context_snapshots": {},
        "final_response": "",
        "fsm_state": "IDLE",
        "expert_outputs": [],
        "stm": mock_stm,
        "registry": mock_registry
    }
    
    print("🎬 运行 Orchestrator...")
    result = await orchestrator.graph.ainvoke(initial_state)
    
    print("\n--- 最终输出 ---")
    print(result["final_response"])
    print("----------------\n")
    
    # 6. 验证
    # 顺序应该是：Handle(1) -> QA -> Handle(2)
    if handle_calls["count"] == 2:
        print("✅ 测试通过：办理流程已成功恢复并完成。")
    else:
        print(f"❌ 测试失败：HandleAgent 调用次数为 {handle_calls['count']}，预期为 2。")

if __name__ == "__main__":
    asyncio.run(manual_test_switching())
