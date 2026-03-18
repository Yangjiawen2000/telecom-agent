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

async def manual_test_orchestrator_arbitration():
    print("🚀 开始 Orchestrator 冲突仲裁集成测试...\n")
    
    # 1. 模拟依赖
    mock_registry = MagicMock(spec=ToolRegistry)
    mock_stm = MagicMock(spec=ShortTermMemory)
    
    orchestrator = Orchestrator(registry=mock_registry)
    
    # 2. 模拟复合意图：办业务 (Handle) + 问问题 (QA)
    mock_intent = IntentResult(
        intent=Intent.HANDLE_BIZ,
        confidence=0.9,
        entities={},
        sub_intents=[Intent.GENERAL_QA],
        reasoning="用户想办业务并问咨询"
    )
    orchestrator.classifier.classify = AsyncMock(return_value=mock_intent)
    
    # 3. 构造冲突输出
    # QA 专家说可以办
    orchestrator.qa_agent.run = AsyncMock(return_value={
        "answer": "我们的 5G 套餐目前没有任何办理限制，您可以随时申请。"
    })
    
    # Handle 专家查了系统说不行（因为停机）
    orchestrator.handle_agent.run = AsyncMock(return_value={
        "message": "系统校验失败：您的号码目前处于停机状态，无法办理新业务。",
        "status": "ERROR"
    })
    
    # 4. Mock 仲裁器的检测逻辑
    # 强制触发冲突检测
    orchestrator.arbitrator._get_similarity = AsyncMock(return_value=0.2)
    orchestrator.arbitrator._check_logical_contradiction = AsyncMock(return_value=True)

    initial_state = {
        "session_id": "arb_test_001",
        "user_id": "user_001",
        "user_input": "我想办 5G 套餐",
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
    
    # 验证是否包含仲裁提示
    if "检测到信息冲突" in result["final_response"] and "停机状态" in result["final_response"]:
        print("✅ 测试通过：Orchestrator 成功检测并处理了专家冲突，且采用了 HandleAgent 的结论。")
    else:
        print("❌ 测试失败：未检测到仲裁痕迹或结论不匹配。")

if __name__ == "__main__":
    asyncio.run(manual_test_orchestrator_arbitration())
