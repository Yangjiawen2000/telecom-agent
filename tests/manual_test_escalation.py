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

async def manual_test_orchestrator_escalation():
    print("🚀 开始 Orchestrator 冲突仲裁【人工介入】测试...\n")
    
    # 1. 模拟依赖
    mock_registry = MagicMock(spec=ToolRegistry)
    mock_stm = MagicMock(spec=ShortTermMemory)
    
    orchestrator = Orchestrator(registry=mock_registry)
    
    # 2. 模拟复合意图
    mock_intent = IntentResult(
        intent=Intent.GENERAL_QA,
        confidence=0.9,
        entities={},
        sub_intents=[Intent.GENERAL_QA],
        reasoning="两个 QA 专家产生冲突"
    )
    orchestrator.classifier.classify = AsyncMock(return_value=mock_intent)
    
    # 3. 构造无法通过规则消解的冲突（两个同级别的 QA）
    orchestrator.qa_agent.run = AsyncMock(side_effect=[
        {"answer": "结论 A (QA 专家 1)"},
        {"answer": "结论 B (QA 专家 2)"}
    ])
    
    # 4. 强制触发冲突检测
    orchestrator.arbitrator._get_similarity = AsyncMock(return_value=0.1)
    orchestrator.arbitrator._check_logical_contradiction = AsyncMock(return_value=True)
    
    # 5. 模拟 LLM 仲裁失败（模拟连续失败触发熔断）
    # 我们直接修改 arbitrator 的内部状态来模拟第三次尝试
    orchestrator.arbitrator.escalation_count = 2 
    # 模拟下一次调用 arbitrate 时 LLM 报错或无法解析
    from app.agents.arbitrator import chat
    import app.agents.arbitrator
    app.agents.arbitrator.chat = AsyncMock(side_effect=Exception("LLM 仲裁服务暂时不可用"))

    initial_state = {
        "session_id": "escalate_test_001",
        "user_id": "user_001",
        "user_input": "深度业务冲突测试",
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
    
    print("🎬 运行 Orchestrator (第 3 次尝试)...")
    result = await orchestrator.graph.ainvoke(initial_state)
    
    print("\n--- 最终输出 ---")
    print(result["final_response"])
    print("----------------\n")
    
    # 验证是否包含人工介入提示
    if "转接人工客服" in result["final_response"]:
        print("✅ 测试通过：Orchestrator 在多次仲裁失败后正确输出了人工介入提示。")
    else:
        print("❌ 测试失败：未检测到人工介入提示。")

if __name__ == "__main__":
    asyncio.run(manual_test_orchestrator_escalation())
