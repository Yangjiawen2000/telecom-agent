import asyncio
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock

# 确保能导入 app
sys.path.append(os.getcwd())

from app.agents.orchestrator import Orchestrator
from app.intent.classifier import Intent, IntentResult
from app.memory.stm import ShortTermMemory
from app.tools.registry import ToolRegistry

async def manual_test_concurrency():
    print("🚀 开始并行执行效率测试...\n")
    
    # 1. 模拟依赖
    mock_registry = MagicMock(spec=ToolRegistry)
    mock_stm = MagicMock(spec=ShortTermMemory)
    
    orchestrator = Orchestrator(registry=mock_registry)
    
    # 2. 模拟复合意图：查账单 (Billing) + 问问题 (QA)
    # 按照之前的规则，这两者没有互相依赖，应该并行执行
    mock_intent = IntentResult(
        intent=Intent.QUERY_BILL,
        confidence=0.9,
        entities={},
        sub_intents=[Intent.GENERAL_QA],
        reasoning="用户想同时查账单和咨询问题"
    )
    orchestrator.classifier.classify = AsyncMock(return_value=mock_intent)
    
    # 3. 模拟耗时任务
    # 每个任务模拟耗时 1 秒
    SLEEP_TIME = 1.0
    
    async def slow_run(*args, **kwargs):
        await asyncio.sleep(SLEEP_TIME)
        return {"message": "DONE"}

    orchestrator.billing_agent.run = AsyncMock(side_effect=slow_run)
    orchestrator.qa_agent.run = AsyncMock(side_effect=slow_run)
    
    initial_state = {
        "session_id": "concurrency_test",
        "user_id": "user_001",
        "user_input": "查下账单，顺便问个问题",
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
    
    print(f"⏱️ 启动 Orchestrator... 每个任务耗时 {SLEEP_TIME}s")
    start_time = time.perf_counter()
    
    result = await orchestrator.graph.ainvoke(initial_state)
    
    end_time = time.perf_counter()
    duration = end_time - start_time
    
    print(f"\n✅ 执行完毕！总耗时: {duration:.2f}s")
    
    # 4. 判断结果
    # 如果是串行，耗时 > 2s
    # 如果是并行，耗时应在 1s 左右（加上调度开销，通常 < 1.2s）
    if duration < (SLEEP_TIME * 1.5):
        print(f"🎉 并行测试成功：{duration:.2f}s << {SLEEP_TIME * 2}s (串行预估)")
    else:
        print(f"❌ 并行测试失败：耗时 {duration:.2f}s 过长，疑似串行执行。")

if __name__ == "__main__":
    asyncio.run(manual_test_concurrency())
