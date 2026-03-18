import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from app.agents.orchestrator import Orchestrator, OrchestratorState
from app.intent.classifier import IntentResult, Intent
from app.agents.arbitrator import ConflictReport
from app.tools.registry import ToolRegistry

@pytest.fixture
def mock_registry():
    return MagicMock(spec=ToolRegistry)

@pytest.fixture
def mock_stm():
    m = AsyncMock()
    m.get_history.return_value = []
    return m

@pytest.mark.asyncio
async def test_orchestrator_single_intent(mock_registry, mock_stm):
    orchestrator = Orchestrator(registry=mock_registry)
    
    # Mock IntentClassifier
    mock_intent = IntentResult(
        intent=Intent.QUERY_PLAN,
        confidence=0.9,
        entities={},
        sub_intents=[],
        reasoning="test"
    )
    orchestrator.classifier.classify = AsyncMock(return_value=mock_intent)
    
    # Mock Arbitrator
    orchestrator.arbitrator.detect = AsyncMock(return_value=ConflictReport(has_conflict=False))
    
    # Mock QA Agent
    orchestrator.qa_agent.run = AsyncMock(return_value={"message": "这是套餐详情"})
    
    initial_state = {
        "session_id": "s1",
        "user_id": "u1",
        "user_input": "查套餐",
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
    
    result = await orchestrator.graph.ainvoke(initial_state)
    
    assert "这是套餐详情" in result["final_response"]
    assert len(result["task_dag"]) == 1
    orchestrator.qa_agent.run.assert_called_once()

@pytest.mark.asyncio
async def test_orchestrator_composite_intent(mock_registry, mock_stm):
    orchestrator = Orchestrator(registry=mock_registry)
    
    # Mock IntentClassifier: Handle + Billing
    mock_intent = IntentResult(
        intent=Intent.HANDLE_BIZ,
        confidence=0.9,
        entities={"phone": "186"},
        sub_intents=[Intent.QUERY_BILL],
        reasoning="test composite"
    )
    orchestrator.classifier.classify = AsyncMock(return_value=mock_intent)
    
    # Mock Arbitrator
    orchestrator.arbitrator.detect = AsyncMock(return_value=ConflictReport(has_conflict=False))
    
    # Mock Agents
    orchestrator.handle_agent.run = AsyncMock(return_value={"state": "COLLECTING", "message": "开始办理"})
    orchestrator.billing_agent.run = AsyncMock(return_value={"message": "您的账单为100元"})
    
    initial_state = {
        "session_id": "s2",
        "user_id": "u2",
        "user_input": "我想办卡并查下话费",
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
    
    result = await orchestrator.graph.ainvoke(initial_state)
    
    assert "开始办理" in result["final_response"]
    assert "您的账单为100元" in result["final_response"]
    assert len(result["task_dag"]) == 2
    orchestrator.handle_agent.run.assert_called_once()
    orchestrator.billing_agent.run.assert_called_once()

@pytest.mark.asyncio
async def test_orchestrator_switching(mock_registry, mock_stm):
    orchestrator = Orchestrator(registry=mock_registry)
    
    # Mock IntentClassifier: Handle
    mock_intent = IntentResult(
        intent=Intent.HANDLE_BIZ,
        confidence=0.9,
        entities={},
        sub_intents=[],
        reasoning="test switch"
    )
    orchestrator.classifier.classify = AsyncMock(return_value=mock_intent)
    
    # Mock Arbitrator
    orchestrator.arbitrator.detect = AsyncMock(return_value=ConflictReport(has_conflict=False))
    
    # 第一步：Handle Agent 返回 need_switch
    # 注意：ainvoke 是一次性跑完。要测试中间过程，我们需要让第一次 run 返回 need_switch，
    # 但 StateGraph 会继续根据边走。
    # 模拟 Handle Agent 第一次被调用时要求切换
    calls = {"count": 0}
    async def side_effect(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return {"need_switch": "qa_agent", "reason": "咨询停机原因"}
        return {"state": "DONE", "message": "继续办理完成"}

    orchestrator.handle_agent.run = AsyncMock(side_effect=side_effect)
    orchestrator.qa_agent.run = AsyncMock(return_value={"message": "这是停机解释"})
    
    initial_state = {
        "session_id": "s3",
        "user_id": "u3",
        "user_input": "办业务",
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
    
    result = await orchestrator.graph.ainvoke(initial_state)
    
    # 预期：任务队列里会有 [qa_agent, handle_agent]
    # Aggregate 会汇总结果
    assert "这是停机解释" in result["final_response"]
    assert "继续办理完成" in result["final_response"]
    # 验证顺序：先调用了 1 次 handle (switch), 然后 qa, 最后 handle (resume)
    assert calls["count"] == 2
    orchestrator.qa_agent.run.assert_called_once()

@pytest.mark.asyncio
async def test_orchestrator_dag_rules(mock_registry, mock_stm):
    orchestrator = Orchestrator(registry=mock_registry)
    
    # Mock IntentClassifier: RECOMMEND + HANDLE_BIZ + QUERY_BILL + GENERAL_QA
    mock_intent = IntentResult(
        intent=Intent.RECOMMEND,
        confidence=0.9,
        entities={},
        sub_intents=[Intent.HANDLE_BIZ, Intent.QUERY_BILL, Intent.GENERAL_QA],
        reasoning="test dag rules"
    )
    orchestrator.classifier.classify = AsyncMock(return_value=mock_intent)
    
    # Mock Arbitrator
    orchestrator.arbitrator.detect = AsyncMock(return_value=ConflictReport(has_conflict=False))
    
    # 记录执行顺序
    execution_order = []
    
    async def mock_run_rec(*args, **kwargs):
        execution_order.append("recommend")
        return {"message": "Result from recommend: ok"}
        
    async def mock_run_handle(*args, **kwargs):
        execution_order.append("handle")
        return {"message": "Result from handle: ok"}
        
    async def mock_run_billing(*args, **kwargs):
        execution_order.append("billing")
        return {"message": "Result from billing: ok"}
        
    async def mock_run_qa(*args, **kwargs):
        execution_order.append("qa")
        return {"message": "Result from qa: ok"}

    orchestrator.recommend_agent.run = AsyncMock(side_effect=mock_run_rec)
    orchestrator.handle_agent.run = AsyncMock(side_effect=mock_run_handle)
    orchestrator.billing_agent.run = AsyncMock(side_effect=mock_run_billing)
    orchestrator.qa_agent.run = AsyncMock(side_effect=mock_run_qa)
    
    initial_state = {
        "session_id": "s4",
        "user_id": "u4",
        "user_input": "推荐办卡查账单并问个问题",
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
    
    result = await orchestrator.graph.ainvoke(initial_state)
    
    # 验证依赖规则：
    # RECOMMEND, BILL, QA 应该在第一轮并发
    # HANDLE 应该在第二轮
    # 所以 RECOMMEND 应该在 HANDLE 之前
    
    # 检查顺序中 recommend 是否在 handle 之前
    rec_idx = execution_order.index("recommend")
    handle_idx = execution_order.index("handle")
    assert rec_idx < handle_idx
    
    # 检查 billing 和 qa 是否也运行了
    assert "billing" in execution_order
    assert "qa" in execution_order
    
    # 验证最终回复汇总了所有结果
    assert "Result from recommend" in result["final_response"]
    assert "Result from handle" in result["final_response"]
    assert "Result from billing" in result["final_response"]
    assert "Result from qa" in result["final_response"]
