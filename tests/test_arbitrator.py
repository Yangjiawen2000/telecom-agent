import pytest
from unittest.mock import AsyncMock, MagicMock
from app.agents.arbitrator import ConflictArbitrator, ConflictReport
from app.memory.stm import ShortTermMemory

@pytest.mark.asyncio
async def test_arbitrator_priority_rule():
    """
    验证优先级规则：业务系统数据 (HandleAgent) > 纯文本结论 (QAAgent)
    """
    arbitrator = ConflictArbitrator()
    mock_stm = MagicMock(spec=ShortTermMemory)
    
    # 构造冲突数据
    results = [
        {
            "task_id": "task_1_qa_agent",
            "output": {"answer": "根据知识库，您可以正常办理 5G 套餐。", "confidence": 0.9}
        },
        {
            "task_id": "task_2_handle_agent",
            "output": {"message": "办理失败：检测到您的账户处于欠费停机状态，无法办理新业务。", "status": "ERROR"}
        }
    ]
    
    # 1. 测试检测 (由于相似度低且结论矛盾，应识别为 SEMANTIC 冲突)
    # 注意：这里会调用 LLM 进行语义判断和 embed，我们在测试中 mock 掉相关调用
    arbitrator._get_similarity = AsyncMock(return_value=0.2)
    arbitrator._check_logical_contradiction = AsyncMock(return_value=True)
    
    report = await arbitrator.detect(results, mock_stm)
    
    assert report.has_conflict is True
    assert report.conflict_type == "SEMANTIC"
    assert "task_2_handle_agent" in report.parties
    
    # 2. 测试仲裁
    # 规则：handle_agent 优先级高于 qa_agent
    arb_res = await arbitrator.arbitrate(report, "我想办套餐", results)
    
    assert arb_res.resolved is True
    assert arb_res.winner == "task_2_handle_agent"
    assert "业务系统实时数据" in arb_res.reason
    assert arb_res.escalate is False

@pytest.mark.asyncio
async def test_arbitrator_llm_referee():
    """
    如果规则无法消解，验证是否调用 LLM 裁判
    """
    arbitrator = ConflictArbitrator()
    
    # 构造两个同级别的 QA 专家冲突
    results = [
        {"task_id": "task_1_qa_agent", "output": {"answer": "结论 A"}},
        {"task_id": "task_2_qa_agent", "output": {"answer": "结论 B"}}
    ]
    
    report = ConflictReport(
        has_conflict=True,
        conflict_type="SEMANTIC",
        parties=["task_1_qa_agent", "task_2_qa_agent"],
        description="两个 QA 专家结论不一"
    )
    
    # Mock LLM 裁判返回 JSON
    from app.agents.arbitrator import chat
    import json
    
    mock_llm_response = json.dumps({
        "winner": "task_1_qa_agent",
        "reason": "证据更充分",
        "confidence": 0.95
    })
    
    # 局部 mock app.agents.arbitrator.chat
    import app.agents.arbitrator
    app.agents.arbitrator.chat = AsyncMock(return_value=mock_llm_response)
    
    arb_res = await arbitrator.arbitrate(report, "上下文", results)
    
    assert arb_res.resolved is True
    assert arb_res.winner == "task_1_qa_agent"
    assert "LLM 仲裁结论" in arb_res.reason
