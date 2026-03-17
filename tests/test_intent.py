import pytest
from app.intent.classifier import IntentClassifier, Intent

@pytest.mark.asyncio
async def test_intent_cases():
    classifier = IntentClassifier()
    
    # 1. 套餐查询
    res = await classifier.classify("帮我查一下套餐")
    assert res.intent == Intent.QUERY_PLAN
    assert res.confidence > 0.8

    # 2. 业务办理 (新办卡)
    res = await classifier.classify("我要办一张新卡")
    assert res.intent == Intent.HANDLE_BIZ
    assert res.entities.get("action") or "办" in res.reasoning

    # 3. 投诉/停机原因 (需要触发 RAG)
    res = await classifier.classify("为什么停机了")
    assert res.intent in [Intent.COMPLAINT, Intent.GENERAL_QA]
    
    # 4. 涉诈解封 (复杂实体)
    res = await classifier.classify("18612345678已解除涉诈，帮我开机")
    assert res.intent == Intent.HANDLE_BIZ
    assert res.entities.get("phone") == "18612345678"

    # 5. 复合意图
    res = await classifier.classify("办一张卡顺便查一下账单")
    assert res.intent in [Intent.HANDLE_BIZ, Intent.QUERY_BILL]
    # 检查 sub_intents 或者 reasoning 是否识别到了两个意图
    all_intents = [res.intent] + res.sub_intents
    assert Intent.HANDLE_BIZ in all_intents
    assert Intent.QUERY_BILL in all_intents

@pytest.mark.asyncio
async def test_entity_extraction():
    classifier = IntentClassifier()
    text = "我叫张三，我的电话是15688889999，想换那个畅越129套餐"
    entities = await classifier.extract_entities(text)
    
    assert entities.get("phone") == "15688889999"
    assert "129" in str(entities.get("plan"))
