import json
import asyncio
import logging
import re
from typing import AsyncGenerator, Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.memory.stm import ShortTermMemory
from app.memory.ltm import LongTermMemory
from app.intent.classifier import IntentClassifier, Intent
from app.tools.registry import ToolRegistry
from app.tools.init_tools import register_all_tools
from app.config import settings, get_redis_client

class ChatRequest(BaseModel):
    session_id: str
    user_id: str
    message: str

class ChatResponse(BaseModel):
    type: str
    content: Optional[str] = None
    intent: Optional[str] = None
    session_id: Optional[str] = None


# ---------- GraphRAG 上下文构建 ----------

async def build_graph_context(user_input: str) -> tuple[str, str, bool, bool]:
    causal_keywords = ["为什么", "原因", "导致", "由于", "如果", "后果", "触发", "怎么会", "假如"]
    is_causal_query = any(k in user_input for k in causal_keywords)

    entity_search = await ltm.search_knowledge(user_input, top_k=3)
    found_entities = []
    for d in entity_search:
        if d.get("doc_type") == "entity":
            try:
                name = d["content"].split(",")[0].replace("实体: ", "").strip()
                found_entities.append(name)
            except Exception:
                continue

    graph_facts: List[str] = []
    causal_paths: List[str] = []
    for entity in set(found_entities):
        rels = await ltm.search_related_entities(entity)
        graph_facts.extend([
            f"关系: {r['source']} --({r['relation']})--> {r['target']}" for r in rels
        ])
        if is_causal_query:
            paths = await ltm.search_causal_path(entity, depth=2)
            causal_paths.extend([
                f"因果链路: {p['source']} --({p['relation']})--> {p['target']}" for p in paths
            ])

    all_context = "\n".join(list(set(graph_facts + causal_paths)))
    return all_context, all_context, len(graph_facts) > 0, len(causal_hit := causal_paths) > 0 # 用到了冒号等号在 bool 判断中比较tricky，改简单点
    # 改正：return all_context, all_context, len(graph_facts) > 0, len(causal_paths) > 0


def _extract_phone(user_input: str, history: List[Dict]) -> Optional[str]:
    pattern = re.compile(r'1[3-9]\d{9}')
    match = pattern.search(user_input)
    if match:
        return match.group()
    for msg in reversed(history):
        match = pattern.search(msg.get("content", ""))
        if match:
            return match.group()
    return None


# ---------- 主 Chat 接口 ----------

@router.post("/chat/message")
async def chat_message(request: ChatRequest, background_tasks: BackgroundTasks):
    return StreamingResponse(
        event_generator(request, background_tasks),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive"
        }
    )

# ---------- 并行处理核心逻辑 ----------

async def handle_message(session_id: str, user_id: str, message: str):
    """
    非流式处理函数，用于性能测试和同步调用
    """
    redis_client = get_redis_client()
    try:
        stm = ShortTermMemory(session_id, redis_client)
        # 获取最近3轮对话（6条消息）
        history = await stm.get_history(max_turns=6)
        
        # 第一阶段：并行执行意图识别、RAG、用户信息查询
        graph_task = build_graph_context(message)
        intent_task = classifier.classify(message, history=history)
        user_task = ltm.get_user_context(user_id)
        
        results = await asyncio.gather(graph_task, intent_task, user_task, return_exceptions=True)
        
        graph_ctx_res = results[0] if not isinstance(results[0], Exception) else ("", "", False, False)
        intent_res = results[1] if not isinstance(results[1], Exception) else IntentResult(intent=Intent.GENERAL_QA, confidence=0.5, reasoning="Fallback")
        user_info = results[2] if not isinstance(results[2], Exception) else {}
        
        graph_context = graph_ctx_res[0]
        
        # 第二阶段：串行执行业务数据查询（如果需要）
        tool_output = None
        if intent_res.intent in [Intent.QUERY_PLAN, Intent.QUERY_BILL, Intent.COMPLAINT]:
            entities = intent_res.entities if isinstance(intent_res.entities, dict) else {}
            phone = entities.get("phone") or _extract_phone(message, history)
            if phone:
                res = await registry.call("get_user_info", {"phone": phone})
                tool_output = res.data if res.success else {"error": res.error}

        # 第三阶段：调用 Orchestrator 生成回复
        full_response = ""
        final_metadata = {}
        async for chunk in orchestrator.run_stream(
            user_input=message,
            session_id=session_id,
            user_id=user_id,
            stm=stm,
            graph_context=graph_context,
            tool_output=tool_output,
            intent_result=intent_res,
            user_info=user_info
        ):
            if chunk["type"] == "token":
                full_response += chunk["content"]
            elif chunk["type"] == "metadata":
                final_metadata = chunk["content"]
        
        return {
            "answer": full_response,
            "metadata": final_metadata,
            "intent": intent_res.intent
        }
    finally:
        await redis_client.aclose()

async def event_generator(request: ChatRequest, background_tasks: BackgroundTasks):
    redis_client = get_redis_client()
    try:
        session_id = request.session_id
        user_id = request.user_id
        user_input = request.message

        yield f"data: {json.dumps({'type': 'thinking', 'content': '正在加载上下文与识别意图...'}, ensure_ascii=False)}\n\n"
        stm = ShortTermMemory(session_id, redis_client)
        # 优化：并行化第一阶段任务
        history_task = stm.get_history(max_turns=6)
        user_info_task = ltm.get_user_context(user_id)
        
        # 先拿历史，因为意图识别依赖历史
        history = await history_task
        
        # 并行执行阶段 1
        results = await asyncio.gather(
            classifier.classify(user_input, history=history),
            build_graph_context(user_input),
            user_info_task,
            return_exceptions=True
        )
        
        intent_res = results[0] if not isinstance(results[0], Exception) else IntentResult(intent=Intent.GENERAL_QA, confidence=0.5, reasoning="Parallel Error")
        graph_ctx_res = results[1] if not isinstance(results[1], Exception) else ("", "", False, False)
        user_profile = results[2] if not isinstance(results[2], Exception) else {}
        
        graph_context, _, graph_hit, causal_hit = graph_ctx_res

        yield f"data: {json.dumps({'type': 'thinking', 'content': f'识别意图: {intent_res.intent} (置信度: {intent_res.confidence:.2f})'}, ensure_ascii=False)}\n\n"

        if causal_hit:
            yield f"data: {json.dumps({'type': 'thinking', 'content': '发现深度因果链路，正在进行逻辑诊断...'}, ensure_ascii=False)}\n\n"
        elif graph_hit:
            yield f"data: {json.dumps({'type': 'thinking', 'content': '已检索到业务知识图谱，正在优化回答...'}, ensure_ascii=False)}\n\n"

        tool_output = None
        if intent_res.intent in [Intent.QUERY_PLAN, Intent.QUERY_BILL, Intent.COMPLAINT]:
            entities = intent_res.entities if isinstance(intent_res.entities, dict) else {}
            phone = entities.get("phone") or _extract_phone(user_input, history)
            if phone:
                yield f"data: {json.dumps({'type': 'thinking', 'content': f'正在查询号码 {phone} 的业务数据...'}, ensure_ascii=False)}\n\n"
                res = await registry.call("get_user_info", {"phone": phone})
                tool_output = res.data if res.success else {"error": res.error}

        if intent_res.intent == Intent.RECOMMEND:
            yield f"data: {json.dumps({'type': 'thinking', 'content': '正在为您匹配最优套餐...'}, ensure_ascii=False)}\n\n"

        yield f"data: {json.dumps({'type': 'thinking', 'content': '正在生成回复...'}, ensure_ascii=False)}\n\n"
        
        full_response = ""
        final_metadata = {}
        
        # 核心变动：处理 Orchestrator 产出的结构化 yield
        async for chunk in orchestrator.run_stream(
            user_input=user_input,
            session_id=session_id,
            user_id=user_id,
            stm=stm,
            graph_context=graph_context,
            tool_output=tool_output,
            intent_result=intent_res,
            user_info=user_profile
        ):
            if chunk["type"] == "token":
                token = chunk["content"]
                full_response += token
                yield f"data: {json.dumps({'type': 'token', 'content': token}, ensure_ascii=False)}\n\n"
            elif chunk["type"] == "metadata":
                final_metadata = chunk["content"]

        # 7. 存储到 STM (包含元数据持久化)
        await stm.add_message("user", user_input)
        is_anchor = bool(intent_res.entities) or intent_res.intent in [
            Intent.RECOMMEND, Intent.HANDLE_BIZ, Intent.QUERY_BILL
        ]
        
        # 组装 metadata
        msg_metadata = {
            "is_anchor": is_anchor,
            "intent": intent_res.intent.value if isinstance(intent_res.intent, Intent) else str(intent_res.intent)
        }
        # 将专家产出的元数据（如 handle_state）合并进去
        if final_metadata:
            msg_metadata.update(final_metadata)
            
        await stm.add_message("assistant", full_response, msg_metadata)

        if len(history) >= 5:
            background_tasks.add_task(stm.distill)

        done_data = {
            'type': 'done',
            'intent': str(intent_res.intent),
            'session_id': session_id,
            'causal_hit': causal_hit,
            'graph_hit': graph_hit
        }
        yield f"data: {json.dumps(done_data, ensure_ascii=False)}\n\n"

    except Exception as e:
        logger.error(f"Chat stream error: {e}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"
    finally:
        await redis_client.aclose()

@router.get("/chat/history/{session_id}")
async def get_chat_history(session_id: str):
    redis_client = get_redis_client()
    stm = ShortTermMemory(session_id, redis_client)
    history = await stm.get_history()
    await redis_client.aclose()
    return {"session_id": session_id, "history": history}

@router.get("/chat/anchors/{session_id}")
async def get_anchors(session_id: str):
    redis_client = get_redis_client()
    stm = ShortTermMemory(session_id, redis_client)
    anchors = await stm.get_anchors()
    await redis_client.aclose()
    return {
        "session_id": session_id,
        "anchors": [
            a["content"][:30] + "..." if len(a["content"]) > 30 else a["content"]
            for a in anchors
        ]
    }

@router.delete("/chat/session/{session_id}")
async def clear_session(session_id: str, user_id: str):
    redis_client = get_redis_client()
    stm = ShortTermMemory(session_id, redis_client)
    history = await stm.get_history()
    if history:
        summary = f"会话 {session_id} 归档：{str(history)[:200]}..."
        await ltm.update_user_profile(user_id, summary)
    await stm.clear()
    await redis_client.aclose()
    return {"status": "success", "message": f"Session {session_id} archived and cleared."}
