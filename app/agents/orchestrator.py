import asyncio
import json
import logging
import operator
from typing import Dict, Any, List, Optional, TypedDict, Union, Annotated, AsyncGenerator
from langgraph.graph import StateGraph, END
from app.intent.classifier import IntentClassifier, Intent, IntentResult
from app.agents.qa_agent import QAAgent
from app.agents.recommend_agent import RecommendAgent
from app.agents.handle_agent import HandleAgent
from app.agents.billing_agent import BillingAgent
from app.agents.arbitrator import ConflictArbitrator
from app.memory.stm import ShortTermMemory
from app.tools.registry import ToolRegistry
from app.llm import chat

logger = logging.getLogger(__name__)

class OrchestratorState(TypedDict):
    session_id: str
    user_id: str
    user_input: str
    intent_result: Optional[IntentResult]
    task_dag: List[Dict[str, Any]]
    current_node: str
    context_snapshots: Dict[str, Any]
    final_response: str
    final_metadata: Dict[str, Any]
    fsm_state: str  # IDLE/PLANNING/EXECUTING/SWITCHING/RESUMING/COMPLETED
    expert_outputs: Annotated[List[Dict[str, Any]], operator.add]
    stm: ShortTermMemory
    registry: ToolRegistry
    graph_context: str
    tool_output: Optional[Dict[str, Any]]
    user_info: Optional[Dict[str, Any]]

class Orchestrator:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.qa_agent = QAAgent(tool_registry=registry)
        self.recommend_agent = RecommendAgent(tool_registry=registry)
        self.handle_agent = HandleAgent(tool_registry=registry)
        self.billing_agent = BillingAgent(tool_registry=registry)
        self.classifier = IntentClassifier()
        self.arbitrator = ConflictArbitrator()
        
        self.builder = StateGraph(OrchestratorState)
        self._build_graph()
        self.graph = self.builder.compile()

    def _build_graph(self):
        self.builder.add_node("intent_node", self.intent_node)
        self.builder.add_node("plan_node", self.plan_node)
        self.builder.add_node("dispatch_node", self.dispatch_node)
        self.builder.add_node("switch_node", self.switch_node)
        self.builder.add_node("aggregate_node", self.aggregate_node)

        self.builder.set_entry_point("intent_node")
        self.builder.add_edge("intent_node", "plan_node")
        self.builder.add_edge("plan_node", "dispatch_node")
        
        self.builder.add_conditional_edges(
            "dispatch_node",
            self.should_switch,
            {
                "switch": "switch_node",
                "dispatch": "dispatch_node",
                "aggregate": "aggregate_node"
            }
        )
        
        self.builder.add_edge("switch_node", "dispatch_node")
        self.builder.add_edge("aggregate_node", END)

    # --- 节点逻辑 ---

    async def intent_node(self, state: OrchestratorState):
        logger.info("--- Intent Node ---")
        if state.get("intent_result"):
            logger.info("Using pre-calculated intent result")
            return {"fsm_state": "PLANNING"}
            
        history = await state["stm"].get_history()
        intent_res = await self.classifier.classify(state["user_input"], history=history)
        return {"intent_result": intent_res, "fsm_state": "PLANNING"}

    async def plan_node(self, state: OrchestratorState):
        logger.info("--- Plan Node ---")
        from typing import cast
        intent_res_raw = state.get("intent_result")
        if not intent_res_raw:
            return {"task_dag": [], "fsm_state": "COMPLETED"}

        intent_res = cast(IntentResult, intent_res_raw)
        main_intent = str(intent_res.intent.value)
        sub_intents = [str(s.value) for s in intent_res.sub_intents]
        all_intents = list(dict.fromkeys([main_intent] + sub_intents)) # 去重

        task_dag: List[Dict[str, Any]] = []
        agent_assigned = set()
        tasks_map = {}
        
        for val in all_intents:
            if val == Intent.QUERY_PLAN:
                # 关键修复：套餐查询（特别是针对特定号码的）应由具备工具权限的账务专家处理，而非仅 RAG 的问答专家
                agent_name = "billing_agent"
            else:
                agent_name = self._map_intent_to_agent(val)
            # 关键改进：每个智能体在这轮对话中只处理一次任务（按意图优先级）
            if agent_name in agent_assigned:
                continue
            
            task_id = f"task_{val}"
            tasks_map[val] = task_id
            task_dag.append({
                "id": task_id,
                "intent": val,
                "agent": agent_name,
                "params": intent_res.entities,
                "status": "PENDING",
                "depends_on": []
            })
            agent_assigned.add(agent_name)
            
        if "handle_biz" in tasks_map and "recommend" in tasks_map:
            recommend_id = tasks_map["recommend"]
            for t in task_dag:
                if t["intent"] == "handle_biz":
                    current_deps = t.get("depends_on", [])
                    if isinstance(current_deps, list):
                        current_deps.append(recommend_id)
                        t["depends_on"] = current_deps
                    
        return {"task_dag": task_dag, "fsm_state": "EXECUTING", "expert_outputs": []}

    async def dispatch_node(self, state: OrchestratorState):
        logger.info("--- Dispatch Node ---")
        dag = list(state["task_dag"])
        done_ids = {t["id"] for t in dag if t["status"] == "DONE"}
        to_run = [
            t for t in dag
            if t["status"] == "PENDING" and all(dep in done_ids for dep in t.get("depends_on", []))
        ]
        
        if not to_run:
            if all(t["status"] == "DONE" for t in dag):
                return {"fsm_state": "COMPLETED"}
            return {"fsm_state": "EXECUTING"}

        logger.info(f"Dispatching tasks: {[t['id'] for t in to_run]}")
        
        async def run_task(task):
            agent_name = task["agent"]
            agent = self._get_agent_instance(agent_name)
            res = await agent.run(
                user_input=state["user_input"],
                session_id=state["session_id"],
                user_id=state["user_id"],
                stm=state["stm"]
            )
            if isinstance(res, dict) and res.get("need_switch"):
                logger.info(f"Task {task['id']} interrupted for switch")
            else:
                task["status"] = "DONE"
            return {"task_id": task["id"], "agent": agent_name, "output": res} # 增加 agent 标识

        results = await asyncio.gather(*(run_task(t) for t in to_run))
        return {"expert_outputs": list(results), "task_dag": dag}

    def should_switch(self, state: OrchestratorState):
        for out in state["expert_outputs"]:
            if isinstance(out["output"], dict) and out["output"].get("need_switch"):
                return "switch"
        if any(t["status"] == "PENDING" for t in state["task_dag"]):
            return "dispatch"
        return "aggregate"

    async def switch_node(self, state: OrchestratorState):
        logger.info("--- Switch Node ---")
        target_switch = None
        for out in state["expert_outputs"]:
            if isinstance(out["output"], dict) and out["output"].get("need_switch"):
                target_switch = out["output"]
                out["output"]["need_switch"] = None
                break
        
        if not target_switch:
            return {"fsm_state": "EXECUTING"}

        new_task = {
            "id": f"switch_{len(state['task_dag'])}",
            "agent": target_switch["need_switch"],
            "params": {"reason": target_switch.get("reason")},
            "status": "PENDING",
            "depends_on": []
        }
        updated_dag = [new_task] + state["task_dag"]
        return {"task_dag": updated_dag, "fsm_state": "SWITCHING"}

    async def aggregate_node(self, state: OrchestratorState):
        logger.info("--- Aggregate Node ---")
        outputs = state["expert_outputs"]
        
        # 1. 冲突检测与仲裁 (保持原有逻辑)
        conflict = await self.arbitrator.detect(outputs, state["stm"])
        if conflict.has_conflict:
            arb_res = await self.arbitrator.arbitrate(conflict, state["user_input"], outputs)
            if arb_res.resolved:
                winner_out = next((o for o in outputs if o["task_id"] == arb_res.winner), None)
                if winner_out:
                    final_msg = f"【系统已自动消除专家意见冲突】仲裁原因：{arb_res.reason}\n\n"
                    final_msg += self._get_text_content(winner_out["output"])
                    winner_metadata = {k: v for k, v in winner_out["output"].items() if k not in ["answer", "message"]} if isinstance(winner_out["output"], dict) else {}
                    return {"final_response": final_msg, "final_metadata": winner_metadata, "fsm_state": "COMPLETED"}

        # 2. 智能优先级汇总逻辑
        # 核心：如果 HandleAgent 在执行业务确认或完成，它的回复具有最高优先级，忽略 QAAgent 的冗余回复
        handle_out = next((o for o in outputs if o.get("agent") == "handle_agent"), None)
        qa_out = next((o for o in outputs if o.get("agent") == "qa_agent"), None)
        
        final_msg_parts = []
        aggregated_meta = {}

        # 逻辑：如果 HandleAgent 产出了内容且对话输入包含“确认/好的”等词，极大概率是业务办理成功
        is_confirmation = any(w in state["user_input"] for w in ["确认", "好的", "没问题", "确认办理"])
        
        handle_done = False
        if handle_out and is_confirmation and isinstance(handle_out.get("output"), dict):
            handle_done = handle_out["output"].get("handle_state", {}).get("state") == "DONE"
        
        if handle_out and is_confirmation and handle_done:
            # 这种情况下，HandleAgent 的回复最专业且包含结构化办理结果，通常不需要 QAAgent 再通过 RAG 重复一遍“办理成功”
            data = handle_out["output"]
            final_msg_parts.append(self._get_text_content(data))
            if isinstance(data, dict):
                aggregated_meta.update({k: v for k, v in data.items() if k not in ["answer", "message"]})
            # 也可以考虑增加一些 QAAgent 的独特信息（如果是补充性的），但如果是“办卡成功”这种完全重合的，就跳过 QA
            # 这里简单处理：若 HandleAgent 已输出，且 QA 内容包含“成功”等关键词，则过滤 QA
            if qa_out:
                qa_text = self._get_text_content(qa_out["output"])
                if "成功" not in qa_text or "办理" not in qa_text: # 如果 QA 提供了额外非重复信息，保留
                    final_msg_parts.append(qa_text)
                    if isinstance(qa_out["output"], dict):
                        aggregated_meta.update({k: v for k, v in qa_out["output"].items() if k not in ["answer", "message"]})
        else:
            # 默认汇总模式
            for out in outputs:
                data = out["output"]
                final_msg_parts.append(self._get_text_content(data))
                if isinstance(data, dict):
                    meta = {k: v for k, v in data.items() if k not in ["answer", "message"]}
                    aggregated_meta.update(meta)
        
        return {
            "final_response": "\n".join(dict.fromkeys(final_msg_parts)).strip(), # 基础去重
            "final_metadata": aggregated_meta,
            "fsm_state": "COMPLETED"
        }

    def _get_text_content(self, data: Any) -> str:
        if isinstance(data, dict):
            return data.get("answer", data.get("message", data.get("bill_summary", "")))
        return str(data)

    async def run_stream(
        self,
        user_input: str,
        session_id: str,
        user_id: str,
        stm: ShortTermMemory,
        graph_context: str = "",
        tool_output: Optional[Dict[str, Any]] = None,
        intent_result: Optional[IntentResult] = None,
        user_info: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        initial_state: OrchestratorState = {
            "session_id": session_id,
            "user_id": user_id,
            "user_input": user_input,
            "intent_result": None,
            "task_dag": [],
            "current_node": "intent_node",
            "context_snapshots": {},
            "final_response": "",
            "final_metadata": {},
            "fsm_state": "IDLE",
            "expert_outputs": [],
            "stm": stm,
            "registry": self.registry,
            "graph_context": graph_context,
            "tool_output": tool_output,
            "intent_result": intent_result,
            "user_info": user_info
        }
        
        full_response = ""
        try:
            final_state = await self.graph.ainvoke(initial_state)
            expert_outputs = final_state.get("expert_outputs", [])
            final_response = final_state.get("final_response", "")
            metadata = final_state.get("final_metadata", {})

            # 优化：如果只有一个专家有输出且没有被 aggregate 节点汇总
            # 直接使用该专家的输出，避免二次 LLM 汇总带来的 TTFT 延迟
            if not final_response and len(expert_outputs) == 1:
                out = expert_outputs[0]["output"]
                final_response = self._get_text_content(out)
                if isinstance(out, dict):
                    expert_meta = {k: v for k, v in out.items() if k not in ["answer", "message"]}
                    metadata.update(expert_meta)

            if not final_response and len(expert_outputs) > 1:
                # 实施“真流式汇总”
                sources_text = "\n".join([f"专家[{o['agent']}]: {self._get_text_content(o['output'])}" for o in expert_outputs])
                prompt = f"请根据以下多个专家的意见，统一汇总成一段专业、亲切的回复给用户。只需回复汇总内容，不要提及‘专家’等词汇。\n\n[专家意见]\n{sources_text}\n\n[用户输入]\n{user_input}"
                
                async for token in chat([{"role": "user", "content": prompt}], stream=True):
                    full_response += token
                    yield {"type": "token", "content": token}
            elif final_response:
                # 即使是已生成的结果，为了视觉连贯性，我们分成小块快速吐出
                chunk_s = 8
                for i in range(0, len(final_response), chunk_s):
                    token = final_response[i:i + chunk_s]
                    yield {"type": "token", "content": token}
                    await asyncio.sleep(0.005) 
                full_response = final_response
            
            yield {"type": "metadata", "content": metadata}
                
        except Exception as e:
            logger.error(f"Orchestrator run_stream error: {e}", exc_info=True)
            yield {"type": "token", "content": f"系统运行时出现错误：{str(e)}"}

    def _map_intent_to_agent(self, intent_name: str) -> str:
        mapping = {
            "query_plan": "qa_agent",
            "recommend": "recommend_agent",
            "handle_biz": "handle_agent",
            "query_bill": "billing_agent",
            "complaint": "qa_agent",
            "general_qa": "qa_agent",
            "unknown": "qa_agent"
        }
        return mapping.get(intent_name, "qa_agent")

    def _get_agent_instance(self, name: str):
        mapping = {
            "qa_agent": self.qa_agent,
            "recommend_agent": self.recommend_agent,
            "handle_agent": self.handle_agent,
            "billing_agent": self.billing_agent
        }
        return mapping.get(name, self.qa_agent)
