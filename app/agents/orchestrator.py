import asyncio
import json
import logging
import operator
from typing import Dict, Any, List, Optional, TypedDict, Union, Annotated
from langgraph.graph import StateGraph, END
from app.intent.classifier import IntentClassifier, IntentResult
from app.agents.qa_agent import QAAgent
from app.agents.recommend_agent import RecommendAgent
from app.agents.handle_agent import HandleAgent
from app.agents.billing_agent import BillingAgent
from app.agents.arbitrator import ConflictArbitrator
from app.memory.stm import ShortTermMemory
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

class OrchestratorState(TypedDict):
    session_id: str
    user_id: str
    user_input: str
    intent_result: Optional[IntentResult]
    task_dag: List[Dict[str, Any]] # 默认替换
    current_node: str
    context_snapshots: Dict[str, Any]
    final_response: str
    fsm_state: str # IDLE/PLANNING/EXECUTING/SWITCHING/RESUMING/COMPLETED
    expert_outputs: Annotated[List[Dict[str, Any]], operator.add]
    stm: ShortTermMemory
    registry: ToolRegistry

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
        # 定义节点
        self.builder.add_node("intent_node", self.intent_node)
        self.builder.add_node("plan_node", self.plan_node)
        self.builder.add_node("dispatch_node", self.dispatch_node)
        self.builder.add_node("switch_node", self.switch_node)
        self.builder.add_node("aggregate_node", self.aggregate_node)

        # 定义边
        self.builder.set_entry_point("intent_node")
        self.builder.add_edge("intent_node", "plan_node")
        self.builder.add_edge("plan_node", "dispatch_node")
        
        # 条件边
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
        logger.info(f"--- Intent Node ---")

        # 从 STM 获取历史记录 (可选)
        history = await state["stm"].get_history()
        
        intent_res = await self.classifier.classify(
            state["user_input"],
            history=history
        )
        return {"intent_result": intent_res, "fsm_state": "PLANNING"}

    async def plan_node(self, state: OrchestratorState):
        logger.info(f"--- Plan Node ---")
        from typing import cast
        intent_res_raw = state.get("intent_result")
        if not intent_res_raw:
            return {"task_dag": [], "fsm_state": "COMPLETED"}

        # 显式转换类型以满足 linter
        intent_res = cast(IntentResult, intent_res_raw)
        
        main_intent = str(intent_res.intent.value)
        sub_intents = [str(s.value) for s in intent_res.sub_intents]
        
        all_intents = [main_intent] + sub_intents
        task_dag: List[Dict[str, Any]] = []
        
        # 1. 创建所有任务节点
        tasks_map = {}
        for i, val in enumerate(all_intents):
            task_id = f"task_{i}_{val}"
            tasks_map[val] = task_id
            task_dag.append({
                "id": task_id,
                "intent": val,
                "agent": self._map_intent_to_agent(val),
                "params": intent_res.entities,
                "status": "PENDING",
                "depends_on": []
            })
            
        # 2. 应用规则：RECOMMEND 必须在 HANDLE_BIZ 之前
        if "handle_biz" in tasks_map and "recommend" in tasks_map:
            recommend_id = tasks_map["recommend"]
            for t in task_dag:
                if t["intent"] == "handle_biz":
                    # 显式确保 depends_on 是列表
                    current_deps = t.get("depends_on", [])
                    if isinstance(current_deps, list):
                        current_deps.append(recommend_id)
                        t["depends_on"] = current_deps
                    
        return {"task_dag": task_dag, "fsm_state": "EXECUTING", "expert_outputs": []}

    async def dispatch_node(self, state: OrchestratorState):
        logger.info(f"--- Dispatch Node ---")
        dag = list(state["task_dag"])
        
        # 找到已完成的任务 ID 集合
        done_ids = {t["id"] for t in dag if t["status"] == "DONE"}
        
        # 找到所有可以执行的任务 (PENDING 且 依赖已完成)
        to_run = []
        for t in dag:
            if t["status"] == "PENDING":
                if all(dep in done_ids for dep in t.get("depends_on", [])):
                    to_run.append(t)
        
        if not to_run:
            if all(t["status"] == "DONE" for t in dag):
                return {"fsm_state": "COMPLETED"}
            return {"fsm_state": "EXECUTING"} # 可能在等待切换恢复

        logger.info(f"Dispatching tasks: {[t['id'] for t in to_run]}")
        new_outputs = []
        
        async def run_task(task):
            agent_name = task["agent"]
            agent = self._get_agent_instance(agent_name)
            res = await agent.run(
                state["user_input"], 
                state["stm"]
            )
            
            if isinstance(res, dict) and res.get("need_switch"):
                logger.info(f"Task {task['id']} interrupted")
            else:
                task["status"] = "DONE"
                
            return {"task_id": task["id"], "output": res}

        results = await asyncio.gather(*(run_task(t) for t in to_run))
        new_outputs.extend(results)
            
        return {"expert_outputs": new_outputs, "task_dag": dag}

    def should_switch(self, state: OrchestratorState):
        # 1. 检查是否有专家要求跳转
        # 注意：只检查最新一轮产生的输出 (虽然使用了 operator.add，但我们可以看最后几个)
        # 这里的专家输出包含 task_id，可以辅助判断
        for out in state["expert_outputs"]:
            if isinstance(out["output"], dict) and out["output"].get("need_switch"):
                return "switch"
        
        # 2. 如果还有未完成的任务，继续 dispatch
        if any(t["status"] == "PENDING" for t in state["task_dag"]):
            return "dispatch"
            
        return "aggregate"

    async def switch_node(self, state: OrchestratorState):
        logger.info(f"--- Switch Node ---")
        # 找到需要跳转的任务
        target_switch = None
        for out in state["expert_outputs"]:
            if isinstance(out["output"], dict) and out["output"].get("need_switch"):
                target_switch = out["output"]
                # 清除标记避免死循环
                out["output"]["need_switch"] = None 
                break
        
        if not target_switch:
            return {"fsm_state": "EXECUTING"}

        # 1. 保存快照 (模拟)
        # TODO: 实际可调用 stm.add_message 保存当前 dag 状态
        
        # 2. 注入新任务到 DAG 头部
        new_task = {
            "id": f"switch_{len(state['task_dag'])}",
            "agent": target_switch["need_switch"],
            "params": {"reason": target_switch.get("reason")},
            "status": "PENDING"
        }
        
        updated_dag = [new_task] + state["task_dag"]
        return {"task_dag": updated_dag, "fsm_state": "SWITCHING"}

    async def aggregate_node(self, state: OrchestratorState):
        logger.info(f"--- Aggregate Node ---")
        outputs = state["expert_outputs"]
        
        # 1. 冲突检测与仲裁
        conflict = await self.arbitrator.detect(
            outputs, 
            state["stm"]
        )
        if conflict.has_conflict:
            logger.warning(f"Conflict detected: {conflict.conflict_type}")
            arb_res = await self.arbitrator.arbitrate(
                conflict, 
                state["user_input"], 
                outputs
            )
            
            if arb_res.resolved:
                logger.info(f"Conflict resolved. Winner: {arb_res.winner}")
                winner_out = next((o for o in outputs if o["task_id"] == arb_res.winner), None)
                if winner_out:
                    final_msg = f"【系统提示：检测到信息冲突，已根据系统规则选择最优结论】\n"
                    final_msg += f"仲裁原因：{arb_res.reason}\n\n"
                    final_msg += self._get_text_content(winner_out["output"])
                    return {"final_response": final_msg, "fsm_state": "COMPLETED"}
            
            if arb_res.escalate:
                return {
                    "final_response": "抱歉，系统检测到专家建议存在严重冲突且无法自动消除，为确保准确性，已为您转接人工客服处理。",
                    "fsm_state": "COMPLETED"
                }

        # 2. 正常汇总逻辑
        final_msg = ""
        for out in outputs:
            data = out["output"]
            final_msg += self._get_text_content(data) + "\n"
        
        return {"final_response": final_msg.strip(), "fsm_state": "COMPLETED"}

    def _get_text_content(self, data: Any) -> str:
        if isinstance(data, dict):
            return data.get("message", data.get("answer", data.get("answer", "")))
        return str(data)

    # --- 辅助方法 ---

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
