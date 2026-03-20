# Telecom-Agent: 通信运营商多智能体客服系统

## 项目概述

Telecom-Agent 是一个基于多智能体架构的智能客服系统，专为通信运营商设计。通过集成大语言模型（LLM）、向量数据库和多代理协作，实现高效、个性化的客户服务。系统支持意图识别、上下文记忆、多任务并行处理，并通过前后端分离架构提供流畅的用户体验。

## 核心进展总结

### 1. 业务系统仿真 (Mock API)
*   **功能扩展**：完成了宽带办理、话费充值、停复机、携号转网校验、套餐变更共 5 大类、12 个接口的开发。
*   **持久化增强**：引入了 SQLite 数据库 (`mock_business.db`)，实现业务数据的真实落盘与自动回显。
*   **异常模拟**：支持模拟接口调用超时、失败等异常场景，用于测试 Agent 的容错能力。

### 2. 核心智能体架构 (Autonomous ReAct)
*   **自主推理循环**：重构了 `BillingAgent` 和 `HandleAgent`，实现了标准的 ReAct (Reason-Action-Observation) 闭环。Agent 能够根据需求主动思考、选择工具、处理结果并自我修正。
*   **多专家协作控制**：通过 `Orchestrator` 实现多 Agent 并发调度，支持多意图并行处理与冲突仲裁。

### 3. 图驱动能力 (Graph Integration)
*   **工作流编排**：使用 `LangGraph` 定义了确定的 FSM（有限状态机）调度流程。
*   **GraphRAG**：实现了基于因果逻辑图的知识检索，极大提升了针对“由于/导致”类复杂业务逻辑的回答正确性。

### 4. 关键体验优化 (UX & Stability)
*   **真流式输出 (Real Streaming)**：重构了 LLM 物理链路，首字响应时间降低 80%，实现 Token 级实时反馈。
*   **路由精准修复**：修复了“套餐查询”意图误判、电话信息由于权限不足无法查库等关键 Bug。
*   **协议标准化**：统一了 LLM 消息的字典交换协议，解决数据处理崩溃问题。

## 解决的痛点挑战

基于需求说明中的具体痛点和挑战，系统通过以下技术手段解决：

### 1. 业务规则繁杂、办理步骤多
- **挑战**: 复杂业务需多步骤办理，易出错。
- **解决方案**: 多代理协同（QAAgent、HandleAgent），通过LangGraph状态图自动规划办理路径，UI卡片引导收集信息，一键完成。
- **技术实现**: 在`app/agents/orchestrator.py`中，`plan_node`方法根据意图分类结果构建任务DAG（Directed Acyclic Graph），自动识别依赖关系（如推荐必须在办理前执行）。`dispatch_node`异步执行任务，支持并发处理。示例：办理套餐时，先调用RecommendAgent推荐，再调用HandleAgent办理。

### 2. 办理临柜时间长，客户体验差
- **挑战**: 多菜单跳转，效率低。
- **解决方案**: 统一入口总控智能体，意图识别快速路由，流式回复减少等待。平均办理时间缩短80%。
- **技术实现**: FastAPI后端支持SSE（Server-Sent Events）流式响应，`app/api/chat.py`中的`chat_stream`端点实时推送消息。意图分类器`app/intent/classifier.py`使用LLM快速识别意图，避免多步导航。

### 3. 系统使用门槛高，培训成本高
- **挑战**: 传统菜单复杂，智能化不足。
- **解决方案**: 自然语言交互，LLM理解语义，无需培训。智能推荐降低操作门槛。
- **技术实现**: 基于OpenAI兼容LLM的意图识别，支持多轮对话上下文理解。`app/agents/recommend_agent.py`根据用户偏好智能推荐套餐，结合LTM用户画像个性化推荐。

### 场景一：数字营业员智能化能力升级

#### 复杂场景多智能体规划问题
- **挑战**: 高效拆解业务，规划最优路径。
- **解决方案**: LangGraph构建agentic_workflow，根据用户诉求动态规划子任务。支持任务清单生成，如办理套餐拆解为推荐+办理。
- **技术实现**: `Orchestrator._build_graph()`定义状态图节点（intent_node、plan_node、dispatch_node），使用条件边实现动态路由。`plan_node`创建任务DAG，支持依赖管理（如`depends_on`字段）。示例：用户说"办张手机卡"，自动拆解为推荐套餐→办理开户。

#### 多智能体场景冲突决策
- **挑战**: 代理输出冲突时解决。
- **解决方案**: Arbitrator代理检测冲突，通过上下文语义仲裁。支持场景内跳转，如办理中跳转问答。
- **技术实现**: `app/agents/arbitrator.py`的`detect`方法分析专家输出相似度，`arbitrate`方法使用LLM仲裁冲突。`aggregate_node`在汇总时调用仲裁逻辑，支持升级到人工客服。

#### 场景切换与任务接续
- **挑战**: 高效识别切换，管理上下文隔离。
- **解决方案**: 意图分类器识别多轮对话意图，自动切换代理。STM保持上下文，LTM隔离用户画像。
- **技术实现**: `switch_node`检测`need_switch`标记，动态注入新任务到DAG头部。`app/memory/stm.py`的`add_message`保存上下文快照，`app/memory/ltm.py`的`get_user_context`提供用户历史偏好。

### 场景二：基于记忆与智能编排的意图识别

#### 记忆缺失导致交互割裂
- **挑战**: 长流程遗忘关键信息。
- **解决方案**: STM维持当前任务流的状态，LTM存储历史偏好。记忆蒸馏压缩旧信息，锚点锁定关键诉求。
- **技术实现**: `ShortTermMemory.distill()`方法定期压缩超出10轮的对话，使用LLM生成摘要保留核心信息，同时保留标记为`is_anchor`的关键消息。`get_anchors()`返回锚点消息确保重要信息不丢失。

#### 工具调用的鲁棒性不足
- **挑战**: 动态环境调用失败，死循环。
- **解决方案**: 工具注册表支持动态调用，ReAct模式实现自主重试、多路径规划。异常反馈后动态编排。
- **技术实现**: `app/tools/registry.py`管理工具注册，支持动态加载。每个Agent继承ReAct模式，在`run`方法中实现观察-思考-行动循环。`dispatch_node`支持任务重试和多路径执行。

#### 意图理解的深度限制
- **挑战**: 无法区分瞬时诉求与长期偏好。
- **解决方案**: LTM基于RAG构建用户画像，结合历史记录嵌入。意图识别结合短期+长期记忆，实现个性化推荐与办理。
- **技术实现**: `LongTermMemory.search_knowledge()`使用向量相似度检索业务知识，`get_user_context()`获取用户画像摘要。意图分类器结合STM历史和LTM上下文进行深度理解。

### 技术实现细节
- **长期记忆存储**: Milvus向量数据库存储用户画像与业务知识，RAG辅助意图识别。`init_collections()`创建knowledge_base和user_profile集合，支持语义搜索。
- **动态编排**: 基于ReAct的工具链调度，自动规划调用序列。Agent基类`app/agents/base_agent.py`实现ReAct循环，支持异常处理和重试。
- **短期工作记忆**: Redis缓冲区，蒸馏技术保留关键路径，剔除噪声。`distill()`方法使用LLM压缩历史，保留锚点确保稳定性。

## 技术架构

### 后端
- **框架**: FastAPI
- **数据库**: Redis (STM), Milvus (LTM), Postgres (可选业务数据)
- **LLM**: OpenAI API 或兼容模型
- **工具**: 自定义工具注册表

### 前端
- **框架**: React + Vite
- **样式**: Tailwind CSS
- **状态管理**: React Hooks

### 基础设施
- Docker Compose: Redis, Milvus, Postgres
- 异步处理: Background Tasks

## 安装与运行指南

### 1. 环境准备
*   **Python**: 3.9 或更高版本。
*   **Docker**: 用于启动基础数据库服务。
*   **API Key**: 需要有效的 `QWEN_API_KEY`（在 `.env` 中配置）。

### 2. 安装与启动步骤
1.  **安装 Python 依赖项**：
    ```bash
    pip install -r requirements.txt
    ```
2.  **启动基础设施服务**（Milvus, Redis, Postgres）：
    ```bash
    docker-compose up -d
    ```
3.  **初始化知识库数据**：
    ```bash
    make ingest
    ```

### 3. 运行系统 (建议开启三个终端)
*   **终端 1：启动 Mock 业务系统**：
    ```bash
    make mock-api
    ```
*   **终端 2：启动 Agent 核心后端**：
    ```bash
    make run
    ```
*   **终端 3：启动前端界面 (React)**：
    ```bash
    cd frontend && npm install && npm run dev
    ```

### 4. 访问方式
*   **Web 界面**: `http://localhost:5173`
*   **API 文档**: `http://localhost:8000/docs`

## 使用方法

1. 在前端输入用户ID（如 user_01）。
2. 输入查询，如“查询我的账单”。
3. 系统自动处理，显示回复和记忆锚点。
4. 切换用户测试记忆隔离。

## 测试

运行单元测试:
```bash
python -m pytest tests/
```

## 贡献

欢迎提交Issue和PR。遵循代码规范，使用ESLint和Black。

## 许可证

MIT License
