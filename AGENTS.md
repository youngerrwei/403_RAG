# AGENTS.md — AI Agent 代码修改规范指南

> 本文档面向 AI Agent 和开发者，作为修改本项目代码时的规范参考。

---

## 1. 项目架构概览

### 核心文件职责

| 文件 | 职责 |
|------|------|
| `rag_agent.py` | 核心 RAG 流程：路由判断、查询改写、向量检索、重排序、LLM 生成 |
| `ingest.py` | 知识入库：文档加载、文本清洗、分块切分、向量化、写入 Qdrant |
| `web_app.py` | Flask 服务：用户认证、SSE 流式接口、对话历史管理 |
| `logger.py` | 统一日志模块：提供 `get_logger()` 接口，支持控制台+文件双输出、按日期自动轮转 |
| `create_user.py` | 用户创建脚本：交互式创建/更新用户凭据（PBKDF2 加密） |
| `auto_ingest.sh` | 入库管理脚本：支持增量入库、全量入库、集合销毁 |
| `start_rag.sh` | 服务启动管理脚本：启动前预检（.env、模型路径、Qdrant、目录验证）、环境变量全量加载、vLLM/web_app 启动管理、通过 /api/health 验证就绪性 |
| `templates/index.html` | 对话前端页面 |
| `templates/login.html` | 登录页面 |
| `.env` | 所有配置项（模型路径、Qdrant 连接、检索参数等） |

### 文件间依赖关系

```
web_app.py ──导入并调用──▶ rag_agent.py
     │                        │
     │                        ▼
     │                   Qdrant 集合 ◀── ingest.py（独立运行，共享集合）
     │                                        ▲
     │                                        │
     │          auto_ingest.sh ────调用────────┘
     │                │
     ▼                ▼
  logger.py ◀──── 被所有 Python 主模块导入（共享日志基础设施）
```

- `web_app.py` 导入并调用 `rag_agent.py` 中的核心函数
- `ingest.py` 独立运行，与 `rag_agent.py` 共享 Qdrant 集合
- `auto_ingest.sh` 调用 `ingest.py` 执行入库操作
- `logger.py` 被 `web_app.py`、`rag_agent.py`、`ingest.py` 共同导入，提供统一日志能力

---

## 2. 修改代码前的准备

1. **阅读相关文件的完整代码**，理解上下文和现有实现
2. **检查 `.env` 文件**了解当前配置项及其取值
3. **确认修改范围**，避免与其他并行修改冲突
4. **如果涉及检索逻辑变更**：
   - 了解当前的 Qdrant 集合结构（子块集合 + 父块集合）
   - 确认 metadata 字段、embedding 维度等细节
5. **如果涉及前端修改**：
   - 先了解现有的 SSE 数据格式（见第 7 节）
   - 理解 DOM 结构和事件监听逻辑

---

## 3. 代码修改规范

### 3.1 代码风格

- 使用 `log()` 和 `debug_log()` 输出日志，**不要用 `print()`**
- 遵循现有命名风格：
  - 函数名：`snake_case`
  - 常量名：`UPPER_CASE`
- 保持适当注释，关键逻辑需有**中文注释**说明

### 3.2 依赖管理

- **尽量不引入新的 pip 依赖**
- 如必须引入，需同步更新 README 中的安装说明

### 3.3 配置管理

- 新增配置项**必须**在 `.env` 中添加对应条目
- 新增配置项**必须**在 `load_config()` 中注册读取
- 代码中的默认值应与 `.env` 推荐值保持一致
- **不要硬编码可调参数**，所有可调参数走配置

```python
# ✅ 正确
TOP_K = int(os.getenv("TOP_K", "10"))

# ❌ 错误
TOP_K = 10  # 硬编码
```

### 3.4 错误处理

- 关键操作**必须** try-catch
- 对外错误响应使用通用提示（如 "服务暂时不可用"），详细错误写入日志
- 网络请求**必须**设置超时参数

```python
# ✅ 正确
try:
    resp = requests.post(url, json=payload, timeout=30)
except Exception as e:
    log(f"请求失败: {e}")
    return "服务暂时不可用，请稍后重试"
```

---

## 4. 修改后的验证步骤

### 4.1 语法检查

```bash
python -c "import ast; ast.parse(open('文件名.py').read())"
```

### 4.2 导入检查

确认新增模块已在文件顶部正确导入，不存在循环导入。

### 4.3 逻辑一致性

| 检查项 | 说明 |
|--------|------|
| 配置对应 | `.env` 中定义的配置项与代码中 `os.getenv()` 的 key 一致 |
| 函数签名 | 函数签名变更后，**所有调用处**已同步更新 |
| 并发安全 | `Semaphore` / `Lock` 的 acquire/release 必须配对，推荐 `with` 语句 |

### 4.4 兼容性

- 增量入库与全量入库都应正常工作
- SSE 新事件类型不应破坏旧版前端解析逻辑
- 新增 API 不应影响已有接口的行为

---

## 5. 版本控制规范

- **版本号文件**：`VERSION`（纯数字，semver 格式如 `1.1.0`）
- **变更日志**：`CHANGELOG.md`（每次发布必须更新）
- 格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/)

### 版本号策略

| 变更类型 | 版本位 | 示例 |
|----------|--------|------|
| 修复 Bug | PATCH +1 | `1.1.0` → `1.1.1` |
| 新增功能 | MINOR +1 | `1.1.0` → `1.2.0` |
| 不兼容变更 | MAJOR +1 | `1.1.0` → `2.0.0` |

---

## 6. Qdrant 集合管理注意事项

### 集合说明

| 集合名 | 用途 | 向量维度 |
|--------|------|----------|
| `lab_knowledge_base` | 子块集合，用于相似度检索 | 1024（bge-m3） |
| `lab_knowledge_base_parents` | 父块集合，用于上下文展开 | 4（零向量占位，不参与计算） |

### 何时需要全量重建

修改入库逻辑后，如果改变了以下任一项，**必须全量重建**：

- metadata 字段（新增/删除/重命名）
- chunk 大小或前缀格式
- UUID 生成逻辑

### 全量重建命令

```bash
bash auto_ingest.sh --destroy --force && bash auto_ingest.sh --full
```

---

## 7. SSE 数据协议

### 当前事件类型

```jsonc
// 流开始（web_app.py 注入）
data: {"type": "start", "question": "..."}

// 元数据：路由决策结果（rag_agent.py 发出）
data: {"type": "metadata", "stage": "route", "route": "rag_search|file_list|hybrid", "route_target": "...", "route_reason": "..."}

// 状态变更通知（检索中、生成中）
data: {"type": "status", "stage": "searching"}
data: {"type": "status", "stage": "generating"}

// 工具调用结果（文件系统浏览等）
data: {"type": "tool", "tool_name": "list_catalog_entries", "content": [...]}

// 元数据：检索详情（改写后问题、关键词、召回文档）
data: {"type": "metadata", "stage": "retrieval", "rewritten_question": "...", "keywords": [...], "queries": [...], "retrievals": [...], "source_map": {...}}

// 流式文本片段
data: {"type": "chunk", "content": "..."}

// 元数据：检索覆盖率、引用来源
data: {"type": "metadata", "stage": "coverage", "coverage": {...}, "citations": [...]}

// 结束标记（携带最终元数据）
data: {"type": "final", "content": "...", "route": "...", "coverage": {...}, "citations": [...]}

// 心跳包（web_app.py 注入，保持连接活跃）
data: {"type": "heartbeat"}

// 错误通知（web_app.py 使用 "message" 字段，rag_agent.py 使用 "content" 字段）
data: {"type": "error", "message": "..."}  // web_app.py 超时/异常
data: {"type": "error", "content": "..."}  // rag_agent.py 内部异常

// 流结束信号（固定格式，不是 JSON）
data: [DONE]
```

### 修改规则

- 修改 SSE 协议时**必须同步更新**前端 `templates/index.html` 的解析逻辑
- 新增事件类型时确保前端 `onmessage` 处理中有对未知类型的兜底逻辑
- 保持 `[DONE]` 作为流终止的最后一条消息

---

## 8. 常见陷阱

| 陷阱 | 说明 |
|------|------|
| `.env` 配置优先级 | `.env` 中的值优先级高于代码默认值；但如果 `.env` 未部署，则回退到代码默认值 |
| MatchText 索引要求 | Qdrant 的 `MatchText` 过滤需要对应字段已创建**文本索引**，否则查询会失败 |
| bge-m3 维度 | embedding 维度固定为 **1024**，集合创建时必须匹配 |
| 父块集合向量 | 父块 collection 使用 4 维零向量占位，**不参与相似度计算** |
| `fcntl.flock` 跨平台 | 文件锁在 Windows/macOS 上行为可能与 Linux 不同，需注意兼容性 |
| `teardown_request` | Flask 的 `teardown_request` 无论请求成功或失败**都会执行**，不要在其中做条件性清理 |
| `preflight_check()` 行为 | 启动前预检中 Qdrant 连通性检查失败仅发出警告（不阻止启动），允许 Qdrant 服务稍后启动；模型路径不存在则中止启动 |
| `/api/health` 状态含义 | `"degraded"` 表示部分组件不可用但系统仍可接受请求；`"error"` 表示关键服务不可用；修改状态判断逻辑时需同步更新 `start_rag.sh` 中的 JSON 解析逻辑 |

---

## 9. 配置共识（所有 Agent 必须遵守）

以下配置值为项目确认的正确值，修改代码时**必须**使用这些默认值：

| 配置项 | 正确默认值 | 说明 |
|--------|-----------|------|
| `QDRANT_HOST` | `172.18.216.71` | Qdrant 远程服务地址 |
| `QDRANT_PORT` | `6333` | Qdrant 服务端口 |
| `EMBEDDING_MODEL_NAME` | `./models/bge-m3` | 本地 Embedding 模型路径 |
| `EMBEDDING_DEVICE` | `cuda:2` | Embedding 运行设备 |
| `RERANKER_MODEL_NAME` | `./models/bge-reranker-v2-m3` | 本地 Reranker 模型路径 |
| `RERANKER_DEVICE` | `cuda:2` | Reranker 运行设备 |
| `VLLM_API_KEY` | `lab-secret-key` | vLLM 服务密钥 |
| `VLLM_MODEL_NAME` | `./models/Qwen3-8B-Instruct` | vLLM 推理模型本地路径 |
| `FILE_SEARCH_LIMIT` | `200` | 文件搜索结果上限 |

**注意：** 不要使用 HuggingFace 在线路径（如 `BAAI/bge-m3`），统一使用本地模型路径。

---

## 10. API 端点共识

### GET /api/health（无认证）

| 字段 | 值/格式 | 说明 |
|------|--------|------|
| 路由 | `/api/health` | 固定路由，无需认证 |
| 方法 | `GET` | 只读操作 |
| `status` | `"ok"` / `"degraded"` / `"error"` | 系统整体状态 |
| `components` | `{"embedding": bool, "reranker": bool, "qdrant": bool, "llm": bool}` | 各组件状态 |
| HTTP 200 | status="ok" 或 "degraded" | 系统可用 |
| HTTP 503 | status="error" | 系统不可用 |

**修改规则：**
- 该端点为系统稳定 API，路由和响应格式不可随意变更
- 扩展检查项时向 `components` 对象中添加新键值对，不删除已有键
- 不要改变 `status` 的三个可能值定义
- 修改此端点的响应格式时，**必须同步更新** `start_rag.sh` 中的 JSON 解析逻辑
