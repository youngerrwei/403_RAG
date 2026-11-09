# AGENTS.md — AI Agent 代码修改规范指南

> 本文档面向 AI Agent 和开发者，作为修改本项目代码时的规范参考。

---

## 1. 项目架构概览

### 核心文件职责

| 文件 | 职责 |
|------|------|
| `src/lab_rag/rag_agent.py` | 核心 RAG 流程：路由判断、查询改写、向量检索、重排序、LLM 生成 |
| `src/lab_rag/ingest.py` | 知识入库：文档加载、文本清洗、分块切分、向量化、写入 Qdrant |
| `src/lab_rag/web_app.py` | Flask 服务：用户认证、SSE 流式接口、对话历史管理 |
| `src/lab_rag/logger.py` | 统一日志模块：提供 `get_logger()` 接口，支持控制台+文件双输出、按日期自动轮转 |
| `src/lab_rag/create_user.py` | 用户创建脚本：交互式创建/原子更新用户凭据（PBKDF2 加密） |
| `requirements/rag.txt` | RAG 主环境直接依赖清单，由 `scripts/setup_env.sh --rag` 安装与校验 |
| `scripts/auto_ingest.sh` | 入库管理脚本：支持增量入库、全量入库、集合销毁 |
| `scripts/start_rag.sh` | 服务启动管理脚本：启动前预检（.env、模型路径、Qdrant、目录验证）、环境变量全量加载、vLLM/web_app 启动管理、通过 /api/health 验证就绪性 |
| `src/lab_rag/templates/index.html` | 对话前端页面 |
| `src/lab_rag/templates/login.html` | 登录页面 |
| `.env` | 所有配置项（模型路径、Qdrant 连接、检索参数等） |

### 文件间依赖关系

```
src/lab_rag/web_app.py ──导入并调用──▶ src/lab_rag/rag_agent.py
     │                        │
     │                        ▼
     │                   Qdrant 集合 ◀── src/lab_rag/ingest.py（独立运行，共享集合）
     │                                        ▲
     │                                        │
     │          scripts/auto_ingest.sh ────调用────────┘
     │                │
     ▼                ▼
  src/lab_rag/logger.py ◀──── 被所有 Python 主模块导入（共享日志基础设施）
```

- `src/lab_rag/web_app.py` 导入并调用 `src/lab_rag/rag_agent.py` 中的核心函数
- `src/lab_rag/ingest.py` 独立运行，与 `src/lab_rag/rag_agent.py` 共享 Qdrant 集合
- `scripts/auto_ingest.sh` 调用 `src/lab_rag/ingest.py` 执行入库操作
- `src/lab_rag/logger.py` 被 `src/lab_rag/web_app.py`、`src/lab_rag/rag_agent.py`、`src/lab_rag/ingest.py` 共同导入，提供统一日志能力

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
- 如必须引入，需同步更新 `requirements/rag.txt` 与 README 中的安装说明

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
bash scripts/auto_ingest.sh --destroy --force && bash scripts/auto_ingest.sh --full
```

---

## 7. SSE 数据协议

### 当前事件类型

```jsonc
// 流开始（src/lab_rag/web_app.py 注入）
data: {"type": "start", "question": "..."}

// 元数据：路由决策结果（src/lab_rag/rag_agent.py 发出）
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
data: {"type": "final", "content": "...", "route": "...", "coverage": {...}, "citations": [...], "file_result": {...}}

// Agent 模式扩展事件（仅 use_agent=true 时产生）
// route_direct：编排器 LLM 路由决策（不是模型输出，不伪造 Thought）；内容型路由（rag_search）
//   也仅发出此决策事件，随后同样进入 ReAct 循环，由模型自主调用内容工具，不再确定性直调短路
data: {"type": "route_direct", "step": 1, "tool": "rag_qa", "arg": "...", "route": "rag_search", "reason": "..."}
// step_llm：仅承载模型真实输出
data: {"type": "step_llm", "step": 1, "content": "Thought/Action/Answer 原文"}
// step_tool：observation 为完整原文；tool_call_id 串联一次调用的可追溯链
data: {"type": "step_tool", "step": 1, "tool": "rag_qa", "arg": "...", "observation": "...", "tool_call_id": "tc-1", "elapsed_s": 0.4, "ok": true}
// final 事件额外携带 agent_metrics（前端可选读取，未知字段忽略）
// content_grounded / catalog_grounded 表示是否已获得可作答的正文/目录依据；
// 已无 deterministic_route 字段（内容型问题不再走确定性直调短路）
data: {"type": "final", "content": "...", "agent_metrics": {"llm_calls": 3, "tool_calls": 2, "idle_steps": 0, "context_chars_peak": 4258, "converged": true, "timed_out": false, "content_grounded": true, "catalog_grounded": false, "elapsed_s": 1.2}}

// 心跳包（src/lab_rag/web_app.py 注入，保持连接活跃）
data: {"type": "heartbeat"}

// 错误通知：统一使用 "message" 字段（前端仍兼容读取 content）
data: {"type": "error", "message": "..."}

// 流结束信号（固定格式，不是 JSON）
data: [DONE]
```

### 修改规则

- 修改 SSE 协议时**必须同步更新**前端 `src/lab_rag/templates/index.html` 的解析逻辑
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
| `preflight_check()` 行为 | 启动前预检中 Qdrant 不可达、必要集合缺失或文档目录为空仅发出警告（不阻止启动）；模型路径不存在则中止启动 |
| `/api/health` 状态含义 | `"degraded"` 表示部分组件不可用但系统仍可接受请求；`"error"` 表示关键服务不可用；修改状态判断逻辑时需同步更新 `scripts/start_rag.sh` 中的 JSON 解析逻辑 |

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
| `VLLM_GPU_UTIL` | `0.85` | vLLM 显存占用比例；单卡（如 RTX 3090）可上调至 ~0.90 |
| `VLLM_MAX_MODEL_LEN` | `16384` | 上下文窗口上限；Qwen3-8B 原生 40960、24GB 单卡实测 KV 池约 22752，16384 安全且留并发余量；单卡同栈省显存可下调（3090 模板 8192）。**不要靠换更小的模型来换窗口**——8B 的 ReAct/路由/中文问答质量对 RAG 有效性关键 |
| `VLLM_ENABLE_PREFIX_CACHING` | `true` | 前缀缓存；多步 Agent 复用系统提示词前缀，是连锁场景的关键优化 |
| `VLLM_QUANTIZATION` | `（空）` | 模型量化；单卡 3090 首选 `bitsandbytes`（vLLM 用本地 fp16 实时量化 4-bit，权重 16GB→约 5.7GB，免下载）；亦支持 awq/gptq/fp8 预量化权重（需先获取） |
| `VLLM_DTYPE` | `auto` | 计算精度（auto=fp16 / half / bfloat16） |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | 张量并行度；单卡固定为 1 |
| `VLLM_MAX_NUM_SEQS` | `（空）` | 并发序列上限；单卡可下调以省 KV 显存（空=vLLM 默认） |
| `VLLM_ENABLE_CHUNKED_PREFILL` | `false` | 分块预填充，提升单卡混合预填/解码吞吐 |
| `FILE_SEARCH_LIMIT` | `200` | 文件搜索结果上限 |
| `ENABLE_MARKDOWN_STRUCTURE_GUARD` | `true` | 标题/代码块/表格原子块保护 |
| `ENABLE_PATH_CONTEXT_PREFIX` | `true` | 向量化前缀注入小组/设备目录层级 |
| `CHUNK_FILTER_SPECIAL_CHAR_RATIO` | `0.25` | 质量过滤特殊字符占比上限 |
| `ENABLE_TABLE_FILTER_EXEMPTION` | `true` | 表格/代码块豁免长度与字符占比过滤 |
| `INGEST_MAX_FAILED_FILES` | `0` | 允许的文档加载失败数，超出即中止入库 |
| `QDRANT_INDEX_FAIL_FAST` | `true` | payload 索引缺失时中止入库 |
| `ENABLE_CATALOG_SNAPSHOT_CACHE` | `true` | 目录枚举共用单次遍历快照，消除重复全盘遍历 |
| `CATALOG_SNAPSHOT_TTL` | `300` | 目录树快照缓存 TTL（秒）；键含索引版本；过期后 stale-while-revalidate（返回旧值 + 后台刷新），不阻塞请求 |
| `ENABLE_CATALOG_SNAPSHOT_DISK_CACHE` | `true` | 快照 gzip 落本地 `data/`，进程重启免于重新遍历 CIFS 共享盘（约 20s+） |
| `CATALOG_SNAPSHOT_DISK_MAX_AGE` | `86400` | 磁盘快照最大滞留秒数，超过视为过期并重建 |
| `CATALOG_EXCLUDE_HIDDEN_DIRS` | `true` | 排除 `.svn`/`.git`/隐藏目录，避免噪声路径误命中 |
| `AGENT_MAX_STEPS` | `6` | ReAct 单一编排器的多轮检索步数预算（流式/非流式共用）；单次「工具调用」计 1 步，最终作答不占步数，故 6 步可支撑 5 次检索+1 次作答，覆盖更长链式/多实体任务 |
| `AGENT_TOTAL_TIMEOUT_S` | `240` | Agent 轮次总耗时预算（秒），超时优雅收敛；与 web_app 300s 上游超时错开 |
| `AGENT_FALLBACK_SNIPPET_CHARS` | `300` | 预算耗尽兜底附带的部分依据摘要长度；0 表示不附带 |
| `AGENT_OBSERVATION_MAX_CHARS` | `2000` | 注入 LLM 的单条工具结果字符上限（SSE 事件仍传完整原文） |
| `AGENT_KEEP_RECENT_OBSERVATIONS` | `2` | 保留完整工具结果的近期轮数，更早的折叠为占位，避免上下文随步数膨胀 |
| `AGENT_FINAL_CONVERGE_ROUND` | `true` | 预算用尽后给一次收敛轮（不执行新工具）；关闭可省一次 LLM 调用，但收敛率下降 |
| `AGENT_MAX_IDLE_STEPS` | `2` | 空转轮上限；空转不占工具步数预算，另有 LLM 硬上限防死循环 |
| `AGENT_REPEAT_ACTION_GUARD` | `true` | 抑制「同工具+完全相同参数」重复调用，避免多轮搜索空转 |
| `AGENT_REUSE_ROUTE_HINT` | `true` | 多步连锁场景优化：agent 循环内 rag_qa 子问题复用内容路由，跳过冗余规则/LLM 重路由（模型已选内容工具即等价于内容路由）；关闭可恢复历史逐轮子路由行为 |
| `AGENT_HISTORY_MAX_TURNS` | `20` | Agent 初始消息注入历史的总轮次护栏（超过的更旧轮次参与字符预算折叠） |
| `AGENT_HISTORY_MAX_CHARS` | `3000` | Agent 历史注入的字符预算；超出时最旧轮次折叠为紧凑占位，支持更长对话上下文而不撑爆窗口 |
| `AGENT_CONTEXT_MAX_CHARS` | `16000` | Agent 多轮总上下文**字符**预算护栏；多轮 RAG 结果/长历史累积超长时折叠最旧消息，对抗「RAG 结果爆炸」 |
| `AGENT_CONTEXT_MAX_TOKENS` | `0` | Agent 输入侧 **token** 预算护栏（防 400 上下文超长的关键）；`0`=自动推导 `VLLM_MAX_MODEL_LEN - RESPONSE_MAX_TOKENS - AGENT_CONTEXT_SAFETY_TOKENS`，离线精简 cfg 不含窗口参数时不启用 |
| `AGENT_CONTEXT_SAFETY_TOKENS` | `256` | token 估算（tiktoken）与 Qwen 分词器 / chat template 的误差余量 |
| `READ_FILE_MAX_CHARS` | `6000` | `read_file` 工具单次读取的字符预算（读取文件/目录拼接均受此上限约束） |
| `ENABLE_ADAPTIVE_LEXICAL_RECALL` | `true` | 长纯数字词元（日期戳/编号）在首轮召回身份字段零命中时，自动启用本地词边界匹配补召回（按词元形态判定，无用例语义） |
| `ADAPTIVE_LEXICAL_MIN_DIGITS` | `6` | 触发自适应兜底的纯数字词元最小长度 |
| `ADAPTIVE_LEXICAL_TOP_K` | `16` | 自适应兜底补充候选数上限 |
| `CATALOG_MAX_TRAVERSAL` | `200000` | 目录树快照遍历上限（目录数）；默认覆盖完整语料目录树，避免靠后目录（如 `设备操作指南/VLP小组`）缺失导致 file_list/hybrid 目录枚举错配 |
| `MAX_HYBRID_QUERIES` | `8` | hybrid 目录驱动查询去重后的最大路数；完整清单仍由答案区块提供 |
| `MAX_DENSE_QUERIES` | `4` | Dense 查询最大路数（hybrid 由 `MAX_HYBRID_QUERIES` 单独限流） |
| `MAX_HYDE_QUERIES` | `2` | HyDE 假设文档最大路数 |
| `MAX_SPARSE_CONDITIONS` | `30` | MatchText should 条件预算 |
| `ENABLE_HISTORY_AWARE_REWRITE` | `true` | 改写阶段使用对话历史消解代词 |
| `HISTORY_SOFT_TURN_LIMIT` | `16` | 对话历史软轮次上限；超过该轮数的最旧轮次在 token 截断前先折叠压缩，保留意图线索 |
| `HISTORY_COMPACT_ENABLED` | `true` | 历史超预算时折叠旧轮而非直接丢弃（关闭则退化为旧的硬截断行为） |
| `RERANKER_RELATIVE_RATIO` | `0` | 重排相对 Top-1 比例阈值（0=关闭） |
| `RERANKER_MIN_KEEP` | `0` | 重排最少保留条数（0=关闭） |
| `ENABLE_ANSWERABILITY_GATE` | `true` | 关键词零证据 + 低分时判为知识库无答案，避免假引用 |
| `ANSWERABILITY_TOP_SCORE_FLOOR` | `0.90` | 拒答闸门的 Top-1 分数下限（双证据策略下上调） |
| `ANSWERABILITY_MIN_KEYWORD_HITS` | `1` | 判定可答所需的关键词最少命中数 |
| `ANSWERABILITY_MIN_OVERLAP_HITS` | `2` | 原始问题与召回文档身份字段的中文 2-gram 最少重合数 |
| `MAX_SPARSE_TERMS` | `12` | MatchText 检索词元预算（纯英文型号修复后放大） |
| `PARENT_EXPAND_MAX_CHARS` | `9000` | 父块展开累计预算，与最终上下文预算解耦 |
| `PARENT_MAX_SINGLE_CHARS` | `1500` | 单个父块最大字符数，按行边界截断 |
| `PARENT_MAX_PER_SOURCE` | `3` | 同一来源文件的父块配额，避免单手册垄断候选 |
| `LEXICON_PATH` | `config/retrieval_lexicon.json` | 结构化检索规则词典（词表/答案模板/表格恢复模板） |
| `CORPUS_LEXICON_PATH` | `data/corpus_lexicon.json` | 由语料 `rel_path` 离线生成的派生词典 |
| `ENABLE_CORPUS_LEXICON` | `true` | 启用语料派生的小组别名；关闭则回退静态别名 |
| `ENABLE_NOTATION_SPECIALIZATION` | `true` | 变量格式表特判能力开关，关闭后走通用检索+LLM |
| `ENABLE_EVIDENCE_RERANK` | `true` | 证据加权稳定重排，仅影响引用顺序 |
| `EVIDENCE_RERANK_WEIGHT` | `0.12` | 型号/路径精确命中在重排并列时的加分权重 |
| `EVIDENCE_PRIOR_WEIGHT` | `0.12` | 强证据先验每命中一个（日期/文档标题词元/型号）的加性加分 |
| `EVIDENCE_PRIOR_CAP` | `0.24` | 强证据先验加分上限，避免单文档被过度提升 |
| `TABLE_SEMANTIC_MAX_REPLACEMENT_RATIO` | `0.15` | 表格语义转写跳过 U+FFFD 占比超阈值的行，避免放大 OCR 噪声 |
| `RETRIEVAL_INDEX_VERSION` | `2` | 缓存键索引版本；入库逻辑变更后需递增（v3.2.0 因 page_content 变更已递增） |
| `COVERAGE_HIGH_THRESHOLD` | `0.7` | 高覆盖度阈值 |
| `COVERAGE_MEDIUM_THRESHOLD` | `0.5` | 中覆盖度阈值 |
| `COVERAGE_LOW_THRESHOLD` | `0.3` | 低覆盖度阈值 |
| `RERANK_QUERY_MODE` | `both` | 原始问题+改写问题共同重排 |
| `RERANKER_ALLOW_LOW_SCORE_FALLBACK` | `false` | 全部低分时是否保底返回 1 条 |
| `NOTATION_MATCH_BONUS` | `0.15` | 变量格式表命中块的加分上限 |
| `ENABLE_CONTEXT_NOISE_FILTER` | `true` | 上下文过滤超长链接与 OCR 噪声 |

**注意：** 不要使用 HuggingFace 在线路径（如 `BAAI/bge-m3`），统一使用本地模型路径；新增配置键必须同步 `.env.example`、两侧 `load_config()` 与本表。

---

## 10. API 端点共识

### GET /api/health（无认证）

| 字段 | 值/格式 | 说明 |
|------|--------|------|
| 路由 | `/api/health` | 固定路由，无需认证 |
| 方法 | `GET` | 只读操作 |
| `status` | `"ok"` / `"degraded"` / `"error"` | 系统整体状态 |
| `components` | `{"embedding": bool, "reranker": bool, "qdrant": bool, "llm": bool}` | 各组件状态；`qdrant=true` 还要求子块与父块集合都存在 |
| HTTP 200 | status="ok" 或 "degraded" | 系统可用 |
| HTTP 503 | status="error" | 系统不可用 |

**修改规则：**
- 该端点为系统稳定 API，路由和响应格式不可随意变更
- 扩展检查项时向 `components` 对象中添加新键值对，不删除已有键
- 不要改变 `status` 的三个可能值定义
- 修改此端点的响应格式时，**必须同步更新** `scripts/start_rag.sh` 中的 JSON 解析逻辑
