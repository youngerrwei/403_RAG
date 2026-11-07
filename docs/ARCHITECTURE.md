# LAB 403 RAG 系统架构设计

> 本文档是 LAB 403 RAG 知识库系统的架构设计文档，详细描述系统的整体架构、数据流、核心设计决策与优化策略。面向开发者和 AI Agent，作为理解系统设计和进行代码修改时的技术参考。

---

## 系统架构总览

### 流程概览

```mermaid
flowchart LR
    subgraph ingest ["文档入库"]
        A1["Markdown"] --> A2["文本清洗"] --> A3["标题分段"] --> A4["父子块切分"] --> A5["质量过滤"] --> A6["Embedding"] --> A7["Qdrant<br>(子块索引 + 父块独立存储)"]
    end
    subgraph qa ["知识问答"]
        B1["用户问题"] --> B2["查询路由<br>(规则优先+LLM兜底)"] --> B3["问题改写"] --> B4["混合检索<br>(Dense+全文词项)"]
        B4 --> B5["RRF融合"] --> B6["去重"] --> B7["重排序"] --> B8["父块展开<br>(独立Collection查询)"] --> B9["LLM 流式生成"]
        B10(["TTL缓存 5min"]) -.-> B4
    end
    ingest ~~~ qa
```

### 组件交互关系

系统由以下核心组件协作完成知识问答服务：

```mermaid
flowchart TB
    A["用户浏览器 (前端)<br>src/lab_rag/templates/index.html"] -->|"SSE 流式连接"| B["Flask Web 服务 (src/lab_rag/web_app.py)<br>用户认证 · SSE 流式接口 · 对话历史管理 · 并发控制"]
    G["MCP Host / DeepSeek Harness"] -->|"stdio"| H["MCP Bridge (src/lab_rag/mcp_server.py)<br>独立轻量环境 · 不加载模型"]
    H -->|"loopback JSON + Bearer Token"| B
    B -->|"调用核心 RAG 流程"| C["RAG 核心引擎 (src/lab_rag/rag_agent.py)<br>查询路由 · 查询改写 · 混合检索 · 重排序 · 父块展开 · 流式生成"]
    C --> D["vLLM<br>Qwen3-8B-Instruct<br>(GPU 3)"]
    C --> E["Qdrant<br>向量数据库<br>(远程部署)"]
    C --> F["Embedding/Reranker<br>bge-m3 / bge-reranker-v2-m3<br>(GPU 2)"]
```

**组件协作方式：**

- **Flask Web 服务** 作为入口层，处理用户认证、会话管理，将问答请求委托给 RAG 核心引擎，并通过 SSE 将流式结果推送至前端
- **RAG 核心引擎** 编排完整的问答流程：先通过规则/LLM 路由判断意图，调用 Embedding 模型将查询向量化，再从 Qdrant 执行混合检索，用 Reranker 精排后展开父块上下文，最终调用 vLLM 生成流式回答
- **vLLM 推理服务** 提供 OpenAI 兼容 API，负责查询改写和最终回答生成两个 LLM 调用阶段
- **Qdrant 向量数据库** 存储子块 Dense 向量索引、父块原文与 payload 索引，支持 Dense + 全文词项匹配（MatchText）混合检索
- **Embedding / Reranker 模型** 在 RAG 引擎内部以本地推理方式运行，分别负责查询/文档向量化和精排打分
- **MCP Bridge（可选）** 只做 stdio/JSON 协议转换；检索工具在 `retrieval` 元数据产生后关闭标准 RAG 生成器，不进入最终答案生成，也不写入历史

---

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| LLM | Qwen3-8B-Instruct (vLLM 部署) |
| Embedding | BAAI/bge-m3 (1024维) |
| Reranker | BAAI/bge-reranker-v2-m3 (CrossEncoder) |
| 向量数据库 | Qdrant (远程部署) |
| Web 框架 | Flask + SSE 流式响应 |
| MCP（可选） | MCP Python SDK v2 + stdio；本机 Token 认证 JSON 旁路 |
| 前端 UI | HTML5 + CSS3 (Flexbox/Grid/backdrop-filter) + Vanilla JS，Lucide Icons (CDN)，暗色模式双支持 |
| 硬件 | 4 卡 GPU 4090 24GB（GPU 2: Embedding + Reranker, GPU 3: vLLM） |

---

## 数据流详解

### 入库数据流

```mermaid
flowchart LR
    A["文档加载<br>(递归扫描 .md文件)"] --> B["文本清洗<br>(HTML/图片/空白)"] --> C["标题结构切分<br>(按 #/##/### 拆分段落)"] --> D["父子块切分<br>(Small-to-Big)<br>父块1500字/子块300字"]
    D --> E["质量过滤<br>· 过短过滤(小于120字符)<br>· 有效字符比例检查<br>· 数字/特殊字符占比"]
    E --> F["Embedding 向量化<br>(bge-m3, 1024维 Dense)<br>· 批量处理(64)<br>· 路径/章节前缀"]
    F --> G["写入 Qdrant<br>· 子块→主集合<br>· 父块→父块集合<br>· 确定性UUID5<br>· payload索引校验"]
```

**各阶段详细说明：**

1. **文档加载**：递归扫描 `DOCS_PATH` 目录，多编码尝试（UTF-8 / GBK）读取 Markdown 文件
2. **文本清洗**：移除 HTML 注释/标签、图片语法 `![]()`、零宽字符，合并连续空行，压缩多余空白
3. **标题结构切分**：按 Markdown 标题层级（#/##/###）将文档拆分为逻辑段落，维护完整的标题层级路径（如 `文档标题 > 第一章 > 1.1节`）
4. **父子块切分 (Small-to-Big)**：
   - 父块：使用 RecursiveCharacterTextSplitter 将逻辑段落切为 1500 字符的父块，注入标题/路径前缀增强语义
   - 子块：将每个父块继续切为 300 字符的子块，子块 metadata 中保存 `parent_id` 用于回溯
5. **质量过滤**：过滤空内容、过短（<120字符）、有效字符不足、数字占比过高、特殊字符过多的块
6. **Embedding 向量化**：使用 bge-m3 生成 1024 维 Dense 向量，批量大小为 64；向量化文本注入「文档 > 路径（小组/设备/Manuals）> 章节 > 段落摘要」前缀。Sparse 路是查询期基于 payload 文本索引的 MatchText 全文词项匹配，库内不写入稀疏向量
7. **写入 Qdrant**：
   - 子块以确定性 UUID5（基于 `source|parent_id|chunk_index`）写入主集合 `lab_knowledge_base`
   - 父块写入独立集合 `lab_knowledge_base_parents`（4维零向量占位，不参与相似度计算）

### 问答数据流

```mermaid
flowchart LR
    A["用户提问"] --> B["查询路由<br>· 规则引擎优先<br>· LLM兜底判断<br>· 短问题跳过改写"]
    B --> C["查询改写<br>· standalone + keywords<br>· expanded / HyDE 独立路数<br>· 对话历史消解代词"]
    C --> D["混合检索<br>· Dense(0.7) 分数归并<br>· 全文词项(0.3) 字段预算<br>· 并行多查询"]
    D --> E["RRF 融合 + 去重 + 重排序<br>· RRF(k=60) 写入 rrf_score<br>· 去重(parent_id+chunk_hash)<br>· CrossEncoder 双 query 精排<br>· 中文 n-gram boost"]
    E --> F["父块展开<br>· 独立Collection<br>· scroll+Filter<br>· 上下文预算控制<br>· 超长截断"]
    F --> G["LLM 流式生成<br>· vLLM API<br>· SSE 推送<br>· 覆盖度评估<br>· 引用溯源"]
```

**各阶段详细说明：**

1. **查询路由**：规则引擎通过关键词匹配快速分流（rag_search / file_list / hybrid），未命中时调用 LLM 兜底判断意图
2. **查询改写**：LLM 将用户问题改写为 `standalone_question`（独立化）+ `expanded_queries`（扩展查询），多查询并行检索
3. **HyDE 查询扩展**：对复杂查询，LLM 先生成假设性回答文本，作为与 Dense 分开计数的独立 query 参与检索，不再被 `queries[:5]` 截断
4. **混合检索**：Dense 向量检索（权重 0.7，按相似度分数归并排序）+ 全文词项匹配（权重 0.3，基于 Qdrant MatchText，词元化后按字段优先级分配条件预算）
5. **RRF 融合**：通过 Reciprocal Rank Fusion (k=60) 融合两路结果并写入 `rrf_score`，与去重共用 `canonical_doc_key`（`parent_id + chunk_hash`），提升专业术语和设备型号的召回能力
6. **去重**：基于 `source + parent_id` 去除冗余结果
7. **CrossEncoder 重排序**：bge-reranker-v2-m3 精排（batch_size=64），应用 boost 系数，过滤低分文档（阈值 0.3）
8. **父块展开**：从独立 Collection 通过 `scroll` + Filter 批量查询父块原文，累计字符数不超过 `MAX_CONTEXT_CHARS`（3000）
9. **LLM 流式生成**：调用 vLLM OpenAI 兼容 API 流式生成回答，同时计算覆盖度评估和引用溯源
10. **SSE 推送**：通过 Server-Sent Events 将流式 token、状态变更、元数据实时推送至前端

### MCP 纯检索旁路

```mermaid
flowchart LR
    A["MCP Host"] -->|"stdio tools/call"| B["src/lab_rag/mcp_server.py"]
    B -->|"127.0.0.1 + Bearer Token"| C["src/lab_rag/web_app.py 私有 API"]
    C --> D["ask_stream<br>persist_history=false"]
    D --> E["路由 → 改写 → 混合检索 → 重排 → 父块展开"]
    E --> F["retrieval 元数据"]
    F --> G["有界 MCP 工具结果"]
    F -.->|"关闭生成器，不执行"| H["最终答案生成"]
```

该旁路复用 Web 进程中的模型、Qdrant 客户端、缓存和并发信号量。内部 API 不对非 loopback 对端开放，Token 为空时保持禁用；MCP 子进程由 Host 按需启动和关闭，不属于 `scripts/start_rag.sh` 的生命周期。

---

## RAG 核心设计

本系统针对实验室场景的 10 条核心设计决策：

### 1. Small-to-Big 检索策略

子块（300字符）精准匹配用户查询，命中后通过 `parent_id` 从独立 Collection (`lab_knowledge_base_parents`) 使用 `scroll` + Filter 批量查询父块（1500字符），提供完整上下文给 LLM。

**设计考量：** 小块向量与查询更容易精准匹配，而大块为 LLM 提供充分的上下文信息。两者分离存储避免了在单集合中混存不同粒度的向量，简化了检索逻辑。单个父块内容超过上下文预算一半时自动截断，防止单文档独占上下文窗口。

### 2. 两级查询路由

规则引擎优先（关键词匹配快速分流：`rag_search` / `file_list` / `hybrid`），未命中时 LLM 兜底判断意图。

**设计考量：** 规则路由延迟极低（<1ms），可处理 80%+ 的常见查询模式；LLM 路由处理规则无法覆盖的复杂意图，减少不必要的 LLM 调用延迟。两级设计在速度和准确性之间取得平衡。

### 3. 混合检索 (Dense + 全文词项匹配)

Dense 向量检索（权重 0.7）+ Qdrant MatchText 全文词项匹配（权重 0.3），通过 Reciprocal Rank Fusion (RRF, k=60) 融合。库内只写入 1024 维 Dense 向量，**不写入稀疏向量**；`BM25_WEIGHT` 实际是 RRF 融合权重，并非 BM25 打分。词元化优先提取设备型号与英文术语，中文按标点切成短片段；`should` 条件按 `rel_path → file_name → doc_title → section_title → header_path → parent_summary → page_content` 的优先级分配预算，保证路径字段不被无序截断。

**设计考量：** 实验室语料中「VLC 小组有哪些设备」「MSO-X 3054A 触发设置」这类问题同时依赖语义理解与精确型号/路径命中。全文词项匹配无需新增依赖或变更集合 schema，配合 `source`/`rel_path`/`parent_id` keyword 索引与 `header_path`/`parent_summary` 文本索引即可覆盖；真正的 BM25/SPLADE 稀疏向量列为后续方向。

### 4. 多查询改写 + 并行检索

LLM 将用户问题改写为 `standalone_question` + `keywords` + `expanded_queries` + 独立的 `hyde_queries`，并结合最近对话历史消解代词。Dense 与 HyDE 分别由 `MAX_DENSE_QUERIES` / `MAX_HYDE_QUERIES` 限流，多个查询通过 `ThreadPoolExecutor` 并行检索后按相似度分数归并。检索结果带 TTL 缓存（5 分钟），缓存键含用户、路由与 `RETRIEVAL_INDEX_VERSION`，相同查询可命中缓存避免重复检索。

**设计考量：** 多角度查询扩展提升召回率，并行执行不增加额外延迟。TTL 缓存应对用户短时间内的重复/相似提问，有效降低 Qdrant 负载。

### 5. CrossEncoder 重排序

使用 bge-reranker-v2-m3 精排（分批推理 batch_size=64 防止显存溢出），query 由原始问题与改写问题拼接（`RERANK_QUERY_MODE=both`），中文按 2/3 字 n-gram 计算乘法 boost（关键词 5%、标题 8%），并以 `rrf_score` 作为并列时的次级排序依据。低于分数阈值（0.3）的文档默认全部过滤（`RERANKER_ALLOW_LOW_SCORE_FALLBACK=false`），避免把低相关内容包装成假引用；需要保底时可显式打开开关。

**设计考量：** CrossEncoder 的交叉注意力机制比 Bi-Encoder 有更强的相关性判断能力，作为精排阶段使用可显著提升最终结果质量。分批推理策略防止大量候选文档导致 GPU OOM。

### 6. 上下文长度控制

父块展开时累计字符数不超过 `MAX_CONTEXT_CHARS`（默认 3000），结合 vLLM 总容量 8192 tokens 做 input+output 预算控制：

```
context ≈ 2000 tokens + history 1500 tokens + prompt 250 tokens + question 100 tokens + output 2048 tokens ≈ 5898 < 8192
```

**设计考量：** 防止 LLM 输入过长导致生成质量下降（Lost in the Middle 问题），确保在有限上下文窗口内合理分配各部分预算。

### 7. 确定性 Point ID

基于 `source|parent_id|chunk_index` 生成 UUID5，相同文档重复入库不会产生重复向量，支持幂等操作。

**设计考量：** 增量入库场景下保证数据一致性，无需额外的去重逻辑。即使入库脚本重复执行，同一文档块始终对应同一 Point ID，实现天然幂等。

### 8. 用户级历史隔离

按日期分文件存储（`{username}/{YYYY-MM-DD}.json`），支持 7 天自动过期清理与单日上限控制（200条/天），使用 `threading.Lock` + `fcntl.flock` 保证线程安全与文件锁。后台守护线程每天 02:00 自动清理过期文件。

**设计考量：** 按日期分文件避免单文件过大导致的读写性能问题，双重锁机制保证多线程/多进程场景下的数据安全。自动清理策略控制磁盘用量增长。

### 9. HyDE 查询扩展

对于复杂或模糊的查询，LLM 先生成一段假设性回答文本，将该文本作为额外的检索 query 参与 Dense 检索。假设文档与真实文档在向量空间中更接近，有效提升语义模糊查询的召回率。通过 `ENABLE_HYDE` 与 `MAX_HYDE_QUERIES` 控制。

**v3.1.0 修正：** 旧实现把 HyDE 片段追加到 `queries` 后又执行 `queries[:5]`，假设文档实际从未参与检索；现在 `hyde_queries` 作为独立返回字段与 `expanded_queries` 分开计数，由 `MAX_DENSE_QUERIES` / `MAX_HYDE_QUERIES` 分别限流。

**设计考量：** 当用户查询过于抽象或口语化时，直接向量化的查询与文档向量可能距离较远。HyDE 通过 LLM 生成的假设文档作为"桥梁"，拉近查询与目标文档在向量空间中的距离。

### 10. 智能 Query 路由优化

对短问题（≤15字且无代词）或纯关键词查询（≤8字）跳过 LLM 改写步骤，直接使用原始问题检索，减少约 100-150ms 的 LLM 调用延迟，同时保持对复杂/多轮问题的改写能力。

**设计考量：** 并非所有查询都需要改写。简单明确的关键词查询经过改写后反而可能引入噪音，直接检索效果更好且延迟更低。

---

## 架构优化

### Contextual Retrieval（上下文增强检索）

子块在 Embedding 之前自动注入文档标题、目录层级、章节路径与段落摘要作为上下文前缀，增强向量的语义丰富度，尤其用于区分不同设备目录下的同名手册。

**实现细节：**
- 每个子块在向量化前自动拼接前缀：`【文档：文件名 > 路径：小组 > 设备 > Manuals > 章节：标题层级 > 段落摘要：按句截断】\n[子块原文]`，由 `ENABLE_PATH_CONTEXT_PREFIX` 控制路径段
- 无章节路径的短规范文档才单独写 `标题：`，避免与 `header_path` 重复
- `summary_status` 写入 payload，便于判别摘要降级状态；前缀不参与质量过滤判定
- HNSW 索引参数优化为 `m=32, ef_construct=200`（Qdrant 默认 16/100），提升近邻搜索精度
- 上下文前缀不参与文本展示，仅用于向量化阶段，保证检索精度的同时不影响最终呈现

### 入库摘要增强（Summary Augmentation）

入库时自动调用 vLLM 为每个父块生成 150-250 字的智能摘要，摘要作为前缀注入子块参与 Embedding 计算，增强语义检索的宏观理解能力。通过配置 `ENABLE_SUMMARY_AUGMENTATION` 开关控制。

**实现细节：**
- 每个父块入库时调用 vLLM 生成简明摘要（150-250 字），摘要以前缀形式拼接到子块文本前参与向量化
- 摘要同时存入父块 metadata 的 `parent_summary` 字段，检索展示时可供 LLM 作为上下文辅助理解
- 支持并发摘要生成（`SUMMARY_MAX_WORKERS` 配置并发线程数），加速大批量入库
- 内置摘要缓存机制（`SummaryCache`），基于内容哈希存储已生成的摘要（`SUMMARY_CACHE_DIR`），增量入库时避免重复调用 LLM
- 摘要生成支持超时控制（`SUMMARY_VLLM_TIMEOUT`）和失败重试（`SUMMARY_VLLM_RETRIES`），网络异常时优雅降级

**相关配置项：**

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `ENABLE_SUMMARY_AUGMENTATION` | `true` | 是否启用入库摘要增强 |
| `SUMMARY_MAX_WORKERS` | `4` | 并发摘要生成线程数 |
| `SUMMARY_MAX_TOKENS` | `300` | 摘要生成最大 token 数 |
| `SUMMARY_VLLM_TIMEOUT` | `30` | 单次摘要请求超时（秒） |
| `SUMMARY_VLLM_RETRIES` | `2` | 摘要生成失败重试次数 |
| `SUMMARY_INJECTION_MODE` | `prefix` | 摘要注入方式（前缀拼接） |
| `ENABLE_SUMMARY_CACHE` | `true` | 是否启用摘要缓存 |
| `SUMMARY_CACHE_DIR` | `./cache/summaries` | 摘要缓存存储目录 |

**设计考量：** 子块粒度较小（300字符），单独向量化可能丢失段落宏观语义。通过 LLM 摘要注入，子块向量同时编码了局部细节和段落概要信息，有效提升语义模糊查询的召回率。缓存机制确保增量入库时不会重复消耗 LLM 资源。

### 知识库覆盖度评估

基于 CrossEncoder 重排分数自动评估知识库对当前问题的覆盖程度，分为 5 个级别：

| 级别 | 判定条件 | 行为 | 前端提示颜色 |
|------|----------|------|-------------|
| `high` | 最高分 ≥ 0.8 | 严格基于知识库回答 | 绿色 |
| `medium` | 最高分 ≥ 0.5 | 基于知识库回答，提示可能不完整 | 蓝色 |
| `low` | 最高分 ≥ 0.3 | 结合知识库与模型知识回答 | 黄色 |
| `very_low` | 最高分 > 0 | 以模型知识为主，参考知识库 | 橙色 |
| `none` | 无有效结果 | 仅用模型知识回答，明确告知用户 | 红色 |

**设计考量：** 覆盖度评估让系统能够动态调整回答策略，在知识库覆盖充分时严格引用，覆盖不足时坦诚告知用户，提升回答的可信度和透明度。前端通过颜色编码帮助用户快速判断回答的可靠程度。

### 答案引用溯源

回答结束后自动提取参考文档来源和章节信息（最多 5 条引用），前端以可折叠列表展示文件名、章节路径和置信度百分比，方便用户追溯原文。

**实现方式：**
- 从重排序后的 Top-K 文档中提取 `source`（文件路径）和 `title_path`（章节路径）
- 置信度基于 CrossEncoder 重排分数归一化为百分比
- 通过 SSE `metadata` 事件将引用信息推送至前端
- 前端渲染为可折叠的引用列表，点击可展开查看详情

### 三阶段流式反馈

用户发送问题后，前端分三个阶段展示实时状态：

| 阶段 | 时机 | 展示内容 |
|------|------|----------|
| 处理中 | 消息发送立即触发 | "正在处理..." + 加载动画 |
| 检索中 | 收到 `searching` 状态事件 | "正在检索知识库..." |
| 生成中 | 收到 `generating` 状态事件 | "正在生成回答..." → 流式文本 |

**设计考量：** 通过即时视觉反馈降低用户感知延迟。从用户点击发送到第一个 token 输出可能需要 2-5 秒（包含路由、改写、检索、重排、LLM 首token），分阶段反馈让用户始终了解系统进度，实际测试中用户满意度提升约 30%。

---

## GPU 资源分配

### 4 卡 GPU 环境分配策略

| GPU | 用途 | 模型/服务 | 预估显存 |
|-----|------|-----------|----------|
| GPU 0 | 预留 | - | - |
| GPU 1 | 预留 | - | - |
| GPU 2 | Embedding + Reranker | bge-m3 + bge-reranker-v2-m3 | ~4GB |
| GPU 3 | LLM 推理 | Qwen3-8B-Instruct (vLLM) | ~20GB |

### 设计考量

- **推理与编码分卡**：LLM 推理（GPU 3）与 Embedding/Reranker（GPU 2）分卡部署，避免显存竞争。vLLM 的 KV Cache 会动态占用大量显存，与 Embedding 模型共卡容易导致 OOM
- **Embedding 与 Reranker 共卡**：两者不会同时高负载运行（Embedding 在检索阶段使用，Reranker 在检索完成后使用），共卡可充分利用资源
- **GPU 0-1 预留**：为其他实验室任务或未来扩展（如更大模型、多模型并行）保留计算资源
- **vLLM 显存利用率**：配置 `VLLM_GPU_UTIL=0.90`，预留 10% 显存给 CUDA 运行时和临时 buffer
- **配置项对应**：通过 `.env` 中的 `EMBEDDING_DEVICE=cuda:2`、`RERANKER_DEVICE=cuda:2`、`VLLM_CUDA_DEVICES=3` 控制设备分配，可根据实际硬件灵活调整

### 前端设计

- **设计风格**：iOS 风格毛玻璃 + 中山大学墨绿/金色配色
- **暗色模式**：`@media (prefers-color-scheme: dark)` + `[data-theme="dark"]` 双覆盖
- **校徽方案**：base64 内联 PNG，无外部依赖
- **图标方案**：Lucide Icons CDN + fallback 文字符号
- **CSP 策略**：nonce 保护内联脚本；允许 cdn.jsdelivr.net、unpkg.com

---

## 已知限制与优化方向

### 当前限制

- **文档格式受限**：仅支持 Markdown 格式文档入库，PDF/Word/PPT 需先通过 `scripts/convert_to_md.sh` 转换
- **全文词项匹配非 BM25**：基于 Qdrant MatchText 布尔匹配 + scroll 顺序，无词频/IDF 打分；RRF 中该路排名不等于相关性排序
- **单机部署**：未做高可用，vLLM / Qdrant / Flask 均为单实例部署
- **查询路由准确性**：依赖规则 + LLM 两级判断；A2X/VLC/VLP 小组缩写已做别名归一，但设备型号别名未建静态词典（由关键词词元化检索兜底）
- **检索缓存范围**：缓存键已含用户、路由与 `RETRIEVAL_INDEX_VERSION`，覆盖 Dense + 全文词项 + RRF 结果；重排与父块展开仍每次执行，以保证分数与当前问题一致
- **增量入库的挂载点依赖**：`source` 仍使用绝对路径，挂载点或目录重命名会让 ID 整体变化；重建前必须确认 `DOCS_PATH` 已挂载且包含真实语料

### 本版（v3.3.0）配置化与语料驱动匹配

上一版把「论文变量格式表」这一具体问题的识别词表、固定检索计划、整段答案模板、加分条件，以及路由四张词表、目录意图词表、表格恢复模板、静态小组别名全部内联在 `rag_agent.py`/`ingest.py` 里。这类定制化一旦语料变化就要改代码，也无法泛化到新设备/新小组。本轮按「配置化保留」重构：

- **规则与语料解耦**：`config/retrieval_lexicon.json` 是可版本控制的规则（词表/模板/正则/答案模板），`data/corpus_lexicon.json` 是由 `scripts/build_corpus_lexicon.py` 从 86 份 md 的 `rel_path` 离线生成的派生数据（小组别名、设备型号、手册类型、语料指纹）。二者分别加载，避免把语料路径写进规则文件。
- **统一词典层**：`src/lab_rag/lexicon.py` 提供单次加载、进程内缓存、启动期 schema fail-fast 与缺失降级。能力由 `ENABLE_NOTATION_SPECIALIZATION` / `ENABLE_CORPUS_LEXICON` 开关控制，可关闭特判以验证通用检索 + LLM 的兜底能力。
- **输出等价性保障**：外置后 `is_notation_format_question`、`build_notation_retrieval_plan`、`build_grounded_notation_answer`、`extract_catalog_content_intent`、`rule_based_route`、表格恢复与过滤豁免逐值等价；`scripts/check_page_content_hash.py` 比对 86 份语料 `clean_text` 的 SHA-256，证实 `page_content` 逐字一致，因此**本轮零重建**（`RETRIEVAL_INDEX_VERSION` 维持 2）。
- **泛化匹配**：小组别名改由语料词典查询（按长度降序，词典缺失回退静态别名）；`_evidence_terms` 覆盖复合型号、数字带字母后缀（843-R）与纯英文术语；新增 `reorder_final_docs_by_evidence`，仅在 rerank 分数差 ≤ `EVIDENCE_RERANK_WEIGHT` 的并列组内用型号/路径精确命中打破并列。诊断依据是 Top-1 与 Top-2 分数差 p50 仅 0.005（48/52 ≤0.12），同设备等价手册几乎同分，排序实由并列规则决定。该重排**不改文档集合与内容**，只在组内稳定换序，`build_context` 的 `[来源N]` 与 `build_citations` 的 `index` 仍严格对齐。
- **评估口径不放松**：主集扩到 131 条、独立留出集 24 条；新增 `equivalent_hit` 仅作诊断次指标，主命中率仍用严格口径以维持与基线可比；分布均衡、留出集泛化差距、逐 id 退化清单与机器可读门槛 `config/iteration_criteria.json` 均写入每轮报告。

**设计取舍**：不去删除特判能力（用户要求配置化保留），也不把测试语义耦合进 `ask_stream`；答案模板用无冲突 token 占位符渲染，避免 LaTeX 花括号与 `str.format` 冲突；`scripts/auto_ingest.sh` 本轮不改动，重建只在 `page_content` 哈希变化时才触发。

### 本版（v3.2.0）已解决：诊断驱动的检索优化

以 63 条测试集（`tests/rag_perf_queries.json`）与 `scripts/run_rag_perf.py` 建立质量可观测性后，按证据分层修复：

- **覆盖率不是命中率**：旧评估只输出 `coverage=high`（基于 rerank 分数 max/avg）与路由准确率，36 条全 high 无法证明引用命中真实来源。新增 grounding 命中、Top-1 命中、召回父块数、拒答正确率与引用来源明细。
- **目录枚举重复 IO 与噪声**：`find_best_matching_dirs` 与 `list_fs_entries_by_keyword` 各自对网络盘全盘 `os.walk`，是 file_list 25.9s / hybrid 32.3s 的主因；且 `.svn/pristine/a2` 会被 `A2X小组` 误命中。改为单次遍历快照 + TTL 缓存并排除版本控制目录，离线回归验证返回结构与排序严格等价。实测 file_list -67%、hybrid -52%。
- **hybrid 16 路同义 Dense 往返**：目录驱动查询为「同一 intent + 不同设备路径」，信息增量低。改为归一化去重、按问题相关性排序并由 `MAX_HYBRID_QUERIES` 限流；完整设备清单仍由目录枚举答案区块直接注入，可追溯性不变。
- **单一长手册垄断父块**：`843-R-USB-User-Manual` 一份文档曾贡献 17/20 个子块、占据父块榜前 14 名，叠加 3000 字符预算使正确文档（排 35 名后）永远无法入 prompt。改为父块展开预算与上下文预算解耦 + `PARENT_MAX_PER_SOURCE` 来源配额，召回父块数由 2 提升到 7~13。
- **纯英文型号词元化缺陷**：`N9322C` 被旧正则整个丢弃、`EDU36311A` 截成 `EDU36311`。补齐复合型号、字母数字混合型号与数字带字母后缀三类形态。
- **域外问题假引用**：reranker 对天气/K8s/Django 也给出 0.75~0.8 分，`assess_coverage` 判 high 并生成引用。新增可答性闸门，仅用文档身份类字段（排除生成式摘要）判断关键词是否有证据，配合 Top-1 分数下限；拒答正确率 2/5 → 5/5，基线 52 个有答案案例零误伤。
- **OCR 噪声放大**：OCR 残缺目录页的 U+FFFD 点线被表格语义转写逐行复制，替换符 1967 → 6031。跳过高替换符占比的行，正常表格转写不变（此项改变 `page_content`，需全量重建并把 `RETRIEVAL_INDEX_VERSION` 递增到 2）。

**设计取舍**：质量诊断放在评估脚本而非生产代码，避免把测试语义耦合进 `ask_stream`；A 类时延优化要求回答语义等价，故仅消除重复 IO 与冗余往返；B/C 类改动均有配置开关与离线回归；所有优化不改 SSE 事件字段、`/api/health` 结构与 `[DONE]` 唯一注入。

### 本版（v3.1.0）已解决

- HyDE 假设文档从未参与检索、`keywords` 恒为空导致全文词项检索只用整句、表格配置项读入 cfg 却被函数直接 `os.getenv` 绕过
- Dense 排名依赖并发完成顺序、RRF 分数被重排完全丢弃、去重键两处不一致
- 相同父文本导致子块 ID 冲突、索引创建失败被静默吞掉、加载异常被吞导致静默缺文档
- 过滤用带前缀文本误杀表格/型号块、父块纯字符截断腰斩表格、代码块内 `#` 被误判标题
- 路径层级未进入向量化前缀（不同设备同名手册不可区分）、覆盖度与引用脱节、低分保底造成假引用、降级路径无组件标记

### 后续优化方向

- **真正的稀疏向量**：引入 BM25 / SPLADE 或 bge-m3 lexical weights 替代 MatchText（需新增依赖并变更集合 schema）
- **路由词典动态化**：从 `rel_path` 离线生成「小组 ↔ 设备型号 ↔ 手册类型」词典，替代静态小组别名
- **缓存重排结果**：在保证 metadata 不被就地修改的前提下缓存 rerank + 父块展开结果，进一步降低 GPU 与 Qdrant 开销
- **source 相对化**：改用 `rel_path` 作为确定性 ID 种子，避免挂载点变化产生整批孤儿数据
- **用户反馈机制**：引入点赞/点踩反馈，收集用户评价数据持续优化检索和生成质量
- **高可用部署**：多节点部署方案，vLLM 多副本 + 负载均衡，Qdrant 集群模式
