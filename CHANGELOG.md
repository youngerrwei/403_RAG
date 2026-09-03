# Changelog

本文件记录项目的所有重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/)，版本号遵循 [Semantic Versioning](https://semver.org/)。

---

## [2.0.2] - 2026-09-02

### Fixed

- 健康检查现在同时验证 Qdrant 连通性及子块、父块集合是否存在，避免空实例被误报为可检索；运行时也会在加载 GPU 模型前拒绝缺少集合的实例
- 启动预检会分别提示文档目录为空、Qdrant 不可达和必要集合缺失，并给出全量入库恢复命令
- 统一启动脚本的 conda 定位、配置加载、进程身份校验及认证健康检查，避免误杀端口占用进程和假就绪
- Web 健康检查改为快速快照与后台预热；Qdrant 不可用时保持 `degraded`，vLLM 不可用时严格返回 `error`
- 修复 SSE 并发令牌重复释放、客户端断连后后台生成未及时取消的问题
- ReAct Agent、工具层与普通 RAG 共用唯一运行时和用户历史，消除重复模型加载与历史串用
- 入库任一批失败时严格返回非零，并在成功覆盖后清理同源陈旧子块/父块
- 文档转换改为临时文件校验后原子替换，任一文件失败时返回非零且保留成功结果
- 修复 vLLM 模型响应中权限对象的 `id` 被误当作模型 ID，以及绝对模型路径被误判为不匹配的问题
- 对齐用户文件、日志、文档目录、Embedding 初始化超时与目录浏览上限的代码默认值和 `.env.example`

### Changed

- `.env` 改为本机私有文件；仓库提供 `.env.example`，安装脚本自动生成随机 Flask 密钥
- 三套运行环境均由配置指定，所有管理脚本使用明确解释器，不依赖当前 shell 环境
- 重写 README 首次部署、日常启动与健康状态说明，使命令与实际脚本行为一致

---

## [1.4.1] - 2026-06-26

### Changed

- GPU 配置调整：vLLM 推理服务迁移至 GPU 3（原 GPU 0），Embedding/Reranker 迁移至 GPU 2（原 GPU 1）
- 新增 `ARCHITECTURE.md` 系统架构设计文档，从 README 迁移并扩展架构相关内容
- README.md 精简重组：移除架构详情，保留概述和快速开始，添加 ARCHITECTURE.md 引用链接
- 对齐 TROUBLESHOOTING.md、AGENTS.md 中所有 GPU 设备号引用

---

## [1.4.0] - 2025-06-25

### Added

- 入库摘要增强：入库时自动调用 vLLM 为每个父块生成 150-250 字的智能摘要
- 摘要注入子块前缀，参与 embedding 计算，增强语义检索的宏观理解能力
- 摘要存入 metadata（`parent_summary` 字段），检索时可作为 LLM 上下文辅助
- 并发摘要生成支持（`SUMMARY_MAX_WORKERS` 可配置并发数）
- 摘要缓存机制（`SummaryCache`），增量入库时避免重复生成
- 新增 8 个配置项：`ENABLE_SUMMARY_AUGMENTATION`、`SUMMARY_VLLM_TIMEOUT`、`SUMMARY_VLLM_RETRIES`、`SUMMARY_MAX_WORKERS`、`SUMMARY_MAX_TOKENS`、`SUMMARY_INJECTION_MODE`、`ENABLE_SUMMARY_CACHE`、`SUMMARY_CACHE_DIR`
- `build_context()` 检索展示增强：LLM 生成回答时可参考每个片段的段落摘要

### Changed

- `split_documents()` 返回值新增 `summary_map`，支持摘要透传
- `store_parent_chunks_batch()` 新增 `summary_map` 参数，父块 payload 中存储摘要
- `build_contextual_prefix()` 签名扩展，支持摘要注入

---

## [1.3.0] - 2025-06-25

### Added

- 查询路由增强：`rule_based_route()` 新增计数/列举类关键词支持（"有多少"、"多少篇"、"几篇"、"统计"、"数量"、"一共有"、"总共有"、"共有"、"全部"、"所有"）
- 歧义词扩展：`ambiguous_words` 新增"哪些"、"几个"、"几台"，使列举+学术混合查询自动走 hybrid 路由
- Hybrid 路由上下文预算控制：文件列表部分限制 1500 字符，RAG 检索部分动态分配剩余预算，防止总上下文超限
- LLM 回答策略优化：新增列举型问题的 prompt 引导规则（先完整列出条目，再对 top 3-5 提供摘要）

### Fixed

- 修复计数/列举型查询（如"有多少篇论文"）被错误路由到 rag_search 导致 topK=16 截断的问题
- 修复 hybrid 路由下文件列表+RAG内容拼接可能超出上下文预算的问题

---

## [1.2.1] - 2025-06-25

### 优化

- 移除 `ingest.py` 中不必要的 dict() metadata 拷贝，改为原地修改
- 移除 `ingest.py` 中子文档 metadata 的整体覆盖，改用 update() 增量更新
- 移除 `rag_agent.py` RRF 融合中冗余的 list 转换，改用集合并操作
- 移除 `rag_agent.py` 重排序循环中重复的 `.lower()` 调用
- 优化 `rag_agent.py` 父块展开中的 dict copy + pop 为 dict comprehension

---

## [1.2.0] - 2025-06-25

### Added

- HyDE (Hypothetical Document Embeddings) 查询扩展：LLM 生成假设文档增强 Dense 检索召回率
- 智能 Query 路由：短问题/明确关键词查询跳过 LLM 改写，降低检索时延
- 三阶段流式反馈 UI：即时处理提示 → 检索状态 → 流式生成，用户感知延迟显著降低
- SSE 新增 `status` 事件类型，支持前端实时展示处理阶段
- 入库批量写入指数退避重试机制（最多3次重试）
- 向量维度运行时验证，启动时检查 Embedding 维度是否在预期范围

### Fixed

- Sparse 检索补全 `file_name` 和 `rel_path` 字段，提升按文件名/路径查询的召回率
- `get_runtime()` 添加 Double-check Locking，修复多线程首次调用的竞态条件
- 覆盖度评估改进：综合 max_score、avg_score 和 doc_count 多维度判断
- 查询路由歧义词判断优化：要求至少2个学术词共现，减少误分类
- 入库标题切分 `chunk_overlap` 硬编码改为读取配置
- 集合创建和父块存储添加异常处理
- `web_app.py` 修复 `get_json` 参数冲突（移除 `force=True`）

### Changed

- `MAX_CONCURRENT_REQUESTS` 默认值从 50 调整为 20
- 新增 `.env` 配置项：`ENABLE_HYDE`、`HTTPS_ENABLED`

---

## [1.1.0] - 2025-06-25

### Added

- Contextual Retrieval：子块 embedding 前注入文档/章节上下文前缀，提升向量语义丰富度
- HNSW 索引优化：创建集合时使用 m=32, ef_construct=200 参数，提升检索精度
- 知识库覆盖度评估：基于重排分数分5级自动评估，附带用户提示文案
- 答案引用溯源：自动提取参考文档来源和章节信息（最多5条）
- 分级 Fallback 策略：根据覆盖度级别动态调整 LLM system prompt
- 前端引用展示 UI：可折叠引用来源列表 + 覆盖度提示条
- `auto_ingest.sh --destroy`：快速销毁知识库（Qdrant REST API 删除集合）
- `auto_ingest.sh --destroy --force`：跳过确认的强制销毁模式
- 启动时配置验证：关键配置缺失或设备为 cpu 时输出警告日志
- TTL 缓存主动清理：set() 时自动清除过期和超容条目
- 版本控制：引入 VERSION 文件和 CHANGELOG.md
- AGENTS.md：AI Agent 代码修改规范文档

### Fixed

- 修复 parent_id 碰撞：改为 source+content 组合哈希，防止不同文档相同内容覆盖丢失
- 修复批量写入无容错：单批失败记录并继续，不中止整个入库任务
- 修复有效字符过滤阈值过高（30→10）：减少误杀含公式/表格的学术内容
- 修复代码默认值与 .env 推荐值不一致（差异 60-75%）：统一为 INITIAL_RETRIEVAL_K=64, FINAL_TOP_K=16, MAX_CONTEXT_CHARS=3000
- 修复 RRF 融合 sparse_only 极度降权：消除单路文档不公平惩罚
- 修复规则路由关键词冗余和歧义词处理遗漏 hybrid 场景
- 修复查询改写 prompt 过于简陋：全面优化为结构化指令+规则约束
- 修复异常信息泄露：统一对外错误响应为通用提示

### Changed

- 新增 .env 配置项：RRF_K、KEYWORD_BOOST、TITLE_BOOST、MAX_CONCURRENT_REQUESTS、HTTPS_ENABLED
- Sparse 检索保护：关键词限制10个、MatchText条件限制30个、异常捕获
- 并行检索超时保护：ThreadPoolExecutor 添加 timeout=30s
- 大文件预警：>50MB 自动跳过并警告
- Rerank 阈值保护确认：确保至少保留 top-1 结果

### Security

- Session Cookie 安全：HTTPONLY + SAMESITE=Strict + SECURE（HTTPS时）
- 登录速率限制：5次/分钟/IP
- Flask 密钥弱检测：启动时检查密钥长度并警告
- 并发连接数限制：Semaphore 控制最大50个并发问答请求
- SSE 超时和心跳：5分钟总超时 + 15秒心跳包
- Session ID 重生成：登录时 clear 防 fixation 攻击
- CSP 安全头：完整 Content-Security-Policy 策略
- 请求大小限制：MAX_CONTENT_LENGTH = 1MB

---

## [1.0.0] - 2025-06-24

初始版本。

### Added

- Small-to-Big 检索策略（子块300字符检索 → 父块1500字符展开）
- 混合检索：Dense(bge-m3) + Sparse(Qdrant MatchText) + RRF 融合
- CrossEncoder 重排序（bge-reranker-v2-m3）
- 两级查询路由（规则引擎 + LLM 兜底）
- 多查询改写 + 并行检索
- Flask + SSE 流式问答
- PBKDF2 密码认证
- 按日期分文件的对话历史管理
- Markdown 文档入库（标题结构切分 + 父子块切分 + 质量过滤）
- 确定性 Point ID（UUID5）支持幂等入库
- vLLM 推理服务一键管理
- auto_ingest.sh 增量入库脚本
