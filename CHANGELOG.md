# Changelog

本文件记录项目的所有重要变更。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/)，版本号遵循 [Semantic Versioning](https://semver.org/)。

---

## [Unreleased]

## [3.0.2] - 2026-09-14

### Changed

- 转换脚本在扫描前真实验证输出目录创建权限，避免 MinerU 完成后才因 CIFS `Permission denied` 失败；大文件按体积动态计算超时，默认大小上限调整为 200 MB。
- 入库清洗保留 Markdown 表格/标题/列表换行，并为所有标准表格生成有界逐行语义转写；兼容恢复旧版已压平的变量格式表。
- 转换阶段检测疑似扁平技术表并自动使用 MinerU VLM 重试；无法恢复的表格会被标为不可靠，禁止回答端猜测。
- 变量格式恢复明确按合并行顺序对应：英文字母向量为小写正体粗体，矩阵为大写正体粗体。
- 检索上下文不再二次压平 Markdown 表格，回答提示明确要求忠实保留正体、斜体、粗体和大小写规则；前端支持原文 `\bs{}` 数学宏。
- Qdrant 目录回退结果按目标目录的直接子级聚合，避免设备清单展开成数十个 Manual 文件；混合检索保留用户的使用方法、操作规范和注意事项意图。
- 目录检索同时检查 `KNOWLEDGE_BASE_ROOT` 与 `DOCS_PATH`，并为中文路径增加 Qdrant payload 有界扫描回退。
- 混合回答固定输出目录/文件区块；前端空结果会展示检索目标与根目录状态，便于定位挂载配置问题。
- 混合检索严格使用目录枚举出的路径与文件名驱动查询；文件系统无匹配时自动回退 Qdrant 路径元数据。
- 检索与入库统一过滤 `<think>` 及整段问答污染摘要，避免内部推理泄漏到最终回答。
- 修复 MathJax `\bs{}` 宏参数中的 `{#` 被 Jinja 误判为未闭合注释、导致登录后主页返回 HTTP 500；前端回归增加模板注释闭合检查。
- 修复内容格式类问题被 LLM 误路由到 `file_list`；规则路由新增格式/写作要求意图，并对缺少明确目录枚举意图的 LLM 结果进行纠偏，同时确保短问句不会被误当作纯关键词而跳过查询改写。
- Agent 使用与具体问题无关的路由工具状态机校验模型实际输出的 `Action`，不再合成思考或伪造调用；无依据的提前回答及错误工具会退回模型自行重选，仅混合问题要求“目录后正文检索”，并移除会把问题污染成“这些工具怎么用”的固定提示。

- README 增加生产 SMB Share 的 `/etc/fstab` 持久化挂载、UID/GID 强制映射、按挂载状态激活 systemd、重启后验收与故障诊断步骤
- 入库管理脚本改为无缓冲实时显示各阶段日志并同步落盘，结束时报告耗时，且保留 Python 子进程退出码
- vLLM 启动、状态和入库前置检查增加最小真实生成探针，避免 API/模型端点正常但推理引擎卡死时产生健康假阳性和批量摘要超时
- vLLM 管理脚本新增受约束的 `stop --orphan` 恢复入口，仅允许停止模型路径、端口和进程命令均符合当前配置的未纳管 vLLM
- 入库锁冲突时输出锁持有进程的 PID、父 PID、运行时长、状态和命令行，便于安全定位残留任务
- RAG 启动与状态检查可在 Web 命令身份和健康接口均匹配时恢复丢失的 PID 文件，避免健康旧进程被误报为未知端口占用
- Web 显式绑定包内模板目录，启动前验证登录页和问答页可读，并以 ERROR 级完整记录登录渲染异常
- 混合路由会从完整问题中提取目录实体、拆分中文连接词并识别设备/规范意图，确保目录枚举结果进入 UI 和回答上下文
- Qwen3 Web 与摘要请求默认关闭 thinking 输出，避免内部推理过程混入最终答案；混合模式同时考虑目录命中和正文覆盖度
- Agent 流将文件工具以及内部 RAG 混合路由得到的结构化目录结果继续转发给前端，修复文件卡片为空

---

## [3.0.1] - 2026-09-03

### Fixed

- 修复 `use_agent=true` 分支仍使用迁移前 `agent_entry` 绝对导入、导致包模式启动后 ReAct 链路失败的问题
- 修正 MCP 异常提示和 `.env.example` 中迁移前的根目录脚本路径
- 新增包内导入与 Shell 启动链契约测试，阻止生产包重新引入根目录时代的导入或入口方式

### Changed

- README 新增 v2.x/v3.x 启动命令对照、完整启动调用链、逐文件职责说明及 `requirements/rag.txt` 逐项依赖说明

---

## [3.0.0] - 2026-09-03

### Changed

- 将生产 Python 代码整理为 `src/lab_rag/` 包，并改用模块入口启动 Web、入库、用户管理和 MCP Bridge
- 将部署维护脚本、测试、历史实验、技术文档、依赖清单和样例资料分别归档到 `scripts/`、`tests/`、`legacy/`、`docs/`、`requirements/` 与 `data/samples/`
- 所有 Shell 脚本统一从自身位置解析项目根目录，保持 `.env`、模型、日志、状态文件与运行时数据仍位于项目根目录
- 更新 README、架构文档、排障文档、Agent 规范与可靠性回归中的路径、命令及导入约定

### Added

- 新增 `lab_rag.paths`，统一解析项目根目录与项目相对路径
- 扩充 `.gitignore`，排除 Python 缓存、测试缓存、日志、用户凭据及运行时状态数据

---

## [2.1.1] - 2026-09-03

### Fixed

- 补齐 RAG 主环境直接依赖清单与安装后导入校验，避免首次部署缺少 `requests` 或 `langchain-text-splitters`
- 文档转换脚本统一支持 `RAG_ENV_FILE`、项目相对路径和可靠的 conda 环境识别，并对缺失选项参数给出明确错误
- 入库变更检测增加 manifest 路径比对，避免保留旧 mtime 的新 Markdown 文件被漏过
- 用户凭据文件改为同目录原子写入；损坏的 JSON 不再被静默重置，也不再把盐值和密码哈希输出到终端
- 启动结果明确区分 `ok` 与 `degraded`，并正确解析相对 `DOCS_PATH`

### Changed

- README 补充运行前提、共享盘/Qdrant 恢复、日志、回归、依赖升级和 cron 完整处理链路，并澄清日常入库的实际覆盖语义

---

## [2.1.0] - 2026-09-03

### Added

- 新增可选 stdio MCP Bridge，暴露 `search_lab_knowledge` 纯检索工具与 `list_lab_catalog` 目录工具
- 新增仅接受本机 loopback + Bearer Token 的内部 JSON API，MCP 调用与原 Web 问答共用并发限制和唯一 RAG 运行时
- 新增 `setup_mcp.sh` 与 `start_mcp.sh`，使用独立 `rag-mcp` 环境并自动生成内部 Token
- 新增 MCP 鉴权、结果上限、停止最终回答生成、目录投影及 SDK 内存客户端契约测试

### Changed

- RAG 流式入口的 `persist_history=False` 能力供 MCP 旁路复用；原 `/ask_stream`、SSE 协议和启动链路保持不变
- 运行时调试日志同时隐藏 vLLM API Key 与 MCP 内部 Token

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
