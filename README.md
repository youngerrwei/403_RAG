# LAB 403 RAG 知识库系统

> Author：youngerrwei（韦子扬）<br>
> 当前版本：v1.4.1

## 项目概述

实验室知识库 RAG 系统，支持论文、实验室规范、使用教程等文档的智能问答。基于 Small-to-Big 检索策略，结合混合检索（Dense + Sparse）与 CrossEncoder 重排序，提供高质量的知识检索与流式生成服务。

## 系统架构

四卡 GPU 环境（GPU2: Embedding + Reranker, GPU3: vLLM 推理），核心组件：

- **vLLM**：Qwen3-8B-Instruct 推理服务（端口 8000）
- **Flask Web 应用**：用户认证 + SSE 流式问答（端口 5000）
- **Qdrant**：向量数据库，存储子块集合 + 父块集合
- **bge-m3 / bge-reranker-v2-m3**：Embedding 与重排序模型

> 详细架构设计、数据流和技术决策请参阅 [ARCHITECTURE.md](ARCHITECTURE.md)

## 快速开始

### 环境准备

> **推荐**：可使用一键环境准备脚本完成以下所有步骤：
>
> ```bash
> bash setup_env.sh
> ```
>
> 该脚本会自动完成 Python 依赖安装、文档转换工具安装、模型检查、配置文件初始化、目录创建等操作，并在结束时输出完整的环境检查报告。支持 `--skip-deps`、`--skip-models`、`--skip-converter` 参数跳过特定步骤，详见 `bash setup_env.sh --help`。

#### 手动环境准备

```bash
# 1. Python 依赖安装
pip install flask python-dotenv langchain langchain-openai langchain-huggingface \
    langchain-qdrant qdrant-client sentence-transformers tiktoken vllm

# 2. 文档转换工具（三选一，推荐 MinerU）
pip install uv && uv pip install -U "mineru[all]"
# 备选：pip install marker-pdf[full]
# 备选：pip install docling

# 3. 下载模型
bash download_model.sh                    # 下载 Qwen3-8B-Instruct
# Embedding 和 Reranker 模型需手动放到 ./models/ 目录：
#   ./models/bge-m3
#   ./models/bge-reranker-v2-m3
```

### 首次部署完整流程

```bash
# Step 1: 配置环境变量
#   编辑 .env，确认 GPU 分配、Qdrant 地址、模型路径等
#   重要：FLASK_SECRET_KEY 需替换为随机强密钥

# Step 2: 创建用户账号
python create_user.py

# Step 3: 转换文档为 Markdown（PDF/DOCX/PPTX → .md）
bash convert_to_md.sh --full

# Step 4: 首次全量入库
bash auto_ingest.sh --full

# Step 5: 启动全部服务
bash start_rag.sh start

# Step 6: 验证服务状态
curl http://127.0.0.1:5000/api/health
# 期望返回: {"status": "ok", "components": {...}}
```

### 日常运维流程

```bash
# 新增文档后：转换 + 增量入库
bash convert_to_md.sh          # 仅转换新增/修改的文档
bash auto_ingest.sh            # 仅入库变更的 .md 文件

# 服务管理
bash start_rag.sh status       # 查看服务状态
bash start_rag.sh restart      # 重启服务
bash start_rag.sh stop         # 停止服务

# 重建知识库（修改了切块参数/Embedding 模型后必须执行）
bash auto_ingest.sh --destroy --force
bash auto_ingest.sh --full
```

## 脚本使用详解

### convert_to_md.sh — 文档格式转换

将 PDF/DOCX/DOC/PPTX/PPT 等格式转换为 Markdown，供入库系统使用。

#### 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--source DIR` | .env 中 `DOCS_PATH` | 源文档目录 |
| `--output DIR` | 与源目录相同 | 输出目录 |
| `--engine ENGINE` | 自动选择 | 指定引擎：`mineru` / `marker` / `docling` |
| `--backend BACKEND` | `pipeline` | MinerU 后端：`pipeline` / `hybrid` / `vlm` |
| `--device DEVICE` | `cuda:0` | GPU 设备 |
| `--full` | - | 全量转换（忽略时间戳） |
| `--dry-run` | - | 仅预览待转换文件，不执行 |

#### 使用示例

```bash
# 增量转换（仅处理新增/修改文件）
bash convert_to_md.sh

# 全量转换（首次部署推荐）
bash convert_to_md.sh --full

# 指定源目录和引擎
bash convert_to_md.sh --source /data/papers --engine marker

# 使用 MinerU VLM 后端处理复杂 PDF
bash convert_to_md.sh --engine mineru --backend vlm --device cuda:1

# 预览模式：查看哪些文件会被转换
bash convert_to_md.sh --dry-run
```

#### 注意事项

- **引擎优先级**：MinerU > Marker > Docling（自动检测已安装引擎）
- **增量逻辑**：基于 `data/.convert_state` 时间戳判断文件是否需要重新转换
- **文件锁**：`/tmp/convert_to_md.lock`，防止并发执行
- **支持格式**：`.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`（`.xlsx` 仅 MinerU 支持）
- **日志文件**：`logs/convert_to_md.log`

---

### auto_ingest.sh — 知识库入库管理

管理知识库的增量入库、全量入库和集合销毁。

#### 参数表

| 参数 | 说明 |
|------|------|
| （无参数） | 增量入库：检测新增/修改的 `.md` 文件并入库 |
| `--full` | 全量入库：重建集合 + 入库所有文件 |
| `--destroy` | 销毁知识库（交互式确认） |
| `--destroy --force` | 强制销毁知识库（跳过确认） |

#### 使用示例

```bash
# 增量入库（日常使用）
bash auto_ingest.sh

# 全量入库（首次部署 / 配置变更后）
bash auto_ingest.sh --full

# 销毁知识库（交互式确认）
bash auto_ingest.sh --destroy

# 强制销毁 + 全量重建（脚本/CI 场景）
bash auto_ingest.sh --destroy --force && bash auto_ingest.sh --full
```

#### 增量检测机制

- **状态文件**：`data/.ingest_state`（记录上次入库时间戳）
- **清单文件**：`data/.ingest_manifest`（记录已入库文件列表）
- 使用 `find -newer` 检测新增/修改文件
- 文件删除检测：对比 manifest 发现缺失文件后自动触发全量重建

#### 注意事项

- **文件锁**：`/tmp/auto_ingest.lock`，防止并发执行
- **日志文件**：`logs/auto_ingest.log`
- 失败时不更新状态文件，下次运行自动重试
- 增量模式自动设置 `QDRANT_RECREATE_COLLECTION=false`
- 配合 cron 实现定时入库：
  ```bash
  0 3 * * * /path/to/403_RAG/auto_ingest.sh >> /path/to/403_RAG/logs/auto_ingest_cron.log 2>&1
  ```

---

### start_rag.sh — 服务启动管理

一键管理 vLLM 推理服务 + Flask Web 应用的生命周期。

#### 子命令

| 命令 | 说明 |
|------|------|
| `start` | 预检 → 启动 vLLM → 启动 web_app |
| `stop` | 优雅停止（SIGTERM → 等 5 秒 → SIGKILL） |
| `restart` | stop + 2 秒等待 + start |
| `status` | 显示服务状态 + GPU 使用情况 |

#### 使用示例

```bash
bash start_rag.sh start      # 启动全部服务
bash start_rag.sh stop       # 停止全部服务
bash start_rag.sh restart    # 重启全部服务
bash start_rag.sh status     # 查看运行状态和 GPU 占用
```

#### 启动预检逻辑（preflight_check）

| 检查项 | 失败行为 |
|--------|----------|
| `.env` 文件存在性 | **中止启动** |
| 模型目录存在性（vLLM / Embedding / Reranker） | **中止启动** |
| Qdrant 连通性 | 仅警告，不中止（允许 Qdrant 稍后启动） |
| 日志/数据目录 | 自动创建 |

#### 内部行为

- **vLLM 启动**：调用 `start_vllm.sh --background`，检测端口 8000 占用，相同模型已运行则跳过
- **vLLM 就绪**：轮询 `/health` 端点，最多等待 120 秒
- **web_app 启动**：`nohup python web_app.py`
- **web_app 就绪**：优先 `/api/health`（status=ok/degraded），回退 `/login`，最多等待 60 秒
- **PID 管理**：`data/.vllm.pid` 和 `data/.web_app.pid`

---

### start_vllm.sh — vLLM 推理服务

独立管理 vLLM 推理服务的启动。

#### 参数表

| 参数 | 说明 |
|------|------|
| `--gpu N` | 指定 GPU（如 `--gpu 0` 或 `--gpu 0,1`），覆盖 .env 中的配置 |
| `--background` | 后台运行，日志输出到 `logs/vllm_server.log` |
| （无参数） | 前台运行，阻塞当前终端 |

#### 使用示例

```bash
# 后台启动（start_rag.sh 内部调用方式）
bash start_vllm.sh --background

# 前台启动（调试用）
bash start_vllm.sh

# 指定 GPU 后台启动
bash start_vllm.sh --gpu 0,1 --background
```

#### 注意事项

- **健康检查**：轮询 `/health` + `/v1/models`，最多 120 秒
- **PID 文件**：后台模式保存到 `data/.vllm.pid`
- **模型路径**：本地路径不存在时提示运行 `download_model.sh`
- **端口**：默认 8000（由 .env 中 `VLLM_PORT` 控制）

## 环境配置 (.env)

关键配置项概览（完整配置见 `.env` 文件注释）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `VLLM_CUDA_DEVICES` | `3` | vLLM GPU 设备号 |
| `VLLM_MODEL_NAME` | `./models/Qwen3-8B-Instruct` | 推理模型本地路径 |
| `VLLM_API_KEY` | `lab-secret-key` | vLLM 服务密钥 |
| `VLLM_PORT` | `8000` | vLLM 端口 |
| `VLLM_GPU_UTIL` | `0.85` | GPU 显存利用率 |
| `VLLM_MAX_MODEL_LEN` | `6000` | 最大上下文长度 |
| `QDRANT_HOST` | `172.18.216.71` | Qdrant 服务地址 |
| `QDRANT_PORT` | `6333` | Qdrant 端口 |
| `EMBEDDING_MODEL_NAME` | `./models/bge-m3` | Embedding 模型路径 |
| `EMBEDDING_DEVICE` | `cuda:2` | Embedding 运行设备 |
| `RERANKER_MODEL_NAME` | `./models/bge-reranker-v2-m3` | Reranker 模型路径 |
| `RERANKER_DEVICE` | `cuda:2` | Reranker 运行设备 |
| `DOCS_PATH` | `/mnt/cpu_share` | 知识库文档目录 |
| `FILE_SEARCH_LIMIT` | `200` | 文件搜索结果上限 |
| `MAX_CONCURRENT_REQUESTS` | `20` | 最大并发问答数 |

> **重要**：所有模型统一使用本地路径（`./models/`），不要使用 HuggingFace 在线路径。

## 知识入库详解

### 入库流程说明

系统采用 **Small-to-Big** 切分策略：

1. **文档加载**：递归扫描目录，多编码尝试读取 Markdown 文件
2. **文本清洗**：移除 HTML 注释/标签、图片语法、零宽字符
3. **标题结构切分**：按 Markdown 标题（#/##/###）拆分为逻辑段落
4. **父块切分**：1500 字符父块，200 字符重叠
5. **子块切分**：300 字符子块，50 字符重叠，metadata 中保存 parent_id
6. **质量过滤**：过滤过短、无效字符不足、数字占比过高的块
7. **向量化写入**：bge-m3 生成 1024 维向量，分批写入 Qdrant

### 增量入库 vs 全量入库

| 模式 | 命令 | 适用场景 |
|------|------|----------|
| 增量 | `bash auto_ingest.sh` | 日常新增/修改少量文档 |
| 全量 | `bash auto_ingest.sh --full` | 首次部署、修改切块参数后 |

增量入库使用确定性 UUID（基于 source + parent_id + chunk_index 的 UUID5），相同内容不会产生重复。

### 重建知识库

修改以下任一项后，**必须全量重建**：
- metadata 字段（新增/删除/重命名）
- chunk 大小或前缀格式
- UUID 生成逻辑
- Embedding 模型

```bash
bash auto_ingest.sh --destroy --force && bash auto_ingest.sh --full
```

## API 端点

### GET /api/health（无需认证）

健康检查端点，被 `start_rag.sh` 用于验证服务就绪。

```bash
curl http://127.0.0.1:5000/api/health
```

响应：

```json
{
  "status": "ok",
  "components": {
    "embedding": true,
    "reranker": true,
    "qdrant": true,
    "llm": true
  }
}
```

| status 值 | HTTP 状态码 | 含义 |
|-----------|------------|------|
| `"ok"` | 200 | 全部组件正常 |
| `"degraded"` | 200 | 部分组件不可用，系统仍可接受请求 |
| `"error"` | 503 | 关键服务不可用 |

### POST /ask_stream（需认证）

SSE 流式问答接口，需先登录获取 Session。

### POST /clear（需认证）

清除当前用户的对话历史记录。

### GET /whoami（需认证）

获取当前登录用户信息。

### POST /save_history（需认证）

保存前端对话历史到服务端（合并去重）。

### POST /login（无需认证）

用户登录接口，速率限制 5 次/分钟/IP。

### GET|POST /logout（无需认证）

用户登出，清除 Session。

## 常见问题与故障排查

### 启动类问题

**现象**：`start_rag.sh start` 报错 "模型目录不存在"
- **原因**：`./models/` 下缺少对应模型文件
- **解决**：运行 `bash download_model.sh` 下载 LLM 模型；Embedding/Reranker 模型需手动放置

**现象**：vLLM 启动超时（等待 120 秒后失败）
- **原因**：GPU 显存不足或模型文件损坏
- **解决**：用 `nvidia-smi` 检查 GPU 显存；确认 `VLLM_GPU_UTIL` 未设过高；检查模型文件完整性

**现象**：web_app 启动后 `/api/health` 返回 `"error"`
- **原因**：Embedding/Reranker 模型加载失败或 Qdrant 不可达
- **解决**：检查 `cuda:2` 显存是否足够；确认 Qdrant 服务（`172.18.216.71:6333`）是否可访问

### 入库类问题

**现象**：增量入库未检测到新文件
- **原因**：文件修改时间早于 `data/.ingest_state` 记录的时间戳
- **解决**：使用 `bash auto_ingest.sh --full` 执行全量入库

**现象**：入库后查询无结果
- **原因**：检索缓存 TTL 为 300 秒，新入库内容最多 5 分钟延迟
- **解决**：等待 5 分钟后重试；或重启 web_app 清除缓存

**现象**：入库时报 Qdrant 连接失败
- **原因**：Qdrant 服务未启动或网络不通
- **解决**：`curl http://172.18.216.71:6333/collections` 验证连通性

### 检索类问题

**现象**：检索结果质量差，不相关内容多
- **原因**：`RERANKER_SCORE_THRESHOLD` 过低或文档切块粒度不合适
- **解决**：适当提高 `RERANKER_SCORE_THRESHOLD`（默认 0.3）；调整切块参数后全量重建

**现象**：`MatchText` 过滤报错
- **原因**：Qdrant 中对应字段未创建文本索引
- **解决**：销毁并重建知识库（全量入库时会自动创建索引）

### 性能类问题

**现象**：响应速度慢（>10 秒）
- **原因**：并发请求过多或 GPU 负载高
- **解决**：检查 `MAX_CONCURRENT_REQUESTS` 设置；用 `nvidia-smi` 查看 GPU 利用率

**现象**：OOM（显存溢出）
- **原因**：Embedding + Reranker + vLLM 显存冲突
- **解决**：确保 Embedding/Reranker（cuda:2）与 vLLM（cuda:3）在不同卡上

## 安全加固

| 防护项 | 实现方式 |
|--------|----------|
| Session 安全 | HttpOnly + SameSite=Lax；HTTPS 模式下启用 Secure；登录后重生成 Session ID |
| 速率限制 | 登录接口 5 次/分钟/IP |
| 并发控制 | Semaphore 限制最大并发请求数（默认 20） |
| 信息泄露防护 | 异常响应统一通用错误提示，不暴露内部信息 |
| SSE 连接保护 | 超时 5 分钟断开，每 15 秒心跳保活 |
| 安全响应头 | CSP 限制资源加载来源 |
| 请求限制 | 请求体大小 1MB |
| 密钥检测 | 启动时检测 SECRET_KEY 强度 |

## 文件结构

#### 当前在用文件

| 文件 | 用途 |
|------|------|
| `.env` | 环境变量配置（vLLM、Qdrant、Embedding、Reranker、切块参数等） |
| `rag_agent.py` | 核心 RAG 流程：两级查询路由（规则+LLM）、问题改写、多路并行检索（Dense+Sparse）、RRF 融合、重排序、父块展开（独立 Collection 查询）、TTL 缓存、流式回答生成、文件系统工具 |
| `web_app.py` | Flask 后端：登录认证（PBKDF2）、SSE 流式问答接口、历史管理 |
| `ingest.py` | 知识库入库：Markdown 加载、标题结构切分、父子块切分（Small-to-Big）、质量过滤、批量写入 Qdrant |
| `create_user.py` | 用户账号创建脚本 |
| `setup_env.sh` | 环境准备一键脚本（依赖安装、模型检查、配置初始化） |
| `start_vllm.sh` | vLLM 推理服务启动脚本（支持前台/后台/多卡/健康检查） |
| `start_rag.sh` | RAG 系统一键启动/停止/重启/状态查看脚本（管理 vLLM + web_app），支持启动前预检与 /api/health 就绪验证 |
| `auto_ingest.sh` | 知识库自动增量入库脚本（检测新增/修改文件，支持 cron 定时执行） |
| `convert_to_md.sh` | 文档格式转换脚本（PDF/DOCX/PPTX → Markdown，支持 MinerU/Marker/Docling） |
| `templates/index.html` | 对话前端页面 |
| `templates/login.html` | 登录页面 |

#### 未使用 / 历史遗留文件

| 文件 | 说明 |
|------|------|
| `rag_core.py` | 早期 RAG 核心逻辑（已被 `rag_agent.py` 替代） |
| `rag_tool.py` | 早期 RAG 工具封装（已被 `rag_agent.py` 替代） |
| `tools.py` | LangChain 工具定义（配合 `agent_entry.py` 使用，当前未启用） |
| `agent_entry.py` | ReAct Agent 实现（当前未启用，路由逻辑已内置于 `rag_agent.py`） |
| `ingest_new.py` | 语义切分入库实验版本 |
| `ingest_fiass.py` | FAISS 本地向量库入库（已改用 Qdrant） |
| `test.py` | 环境连通性检查脚本 |
| `test_qdrant_conn.py` | Qdrant 连接测试 |
| `app_test.py` | 应用测试 |
| `test_html.py` | HTML 页面测试脚本 |
| `mock_server.py` | 前端预览用 Mock SSE 服务器（模拟流式响应，便于前端独立开发调试） |
