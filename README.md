# LAB 403 RAG 知识库系统

> Author：youngerrwei（韦子扬）<br>
> 当前版本：v2.0.2

## 项目概述

实验室知识库 RAG 系统，支持论文、实验室规范、使用教程等文档的智能问答。基于 Small-to-Big 检索策略，结合混合检索（Dense + Sparse）与 CrossEncoder 重排序，提供高质量的知识检索与流式生成服务。

## 系统架构

四卡 GPU 环境（GPU2: Embedding + Reranker, GPU3: vLLM 推理），核心组件：

- **vLLM**：Qwen3-8B-Instruct 推理服务（端口 8000）
- **Flask Web 应用**：用户认证 + SSE 流式问答（端口 5000）
- **Qdrant**：向量数据库，存储子块集合 + 父块集合
- **bge-m3 / bge-reranker-v2-m3**：Embedding 与重排序模型

> 详细架构设计、数据流和技术决策请参阅 [ARCHITECTURE.md](ARCHITECTURE.md)

### 环境架构

项目采用三个独立的 conda 环境，避免 vLLM 和 MinerU 的 PyTorch/paddlepaddle 版本冲突：

| 环境名 | 用途 | 核心组件 | 使用的 GPU |
|--------|------|----------|------------|
| `rag-vllm` | vLLM 推理服务 | Python 3.10 + PyTorch 2.5.1+cu124 + vLLM 0.8.5.post1 | GPU 3 |
| `rag-mineru` | 文档格式转换 | Python 3.10 + PyTorch 2.5.1+cu124 + MinerU | GPU 0 |
| `rag` | RAG 主应用 | Flask + LangChain + bge-m3 + bge-reranker | GPU 2 |

> 所有管理脚本会自行定位 conda 并使用 `.env` 指定的环境，不依赖调用者当前激活的环境。

## 快速开始

### 环境准备

> **推荐**：可使用一键环境准备脚本完成以下所有步骤：
>
> ```bash
> bash setup_env.sh
> ```
>
> 该脚本会创建或验证三个 conda 环境、安装对应依赖、从 `.env.example` 初始化本机 `.env`、自动生成随机 Flask 密钥，并输出环境检查报告。支持 `--vllm`、`--mineru`、`--rag`、`--skip-vllm`、`--skip-mineru` 和 `--force`，详见 `bash setup_env.sh --help`。

#### 手动环境准备

> **重要**：vLLM 和 MinerU **必须**在独立环境中安装，不能混合，否则会产生 PyTorch/paddlepaddle 版本冲突。

```bash
# ─── 环境 1：rag（RAG 主服务）───
conda create -n rag python=3.10 -y
conda activate rag
pip install flask python-dotenv langchain langchain-openai langchain-huggingface \
    langchain-qdrant qdrant-client sentence-transformers tiktoken modelscope

# ─── 环境 2：rag-vllm（推理服务）───
conda create -n rag-vllm python=3.10 -y
conda activate rag-vllm
pip install vllm==0.8.5.post1  # CUDA 12.4；12.6+ 用 pip install vllm

# ─── 环境 3：rag-mineru（文档转换）───
conda create -n rag-mineru python=3.10 -y
conda activate rag-mineru
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install uv && uv pip install -U "mineru[all]"

# ─── 下载模型 ───
bash download_model.sh                    # 下载 Qwen3-8B（默认从 ModelScope）
# Embedding 和 Reranker 模型需手动放到 models/ 目录：
#   models/bge-m3
#   models/bge-reranker-v2-m3
```

### 首次部署完整流程

```bash
# Step 1: 创建/验证三套环境，并生成私有 .env
bash setup_env.sh

# Step 2: 编辑 .env，确认 Qdrant、文档目录、GPU 与本地模型路径
# .env 已被 Git 忽略；FLASK_SECRET_KEY 由 setup_env.sh 自动随机生成

# Step 3: 下载 LLM，并确认另外两个模型也位于配置指定目录
bash download_model.sh
# ./models/bge-m3
# ./models/bge-reranker-v2-m3

# Step 4: 在明确的 RAG 环境中创建登录用户
conda run -n rag python create_user.py

# Step 5: 转换文档为 Markdown（内部使用 rag-mineru 环境）
bash convert_to_md.sh --full

# Step 6: 先启动 vLLM（入库摘要增强需要它）
bash start_vllm.sh --background

# Step 7: 首次全量入库
bash auto_ingest.sh --full

# Step 8: 一键启动系统；已健康的 vLLM 会被复用，只启动 Web
bash start_rag.sh start

# Step 9: 两种方式验证状态
bash start_rag.sh status
curl http://127.0.0.1:5000/api/health
```

正常完整就绪时健康接口返回 `status=ok`。Qdrant 暂时不可达、必要的子块/父块集合缺失，或 RAG 模型仍在后台预热时返回 HTTP 200 / `degraded`；vLLM 不可用时返回 HTTP 503 / `error`。入库摘要失败会按原有降级策略继续使用无摘要内容。

完成首次部署后，日常启动只有一条命令：

```bash
bash start_rag.sh start
```

### 日常运维流程

```bash
# 新增文档后：转换 + 增量入库
bash convert_to_md.sh          # 仅转换新增/修改的文档
bash auto_ingest.sh            # 有变化时执行幂等增量覆盖

# 服务管理
bash start_rag.sh status       # 查看服务状态
bash start_rag.sh restart      # 重启 web_app（vLLM 保持运行）
bash start_rag.sh stop         # 停止 web_app（vLLM 保持运行）
bash start_rag.sh stop --all   # 同时停止 web_app + vLLM

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

- **运行环境**：脚本在 `rag-mineru` 环境中执行（内部自动处理，用户无需手动激活）
- **引擎优先级**：MinerU > Marker > Docling（自动检测已安装引擎）
- **增量逻辑**：基于 `data/.convert_state` 时间戳判断文件是否需要重新转换
- **文件锁**：`/tmp/convert_to_md.lock`，防止并发执行
- **支持格式**：`.pdf`, `.docx`, `.doc`, `.pptx`, `.ppt`（`.xlsx` 仅 MinerU 支持）
- **日志文件**：`logs/convert_to_md.log`
- **失败语义**：输出先写同目录临时文件并校验，再原子替换；任一文件失败时脚本返回非零，已成功文件保留

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

- **文件锁**：`data/.auto_ingest.lock`，防止并发执行
- **日志文件**：`logs/auto_ingest.log`
- 失败时不更新状态文件，下次运行自动重试
- 增量模式自动设置 `QDRANT_RECREATE_COLLECTION=false`
- 子块或父块任一批失败都会令任务失败；成功覆盖后会清理同源旧 point
- 配合 cron 实现定时入库：
  ```bash
  0 3 * * * bash /path/to/403_RAG/auto_ingest.sh >> /path/to/403_RAG/logs/auto_ingest_cron.log 2>&1
  ```

---

### start_rag.sh — 服务启动管理

一键管理 vLLM 推理服务 + Flask Web 应用的生命周期。

#### 子命令

| 命令 | 说明 |
|------|------|
| `start` | 预检 → 启动 vLLM → 启动 web_app |
| `stop` | 仅停止 web_app（vLLM 保持运行） |
| `stop --all` | 同时停止 web_app + vLLM |
| `restart` | 仅重启 web_app（vLLM 保持运行） |
| `restart --all` | 重启 web_app + vLLM |
| `status` | 使用认证健康检查显示 vLLM 与 Web 状态 |

#### 使用示例

```bash
bash start_rag.sh start          # 启动全部服务
bash start_rag.sh stop           # 仅停止 web_app（vLLM 保持运行）
bash start_rag.sh stop --all     # 同时停止 web_app + vLLM
bash start_rag.sh restart        # 仅重启 web_app（vLLM 保持运行）
bash start_rag.sh restart --all  # 重启 web_app + vLLM
bash start_rag.sh status         # 查看运行及健康状态
```

#### 启动预检逻辑（preflight_check）

| 检查项 | 失败行为 |
|--------|----------|
| `.env` 文件存在性 | **中止启动** |
| `FLASK_SECRET_KEY` 为空、过短或为公开占位值 | **中止启动** |
| 模型目录存在性（vLLM / Embedding / Reranker） | **中止启动** |
| `flock`、conda 或指定环境解释器不可用 | **中止启动** |
| 文档目录 | 不存在或为空时警告；为空通常表示共享盘尚未挂载 |
| Qdrant 连通性及子块/父块集合 | 不可达或集合缺失时警告并以 `degraded` 启动；集合缺失需执行全量入库 |
| 日志/数据目录 | 自动创建 |

#### 内部行为

- **vLLM 启动**：调用一次 `start_vllm.sh --background`；同时验证 `/health`、带 API Key 的 `/v1/models` 及模型名
- **vLLM 就绪**：最长等待 300 秒；超时会安全停止本次创建的进程并返回非零
- **web_app 启动**：使用 `RAG_CONDA_ENV` 对应解释器执行 `web_app.py`
- **web_app 就绪**：只以 `/api/health` 的 `ok/degraded` 为准，最多等待 60 秒
- **PID 管理**：`data/.vllm.pid` 和 `data/.web_app.pid`

---

### start_vllm.sh — vLLM 推理服务

独立管理 vLLM 推理服务的启动，**在 `rag-vllm` 环境中执行**。

#### 子命令

| 命令 | 说明 |
|------|------|
| （无子命令） | 启动 vLLM（默认前台模式） |
| `stop` | 停止受 PID 文件管理的 vLLM（先 SIGTERM，超时后 SIGKILL） |
| `status` | 查看 vLLM 运行状态、健康检查和模型信息 |

#### 启动参数

| 参数 | 说明 |
|------|------|
| `--gpu N` | 指定 GPU（如 `--gpu 0` 或 `--gpu 0,1`），覆盖 .env 中的配置 |
| `--background` | 后台运行，日志输出到 `logs/vllm_server.log` |

#### 使用示例

```bash
# 后台启动（start_rag.sh 内部调用方式）
bash start_vllm.sh --background

# 前台启动（调试用）
bash start_vllm.sh

# 指定 GPU 后台启动
bash start_vllm.sh --gpu 0,1 --background

# 停止 vLLM
bash start_vllm.sh stop

# 查看状态
bash start_vllm.sh status
```

#### 注意事项

- **环境依赖**：启动前须确保 `rag-vllm` 环境已由 `setup_env.sh --vllm` 创建
- **端口检查**：只复用健康且模型匹配的服务；模型标识兼容配置相对路径与服务返回绝对路径，对身份不明的占用进程拒绝停止或覆盖
- **健康检查**：轮询 `/health` + 带 API Key 的 `/v1/models`，默认 300 秒；超时严格返回非零
- **PID 文件**：后台模式保存到 `data/.vllm.pid`
- **模型路径**：本地路径不存在时提示运行 `download_model.sh`
- **端口**：默认 8000（由 .env 中 `VLLM_PORT` 控制）

## 环境配置（.env）

仓库只提交 `.env.example`；`setup_env.sh` 会创建被 Git 忽略的本机 `.env` 并生成随机 Flask 密钥。关键配置如下：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `VLLM_CUDA_DEVICES` | `3` | vLLM GPU 设备号 |
| `RAG_CONDA_ENV` | `rag` | Web、入库与下载工具环境 |
| `VLLM_CONDA_ENV` | `rag-vllm` | vLLM 环境 |
| `MINERU_CONDA_ENV` | `rag-mineru` | 文档转换环境 |
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
| `QDRANT_RECREATE_COLLECTION` | `false` | 常规运行禁止隐式重建；全量脚本会临时覆盖为 true |
| `VLLM_STARTUP_TIMEOUT` | `300` | vLLM 就绪等待秒数 |
| `WEBAPP_STARTUP_TIMEOUT` | `60` | Web 就绪等待秒数 |
| `FLASK_SECRET_KEY` | 首次安装随机生成 | Session 签名密钥，至少 32 字符且不得提交 |

> **重要**：所有模型统一使用本地路径（`models/`），不要使用 HuggingFace 在线路径。

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

增量入库使用确定性 UUID（基于 source + parent_id + chunk_index 的 UUID5）。新子块和父块全部写入成功后，系统清理同源但不再出现的旧 ID，避免文档修改后残留陈旧向量。

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
- **原因**：`models/` 下缺少对应模型文件
- **解决**：运行 `bash download_model.sh` 下载 LLM 模型；Embedding/Reranker 模型需手动放置

**现象**：vLLM 启动超时（默认 300 秒）
- **原因**：GPU 显存不足或模型文件损坏
- **解决**：用 `nvidia-smi` 检查 GPU 显存；确认 `VLLM_GPU_UTIL` 未设过高；检查模型文件完整性；可在 `.env` 中设置 `VLLM_STARTUP_TIMEOUT` 调整超时时间

**现象**：web_app 启动后 `/api/health` 返回 `"error"`
- **原因**：关键 vLLM 服务不可用或鉴权配置不一致
- **解决**：执行 `bash start_vllm.sh status`，核对 `VLLM_API_KEY`、模型名和 `logs/vllm_server.log`；Qdrant 单独不可达时应为 `degraded`

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

### 前端加载问题

**校徽不显示**：检查 CSP 头是否包含 `img-src 'self' data:;`

**Lucide 图标不显示**：确认服务器可访问 https://unpkg.com；图标加载失败时会显示文字 fallback

**暗色模式异常**：确认浏览器支持 `prefers-color-scheme` 媒体特性；尝试清除 localStorage 后刷新

### 环境类问题

**现象**：`convert_to_md.sh` 报 "conda 环境 'rag-mineru' 不存在"
- **原因**：未运行 setup_env.sh 创建环境
- **解决**：运行 `bash setup_env.sh --mineru`

**现象**：`start_vllm.sh` 报 "conda 环境 'rag-vllm' 不存在"
- **原因**：未运行 setup_env.sh 创建环境
- **解决**：运行 `bash setup_env.sh --vllm`

**现象**：setup_env.sh 安装时报 conda 参数不识别
- **原因**：系统 conda 版本过旧
- **解决**：更新 conda：`conda update -n base conda`

## 安全加固

| 防护项 | 实现方式 |
|--------|----------|
| Session 安全 | HttpOnly + SameSite=Strict；HTTPS 模式下启用 Secure；登录后重生成 Session ID |
| 速率限制 | 登录接口 5 次/分钟/IP |
| 并发控制 | Semaphore 限制最大并发请求数（默认 20） |
| 信息泄露防护 | 异常响应统一通用错误提示，不暴露内部信息 |
| SSE 连接保护 | 超时 5 分钟断开，每 15 秒心跳保活 |
| 安全响应头 | CSP 限制资源加载来源（`script-src` 含 `https://unpkg.com` Lucide CDN；`img-src` 含 `data:` base64 校徽） |
| 请求限制 | 请求体大小 1MB |
| 密钥检测 | 启动时检测 SECRET_KEY 强度 |

## 文件结构

#### 当前在用文件

| 文件 | 用途 |
|------|------|
| `.env.example` | 可提交的完整配置模板；本机 `.env` 私有且被 Git 忽略 |
| `rag_agent.py` | 核心 RAG 流程：两级查询路由（规则+LLM）、问题改写、多路并行检索（Dense+Sparse）、RRF 融合、重排序、父块展开（独立 Collection 查询）、TTL 缓存、流式回答生成、文件系统工具 |
| `web_app.py` | Flask 后端：登录认证（PBKDF2）、SSE 流式问答接口、历史管理 |
| `ingest.py` | 知识库入库：Markdown 加载、标题结构切分、父子块切分（Small-to-Big）、质量过滤、批量写入 Qdrant |
| `create_user.py` | 用户账号创建脚本 |
| `scripts/runtime_common.sh` | Shell 公共运行库：安全加载 `.env`、解析项目相对路径、定位 Conda、检查端口/PID/HTTP 与 vLLM 模型身份 |
| `setup_env.sh` | 环境准备一键脚本（依赖安装、模型检查、配置初始化） |
| `start_vllm.sh` | vLLM 推理服务管理脚本（支持启动/停止/状态查看、前台/后台/多卡/健康检查） |
| `start_rag.sh` | RAG 系统一键启动/停止/重启/状态查看脚本（管理 vLLM + web_app），支持启动前预检与 /api/health 就绪验证 |
| `auto_ingest.sh` | 知识库自动增量入库脚本（检测新增/修改文件，支持 cron 定时执行） |
| `convert_to_md.sh` | 文档格式转换脚本（PDF/DOCX/PPTX → Markdown，支持 MinerU/Marker/Docling） |
| `rag_tool.py` | ReAct 工具兼容适配器；复用 `rag_agent.py` 的唯一 RAG 运行时 |
| `tools.py` | `use_agent=true` 时使用的请求级 ReAct 工具定义 |
| `agent_entry.py` | `use_agent=true` 时由 Web 接口调用的 ReAct Agent 入口 |
| `test_reliability.py` | 启动、健康检查、SSE、并发释放、入库一致性与脚本语法的可靠性回归测试 |
| `templates/index.html` | 对话前端页面（iOS 毛玻璃风格、暗色模式、中山大学校徽内联、Lucide 图标） |
| `templates/login.html` | 登录页面（绿白渐变、校徽内联、制作者署名） |

#### 未使用 / 历史遗留文件

| 文件 | 说明 |
|------|------|
| `rag_core.py` | 早期 RAG 核心逻辑（已被 `rag_agent.py` 替代） |
| `ingest_new.py` | 语义切分入库实验版本 |
| `ingest_fiass.py` | FAISS 本地向量库入库（已改用 Qdrant） |
| `test.py` | 环境连通性检查脚本 |
| `test_qdrant_conn.py` | Qdrant 连接测试 |
| `app_test.py` | 应用测试 |
| `test_html.py` | HTML 页面测试脚本 |
| `mock_server.py` | 前端预览用 Mock SSE 服务器（模拟流式响应，便于前端独立开发调试） |
