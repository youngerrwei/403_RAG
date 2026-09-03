# 故障排查指南

本文档为 LAB 403 RAG 知识库系统的故障排查指南，覆盖核心链路（vLLM 启动 → 知识入库 → Web 问答）中常见问题的诊断与解决方案。

---

## 日志系统概况

### 日志文件位置

| 模块 | 文件路径 | 用途 |
|------|---------|------|
| RAG Agent | `logs/rag_agent.log` | 核心问答引擎日志（路由、检索、重排、生成） |
| Web 服务 | `logs/rag_web.log` | Flask 服务日志（登录、SSE、预热） |
| 入库模块 | `logs/rag_ingest.log` | 文档入库日志（加载、切分、向量化、写入） |
| vLLM 服务 | `logs/vllm_server.log` | LLM 推理服务日志（仅后台模式） |
| 自动入库 | `logs/auto_ingest.log` | 增量入库脚本执行日志 |

### 日志格式

```
%(asctime)s | %(levelname)-5s | %(name)s | %(message)s
```

示例：
```log
2025-01-15 14:23:45 | DEBUG | agent | 父块展开: 8 子块 → 3 父块 (总长=1850)
2025-01-15 14:23:46 | INFO  | web   | SSE 连接建立: user=admin, ip=192.168.1.100
2025-01-15 14:23:47 | ERROR | ingest| Qdrant 连接失败: Connection refused
```

### 日志级别配置

通过 `.env` 文件控制：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LOG_LEVEL` | `DEBUG` | 日志级别（DEBUG/INFO/WARNING/ERROR） |
| `LOG_DIR` | `logs` | 日志目录路径 |
| `LOG_FILE_PREFIX` | `rag` | 日志文件名前缀 |
| `LOG_MAX_DAYS` | `7` | 日志保留天数（按天轮转） |
| `DEBUG_MODE` | `true` | 调试模式开关（影响额外调试输出） |

---

## 场景一：vLLM 启动失败

### 问题描述

执行 `bash start_vllm.sh` 或 `bash start_rag.sh start` 后，vLLM 服务未能成功启动，健康检查超时。

### 排查命令

```bash
# 查看 vLLM 日志（实时）
tail -f logs/vllm_server.log

# 搜索错误关键词
grep -i "error\|CUDA\|OOM\|out of memory\|Address already in use" logs/vllm_server.log

# 检查端口占用
lsof -i :8000

# 检查 GPU 状态
nvidia-smi

# 检查 vLLM 进程是否存在
ps aux | grep vllm
```

### 常见错误日志示例

**显存不足（OOM）：**
```log
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**端口被占用：**
```log
OSError: [Errno 98] Address already in use
```

**模型路径错误：**
```log
OSError: Qwen/Qwen2.5-7B-Instruct is not a local folder and is not a valid model identifier
```

**CUDA 环境问题：**
```log
RuntimeError: No CUDA GPUs are available
```

### 解决方案

| 问题 | 解决方案 |
|------|---------|
| 显存不足 | 降低 `VLLM_GPU_UTIL`（当前 0.85），或减小 `VLLM_MAX_MODEL_LEN`（当前 6000） |
| 端口占用 | `kill $(lsof -ti :8000)` 或修改 `.env` 中的 `VLLM_PORT` |
| 模型路径错误 | 确认模型已下载到本地，检查 `VLLM_MODEL_NAME` 路径是否正确 |
| CUDA 不可用 | 检查 `VLLM_CUDA_DEVICES=3` 配置，确认 `nvidia-smi` 能正常显示 GPU |
| 环境未安装 | 确认 vLLM 已安装：`pip show vllm` |

---

## 场景二：知识入库失败

### 问题描述

执行 `bash auto_ingest.sh` 或 `python ingest.py` 后，文档未能成功写入 Qdrant 集合。

### 排查命令

```bash
# 查看入库日志（实时）
tail -f logs/rag_ingest.log

# 查看自动入库脚本日志
tail -f logs/auto_ingest.log

# 检查 Qdrant 服务是否可达
curl http://172.18.216.71:6333/health

# 查看集合状态
curl http://172.18.216.71:6333/collections/lab_knowledge_base
curl http://172.18.216.71:6333/collections/lab_knowledge_base_parents

# 验证 Embedding 模型路径
ls -la models/bge-m3

# 验证文档目录
ls -la /mnt/cpu_share

# 文件加载统计
grep "加载" logs/rag_ingest.log | tail -20
grep "SUCCESS\|ERROR" logs/auto_ingest.log | tail -20
```

### 常见错误日志示例

**Qdrant 连接失败：**
```log
2025-01-15 14:00:01 | ERROR | ingest| Qdrant 连接失败: Connection refused (172.18.216.71:6333)
```

**Embedding 模型加载失败：**
```log
2025-01-15 14:00:02 | ERROR | ingest| 模型加载失败: models/bge-m3 路径不存在
```

**文档路径不存在：**
```log
[2025-01-15 14:00:03] ERROR: DOCS_PATH 目录不存在: /mnt/cpu_share
```

**集合维度不匹配：**
```log
ValueError: Expected vector size 1024, got 768
```

### 解决方案

| 问题 | 解决方案 |
|------|---------|
| DOCS_PATH 不存在 | 确认 `.env` 中的 `DOCS_PATH=/mnt/cpu_share` 路径存在且有读权限 |
| Qdrant 连接失败 | 检查远程 Qdrant 服务状态，确认网络连通性：`ping 172.18.216.71` |
| Embedding 模型加载失败 | 确认 `models/bge-m3` 目录存在且包含完整模型文件 |
| 集合维度不匹配 | 执行全量重建：`bash auto_ingest.sh --destroy --force && bash auto_ingest.sh --full` |
| 文件锁冲突 | 检查 `/tmp/auto_ingest.lock` 是否残留：`rm -f /tmp/auto_ingest.lock` |
| GPU 设备错误 | 确认 `EMBEDDING_DEVICE=cuda:2` 可用：`python -c "import torch; print(torch.cuda.device_count())"` |

---

## 场景三：Web 应用启动失败

### 问题描述

执行 `python web_app.py` 或通过 `start_rag.sh` 启动 Web 服务失败，无法访问 `http://127.0.0.1:5000`。

### 排查命令

```bash
# 查看 Web 服务日志
tail -f logs/rag_web.log

# 检查端口占用
lsof -i :5000

# 检查运行时预热状态
grep "预热" logs/rag_web.log

# 检查 SECRET_KEY 配置
grep "FLASK_SECRET_KEY" .env

# 验证模块导入是否正常
python -c "import web_app"
```

### 常见错误日志示例

**端口占用：**
```log
OSError: [Errno 98] Address already in use: ('0.0.0.0', 5000)
```

**运行时预热失败：**
```log
2025-01-15 14:00:05 | ERROR | web   | 运行时预热失败: Embedding 模型加载超时
```

**依赖导入失败：**
```log
ModuleNotFoundError: No module named 'flask'
```

### 解决方案

| 问题 | 解决方案 |
|------|---------|
| 端口占用 | `kill $(lsof -ti :5000)` 或修改监听端口 |
| 运行时预热失败 | 检查 Embedding/Reranker 模型路径和 GPU 可用性 |
| SECRET_KEY 未设置 | 确认 `.env` 中 `FLASK_SECRET_KEY` 已设置有效值 |
| 依赖缺失 | 执行 `pip install flask python-dotenv` 安装必要依赖 |
| rag_agent 导入失败 | 检查 `rag_agent.py` 语法：`python -c "import ast; ast.parse(open('rag_agent.py').read())"` |

---

## 场景四：用户问答异常

### 问题描述

用户在 Web 界面提问后无回复、回复为空、超时断开，或检索结果不相关。

### 排查命令

```bash
# 实时监听 RAG Agent 日志
tail -f logs/rag_agent.log

# 检索无结果
grep "\[RECALL\].*docs=0" logs/rag_agent.log

# 生成耗时过长
grep "total_elapsed" logs/rag_agent.log | tail -10

# SSE 连接断开
grep "SSE.*断开\|GeneratorExit" logs/rag_web.log

# 缓存命中情况
grep "\[CACHE HIT\]" logs/rag_agent.log | tail -10

# 路由判断
grep "route=" logs/rag_agent.log | tail -10

# 重排序结果
grep "rerank" logs/rag_agent.log | tail -10
```

### 关键日志标记说明

| 标记 | 含义 |
|------|------|
| `[RECALL]` | 向量检索阶段，关注 `docs=N` 表示召回文档数 |
| `[CACHE HIT]` | 查询缓存命中，跳过检索直接返回 |
| `ask_stream done` | 一次完整问答流程结束 |
| `total_elapsed` | 总耗时统计 |
| `route=` | 路由判断结果（knowledge/chat/tool） |

### 常见错误日志示例

**知识库为空：**
```log
2025-01-15 14:30:00 | WARNING | agent | [RECALL] 检索结果为空, docs=0, query="什么是光通信"
```

**vLLM 超时：**
```log
2025-01-15 14:30:05 | ERROR | agent | LLM 生成超时: timeout=30s, elapsed=31.2s
```

**SSE 连接断开：**
```log
2025-01-15 14:30:10 | WARNING | web   | SSE 连接断开: GeneratorExit, user=admin
```

### 解决方案

| 问题 | 解决方案 |
|------|---------|
| 知识库为空 | 确认已执行入库：`curl http://172.18.216.71:6333/collections/lab_knowledge_base` 查看 `points_count` |
| vLLM 超时 | 检查 vLLM 服务状态、GPU 负载，或增大超时配置 |
| 连接断开 | 客户端网络问题或服务端负载过高，检查并发量 |
| 检索不相关 | 检查 Reranker 分数阈值 `RERANKER_SCORE_THRESHOLD=0.3`，适当降低 |
| 回复为空 | 检查 `MAX_CONTEXT_CHARS` 和 `RESPONSE_MAX_TOKENS` 配置 |

---

## 场景五：GPU/显存问题

### 问题描述

GPU 显存溢出导致服务崩溃，或 GPU 利用率异常。

### 排查命令

```bash
# 实时 GPU 监控
nvidia-smi -l 1

# 查看 GPU 详细信息
nvidia-smi -q

# 检查 GPU 配置
grep "DEVICE\|GPU_UTIL\|CUDA" .env

# 搜索显存溢出日志
grep -i "out of memory\|OOM\|CUDA error" logs/rag_*.log logs/vllm_server.log

# 查看 GPU 进程
nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
```

### GPU 分配策略

当前配置：

| GPU | 用途 | 配置项 |
|-----|------|--------|
| GPU 3 | vLLM 推理服务 | `VLLM_CUDA_DEVICES=3`，显存利用率 85% |
| GPU 2 | Embedding + Reranker | `EMBEDDING_DEVICE=cuda:2`，`RERANKER_DEVICE=cuda:2` |

### 常见问题及调整方案

| 问题 | 现象 | 调整方案 |
|------|------|---------|
| vLLM OOM | vLLM 进程崩溃，日志出现 CUDA OOM | 降低 `VLLM_GPU_UTIL`（如 0.80）或减小 `VLLM_MAX_MODEL_LEN`（如 4096） |
| Embedding OOM | 入库/检索时 GPU 2 显存溢出 | 减小 `INGEST_BATCH_SIZE`（当前 64，可降至 32） |
| GPU 设备冲突 | 两个模型加载到同一 GPU | 确认 `VLLM_CUDA_DEVICES` 和 `EMBEDDING_DEVICE` 指向不同设备 |
| 多用户并发 OOM | 并发请求过多导致显存耗尽 | 降低 `MAX_CONCURRENT_REQUESTS`（当前 20） |

---

## 场景六：认证/登录问题

### 问题描述

用户无法登录系统，出现密码错误、IP 锁定等情况。

### 排查命令

```bash
# 查看登录失败日志
grep "登录失败\|锁定" logs/rag_web.log

# 查看 IP 限流情况
grep "速率限制\|rate limit" logs/rag_web.log

# 检查用户配置文件
cat data/users.json

# 查看近期登录活动
grep "登录" logs/rag_web.log | tail -20
```

### 认证机制说明

| 参数 | 值 | 说明 |
|------|-----|------|
| 速率限制 | 5次/分钟 | 同一 IP 每分钟最多 5 次登录尝试 |
| 锁定阈值 | 5 次失败 | 连续 5 次失败后锁定 IP |
| 锁定时长 | 15 分钟 | 锁定 900 秒后自动解除 |

### 常见错误日志示例

**密码错误：**
```log
2025-01-15 14:00:10 | WARNING | web   | 登录失败: user=admin, ip=192.168.1.100, reason=密码错误
```

**IP 被锁定：**
```log
2025-01-15 14:00:15 | WARNING | web   | 登录拒绝: ip=192.168.1.100, reason=登录尝试过多，请14分钟后重试
```

### 解决方案

| 问题 | 解决方案 |
|------|---------|
| 密码错误 | 使用 `python create_user.py` 重置用户密码 |
| IP 被锁定 | 等待 15 分钟自动解锁，或重启 Web 服务清除内存中的锁定状态 |
| users.json 不存在 | 执行 `python create_user.py` 创建用户文件 |
| SESSION 失效 | 检查 `FLASK_SECRET_KEY` 是否变更（变更后所有 session 失效） |

---

## 场景七：并发/性能问题

### 问题描述

系统在高并发下响应变慢、请求排队或超时。

### 排查命令

```bash
# 查看并发配置
grep "MAX_CONCURRENT" .env

# 统计当前活跃请求数
grep "SSE 连接建立" logs/rag_web.log | wc -l
grep "SSE.*断开\|ask_stream done" logs/rag_web.log | wc -l

# 性能分析：各阶段耗时提取
grep "total_elapsed" logs/rag_agent.log | tail -20

# 检索阶段耗时
grep "recall.*elapsed\|检索耗时" logs/rag_agent.log | tail -10

# 重排序耗时
grep "rerank.*elapsed\|重排耗时" logs/rag_agent.log | tail -10

# 生成阶段耗时
grep "generate.*elapsed\|生成耗时" logs/rag_agent.log | tail -10

# 查看信号量等待（并发瓶颈）
grep "Semaphore\|等待" logs/rag_web.log | tail -10
```

### 并发相关配置

| 配置项 | 当前值 | 说明 |
|--------|--------|------|
| `MAX_CONCURRENT_REQUESTS` | 20 | 最大并发请求数 |
| `VLLM_MAX_MODEL_LEN` | 6000 | 影响 vLLM 并发吞吐 |
| `VLLM_GPU_UTIL` | 0.85 | GPU 显存利用率 |
| `VLLM_ENABLE_PREFIX_CACHING` | true | 启用前缀缓存提升并发效率 |

### 性能优化建议

| 瓶颈 | 诊断方法 | 优化方案 |
|------|---------|---------|
| 检索慢 | 检索耗时 > 2s | 减小 `INITIAL_RETRIEVAL_K`（当前 64） |
| 重排慢 | 重排耗时 > 3s | 减小 `FINAL_TOP_K`（当前 16） |
| 生成慢 | 生成耗时 > 15s | 减小 `RESPONSE_MAX_TOKENS`（当前 2048） |
| 并发排队 | 请求长时间无响应 | 增大 `MAX_CONCURRENT_REQUESTS` 或水平扩展 |
| 首 token 延迟高 | 确认 prefix caching 已启用 | 设置 `VLLM_ENABLE_PREFIX_CACHING=true` |

---

## 场景八：前端样式/功能异常

### 问题描述

校徽不显示、图标加载失败、按钮无法点击、暗色模式不生效。

### 排查步骤

1. 打开浏览器开发者工具（F12），检查控制台是否有 CSP 报错
2. 检查 Network 标签，确认 `unpkg.com` (Lucide) 和 `cdn.jsdelivr.net` (marked/DOMPurify/highlight.js/MathJax) 正常加载
3. 如校徽不显示，检查 CSP 头是否包含 `img-src 'self' data:`
4. 如按钮无法点击，检查是否有 JavaScript 错误阻止事件绑定
5. 清除浏览器缓存和 localStorage 后重试

### 解决方案

| 问题 | 解决方案 |
|------|----------|
| CSP 问题 | 确认 `web_app.py` 中 CSP 头正确设置 |
| CDN 不可达 | 检查服务器网络连通性（`curl https://unpkg.com`） |
| 暗色模式 | 浏览器需支持 CSS3 媒体特性，推荐 Chrome 76+ / Firefox 67+ / Safari 12.1+ |

---

## 快速诊断

一键检查所有服务状态的命令集合：

```bash
#!/bin/bash
echo "========== RAG 系统快速诊断 =========="
echo ""

# 1. 检查 vLLM 服务
echo ">>> [1/6] vLLM 服务状态"
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health | grep -q "200"; then
    echo "  ✓ vLLM 服务正常运行"
    curl -s http://127.0.0.1:8000/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  模型: {d[\"data\"][0][\"id\"]}')" 2>/dev/null || true
else
    echo "  ✗ vLLM 服务不可达"
fi
echo ""

# 2. 检查 Web 应用
echo ">>> [2/6] Web 应用状态"
if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:5000/login | grep -q "200"; then
    echo "  ✓ Web 应用正常运行"
else
    echo "  ✗ Web 应用不可达"
fi
echo ""

# 3. 检查 Qdrant
echo ">>> [3/6] Qdrant 连接状态"
if curl -s -o /dev/null -w "%{http_code}" http://172.18.216.71:6333/health | grep -q "200"; then
    echo "  ✓ Qdrant 服务正常"
    # 查看集合点数
    POINTS=$(curl -s http://172.18.216.71:6333/collections/lab_knowledge_base | python3 -c "import sys,json; d=json.load(sys.stdin); print(d['result']['points_count'])" 2>/dev/null || echo "未知")
    echo "  子块集合文档数: $POINTS"
else
    echo "  ✗ Qdrant 服务不可达"
fi
echo ""

# 4. 检查 GPU
echo ">>> [4/6] GPU 状态"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
else
    echo "  nvidia-smi 不可用"
fi
echo ""

# 5. 检查模型文件
echo ">>> [5/6] 模型文件检查"
[ -d "models/bge-m3" ] && echo "  ✓ Embedding 模型存在" || echo "  ✗ Embedding 模型不存在: models/bge-m3"
[ -d "models/bge-reranker-v2-m3" ] && echo "  ✓ Reranker 模型存在" || echo "  ✗ Reranker 模型不存在: models/bge-reranker-v2-m3"
echo ""

# 6. 检查最近错误
echo ">>> [6/6] 最近错误日志（最近 5 条）"
grep -h "ERROR" logs/rag_*.log logs/vllm_server.log 2>/dev/null | tail -5 || echo "  无错误日志"
echo ""

echo "========== 诊断完成 =========="
```

将以上内容保存为脚本文件使用：

```bash
bash start_rag.sh status
```

或直接逐条执行关键检查：

```bash
# 快速三步检查
curl -s http://127.0.0.1:8000/health && echo " vLLM OK" || echo " vLLM FAIL"
curl -s http://127.0.0.1:5000/login -o /dev/null -w "%{http_code}" && echo " Web OK" || echo " Web FAIL"
curl -s http://172.18.216.71:6333/health && echo " Qdrant OK" || echo " Qdrant FAIL"
```

---

## 日志级别调整

### 通过 .env 控制日志详细程度

编辑 `.env` 文件中的日志配置：

```bash
# 日志级别：DEBUG > INFO > WARNING > ERROR
# DEBUG: 输出所有日志，包括详细的检索过程、父块展开、embedding 向量等
# INFO: 输出关键流程节点信息（推荐生产环境）
# WARNING: 仅输出警告和错误
# ERROR: 仅输出错误信息
LOG_LEVEL=DEBUG

# 调试模式：控制额外的调试输出（如查询改写中间结果、向量距离等）
DEBUG_MODE=true
```

### 级别说明

| 级别 | 适用场景 | 输出内容 |
|------|---------|---------|
| `DEBUG` | 开发/排查问题 | 所有日志，包含向量距离、chunk 内容、路由判断详情 |
| `INFO` | 生产环境 | 请求流程、耗时统计、结果摘要 |
| `WARNING` | 关注异常 | 检索为空、超时警告、配置缺失 |
| `ERROR` | 仅关注故障 | 服务异常、连接失败、未捕获异常 |

### 动态调整（无需重启）

当前系统需要重启才能生效日志级别变更：

```bash
# 修改 .env 后重启 Web 服务
bash start_rag.sh restart
```

### 临时开启全量调试

排查问题时临时启用最详细的日志：

```bash
# 在 .env 中设置
LOG_LEVEL=DEBUG
DEBUG_MODE=true

# 重启服务使配置生效
bash start_rag.sh restart

# 排查完成后恢复
LOG_LEVEL=INFO
DEBUG_MODE=false
```

---

## 日志清理

### 自动轮转机制

系统使用 `TimedRotatingFileHandler`，日志按天自动轮转：

- 轮转时间：每天午夜（midnight）
- 保留天数：由 `LOG_MAX_DAYS` 控制（默认 7 天）
- 轮转后文件命名：`rag_agent.log.2025-01-14`

### 手动清理命令

```bash
# 查看日志目录大小
du -sh logs/

# 查看各日志文件大小
ls -lh logs/

# 清理 7 天前的旧日志
find logs/ -name "*.log.*" -mtime +7 -delete

# 清理所有历史日志（保留当前日志）
find logs/ -name "*.log.*" -delete

# 清空当前日志（不删除文件，服务无需重启）
> logs/rag_agent.log
> logs/rag_web.log
> logs/rag_ingest.log
> logs/vllm_server.log
```

### 磁盘空间不足时的紧急处理

```bash
# 1. 检查日志占用
du -sh logs/*

# 2. 清空最大的日志文件
> logs/vllm_server.log

# 3. 删除所有历史轮转日志
find logs/ -name "*.log.*" -delete

# 4. 调高日志级别减少输出
# 修改 .env: LOG_LEVEL=WARNING
# 然后重启服务
bash start_rag.sh restart
```

### 定时清理（Cron）

```bash
# 每天凌晨 2 点清理 7 天前的日志
0 2 * * * find /Users/weiziyang/Documents/code/403_RAG/logs/ -name "*.log.*" -mtime +7 -delete
```

---

## 附录：常用排查命令速查表

| 场景 | 命令 |
|------|------|
| 查看服务状态 | `bash start_rag.sh status` |
| 重启所有服务 | `bash start_rag.sh restart` |
| 全量重建知识库 | `bash auto_ingest.sh --destroy --force && bash auto_ingest.sh --full` |
| 检查 Python 语法 | `python -c "import ast; ast.parse(open('文件名.py').read())"` |
| 实时 GPU 监控 | `nvidia-smi -l 1` |
| 搜索所有错误日志 | `grep -h "ERROR" logs/rag_*.log` |
| 统计今日请求量 | `grep "$(date +%Y-%m-%d)" logs/rag_web.log \| grep "SSE 连接建立" \| wc -l` |
| 查看最近异常 | `tail -50 logs/rag_agent.log \| grep -i "error\|warning"` |
