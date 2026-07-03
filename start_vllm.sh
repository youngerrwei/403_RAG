#!/bin/bash
set -e

# =============================================================================
# vLLM 模型服务启动脚本
# 适配：单卡 4090 24GB，知识库 QA 场景
# 运行环境：使用独立的 conda 环境 rag-vllm（避免与 MinerU 的依赖冲突）
# 用法：
#   前台运行：bash start_vllm.sh
#   后台运行：bash start_vllm.sh --background
#   指定 GPU：bash start_vllm.sh --gpu 1
#   多卡运行：bash start_vllm.sh --gpu 0,1
#   停止服务：bash start_vllm.sh stop
#   查看状态：bash start_vllm.sh status
# =============================================================================

# ----------------------------- 基础路径 ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

# ----------------------------- 读取 .env 配置 --------------------------------
# 提前加载配置，使 stop/status 子命令也能获取 VLLM_PORT 等信息
if [ -f "$ENV_FILE" ]; then
    # 解析 .env 文件：忽略注释和空行，导出变量
    while IFS='=' read -r key value; do
        # 跳过注释和空行
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        # 去除首尾空格
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        # 仅导出 VLLM 相关变量（避免污染环境）
        case "$key" in
            VLLM_*)
                export "$key=$value"
                ;;
        esac
    done < "$ENV_FILE"
else
    echo "[警告] 未找到 .env 文件：$ENV_FILE，将使用默认值"
fi

# ----------------------------- 设置默认值 ------------------------------------
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-models/Qwen3-8B-Instruct}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_CUDA_DEVICES="${VLLM_CUDA_DEVICES:-0}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
# 显存利用率：0.85 为单卡 4090 安全上限，预留显存给 KV cache 动态扩展
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
# 最大模型长度：6000 tokens 足够覆盖知识库 QA 场景（系统提示词 + 检索上下文 + 用户问题 + 生成）
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6000}"
# 前缀缓存：复用重复的系统提示词 KV cache，显著提升多轮/并发推理效率
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-true}"

# ----------------------------- 子命令：stop -----------------------------------
# 关闭 vLLM 服务，优先通过 PID 文件定位进程，回退到端口查找
do_stop() {
    local pid=""
    local pid_file="$SCRIPT_DIR/data/.vllm.pid"

    # 优先从 PID 文件获取
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "[信息] PID 文件中的进程 ($pid) 已不存在"
            pid=""
            rm -f "$pid_file"
        fi
    fi

    # PID 文件无效时，通过端口查找
    if [ -z "$pid" ]; then
        if command -v lsof &>/dev/null; then
            pid=$(lsof -ti :"${VLLM_PORT}" -sTCP:LISTEN 2>/dev/null | head -1)
        elif command -v ss &>/dev/null; then
            pid=$(ss -tlnp 2>/dev/null | grep ":${VLLM_PORT} " | grep -oP 'pid=\K[0-9]+' | head -1)
        fi
    fi

    if [ -z "$pid" ]; then
        echo "[信息] vLLM 未在运行"
        return 0
    fi

    echo "[信息] 正在停止 vLLM (PID: $pid)..."
    # 先发 SIGTERM，允许进程优雅退出
    kill "$pid" 2>/dev/null || true

    # 等待最多 5 秒
    local wait_count=0
    while kill -0 "$pid" 2>/dev/null && [ $wait_count -lt 5 ]; do
        sleep 1
        wait_count=$((wait_count + 1))
    done

    # 如果进程仍在运行，发送 SIGKILL 强制终止
    if kill -0 "$pid" 2>/dev/null; then
        echo "[信息] 进程未响应 SIGTERM，发送 SIGKILL 强制终止..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi

    # 清理 PID 文件
    rm -f "$pid_file"
    echo "[信息] vLLM 已停止"
}

# ----------------------------- 子命令：status ---------------------------------
# 查看 vLLM 服务运行状态和模型信息
do_status() {
    local pid=""
    local pid_file="$SCRIPT_DIR/data/.vllm.pid"
    local pid_source=""

    # 尝试从 PID 文件获取
    if [ -f "$pid_file" ]; then
        pid=$(cat "$pid_file")
        if kill -0 "$pid" 2>/dev/null; then
            pid_source="PID 文件"
        else
            pid=""
            rm -f "$pid_file"
        fi
    fi

    # PID 文件无效时，通过端口查找
    if [ -z "$pid" ]; then
        if command -v lsof &>/dev/null; then
            pid=$(lsof -ti :"${VLLM_PORT}" -sTCP:LISTEN 2>/dev/null | head -1)
        elif command -v ss &>/dev/null; then
            pid=$(ss -tlnp 2>/dev/null | grep ":${VLLM_PORT} " | grep -oP 'pid=\K[0-9]+' | head -1)
        fi
        if [ -n "$pid" ]; then
            pid_source="端口 ${VLLM_PORT}"
        fi
    fi

    if [ -z "$pid" ]; then
        echo "============================================="
        echo "  vLLM 状态: 未运行"
        echo "  监听端口: ${VLLM_PORT} (无进程)"
        echo "============================================="
        return 0
    fi

    echo "============================================="
    echo "  vLLM 状态: 运行中"
    echo "  PID:       $pid (来源: $pid_source)"
    echo "  监听端口:  ${VLLM_PORT}"
    echo "============================================="

    # 尝试获取健康状态
    local health_url="http://127.0.0.1:${VLLM_PORT}/health"
    local models_url="http://127.0.0.1:${VLLM_PORT}/v1/models"

    local health_code
    health_code=$(curl -s -o /dev/null -w "%{http_code}" "$health_url" 2>/dev/null || echo "000")

    if [ "$health_code" = "200" ]; then
        echo "  健康检查: 正常 (HTTP 200)"

        # 获取模型信息
        local model_response
        model_response=$(curl -s "$models_url" 2>/dev/null || echo "")
        if [ -n "$model_response" ]; then
            local model_id
            model_id=$(echo "$model_response" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('data') and len(data['data']) > 0:
        print(data['data'][0].get('id', '未知'))
    else:
        print('无模型加载')
except:
    print('解析失败')
" 2>/dev/null || echo "查询失败")
            echo "  加载模型: $model_id"
        fi
    else
        echo "  健康检查: 未就绪 (HTTP $health_code)"
        echo "  提示: 模型可能仍在加载中..."
    fi

    echo "============================================="
}

# ----------------------------- 子命令分发 -------------------------------------
# 如果第一个参数是 stop 或 status，执行对应操作后退出
# 这些操作不需要 conda 环境，只是管理进程
case "${1:-}" in
    stop)
        do_stop
        exit $?
        ;;
    status)
        do_status
        exit $?
        ;;
esac

# =============================================================================
# 以下为启动逻辑（需要 conda 环境）
# =============================================================================

# ----------------------------- 检查 rag-vllm conda 环境 -----------------------
# vLLM 运行在独立的 conda 环境中，避免与 MinerU/RAG 主环境的 PyTorch 版本冲突
CONDA_ENV_NAME="rag-vllm"

# 查找 conda 命令
CONDA_CMD=""
if command -v conda &>/dev/null; then
    CONDA_CMD="conda"
elif [[ -f "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_CMD="$HOME/miniconda3/bin/conda"
elif [[ -f "$HOME/anaconda3/bin/conda" ]]; then
    CONDA_CMD="$HOME/anaconda3/bin/conda"
fi

if [[ -z "$CONDA_CMD" ]]; then
    echo "[错误] 未找到 conda 命令，请先安装 Miniconda/Anaconda"
    exit 1
fi

# 检查 rag-vllm 环境是否存在
if ! $CONDA_CMD env list 2>/dev/null | grep -qE "^${CONDA_ENV_NAME}\s"; then
    echo "[错误] conda 环境 '${CONDA_ENV_NAME}' 不存在"
    echo "[提示] 请先运行安装脚本创建环境:"
    echo "        bash setup_env.sh --vllm"
    exit 1
fi

echo "[信息] 使用 conda 环境: ${CONDA_ENV_NAME}"

# 获取 conda 初始化脚本路径（用于后续在子 shell 中激活环境）
# 直接使用 conda activate 需要 conda init，但 source conda.sh 方式兼容所有版本
CONDA_BASE="$($CONDA_CMD info --base 2>/dev/null)"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "[错误] 无法找到 conda 初始化脚本: ${CONDA_SH}"
    exit 1
fi

# ----------------------------- 解析命令行参数 --------------------------------
BACKGROUND=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)
            # 支持 --gpu N 指定 GPU 设备号（覆盖 .env 配置）
            # 例如：bash start_vllm.sh --gpu 1
            #       bash start_vllm.sh --gpu 0,1  (多卡)
            #       bash start_vllm.sh --gpu 2 --background
            if [[ -n "$2" && "$2" != --* ]]; then
                VLLM_CUDA_DEVICES="$2"
                shift
            else
                echo "[错误] --gpu 参数需要指定 GPU ID，例如：--gpu 0 或 --gpu 0,1"
                exit 1
            fi
            ;;
        --background)
            BACKGROUND=true
            ;;
    esac
    shift
done

# 导出 CUDA 设备号（从 VLLM_CUDA_DEVICES 映射到 CUDA_VISIBLE_DEVICES）
export CUDA_VISIBLE_DEVICES="$VLLM_CUDA_DEVICES"

# ----------------------------- 模型存在性检查 --------------------------------
# 判断是否为本地路径（不含斜杠分隔的纯模型名如 Qwen/xxx 视为在线路径）
if [[ "$VLLM_MODEL_NAME" == ./* || "$VLLM_MODEL_NAME" == /* ]]; then
    # 本地路径，检查目录是否存在
    if [ ! -d "$VLLM_MODEL_NAME" ]; then
        echo "[错误] 模型路径不存在: $VLLM_MODEL_NAME"
        echo "[提示] 请先运行下载脚本获取模型:"
        echo "        bash download_model.sh"
        exit 1
    fi
fi

# ----------------------------- 打印配置摘要 ----------------------------------
echo "============================================="
echo "  vLLM 服务启动配置摘要"
echo "============================================="
echo "  模型路径:       $VLLM_MODEL_NAME"
echo "  监听地址:       $VLLM_HOST:$VLLM_PORT"
echo "  API 密钥:       ${VLLM_API_KEY:0:4}****"
echo "  GPU 设备:       VLLM_CUDA_DEVICES=$VLLM_CUDA_DEVICES (export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  显存利用率:     $VLLM_GPU_UTIL"
echo "  最大模型长度:   $VLLM_MAX_MODEL_LEN"
echo "  前缀缓存:       $VLLM_ENABLE_PREFIX_CACHING"
echo "============================================="

# ----------------------------- 构建启动命令 ----------------------------------
# 注意：不使用 conda run 包裹 vLLM 进程，原因：
#   1. conda run 会创建中间进程，导致 $! 获取的 PID 是 conda 进程而非 vLLM 进程
#   2. 信号无法正确传播到实际的 Python 进程，导致无法正常停止服务
#   3. 旧版 conda 的 conda run 不支持 --live-stream 等参数
# 改用方案：通过 source conda.sh && conda activate 激活环境后 exec python，
#   exec 会替换当前 shell 进程为 python，使 $! 直接指向 vLLM Python 进程
VLLM_ARGS="python -m vllm.entrypoints.openai.api_server \
  --model $VLLM_MODEL_NAME \
  --served-model-name $VLLM_MODEL_NAME \
  --host $VLLM_HOST \
  --port $VLLM_PORT \
  --api-key $VLLM_API_KEY \
  --max-model-len $VLLM_MAX_MODEL_LEN \
  --gpu-memory-utilization $VLLM_GPU_UTIL \
  --dtype auto \
  --trust-remote-code"

# 前缀缓存优化（仅当 VLLM_ENABLE_PREFIX_CACHING=true 时启用）
# 原理：知识库 QA 场景中系统提示词高度重复，启用后可复用已计算的 KV cache 前缀，
#       减少重复计算开销，显著提升并发推理吞吐量和首 token 延迟
if [ "$VLLM_ENABLE_PREFIX_CACHING" = "true" ]; then
    VLLM_ARGS="$VLLM_ARGS --enable-prefix-caching"
fi

# ---- 启动前端口检查 ----
# 检查 VLLM_PORT 是否已被占用，如已占用则尝试关闭旧进程
check_and_kill_port() {
    local port=$1
    local pid=""

    # 尝试用 lsof 获取占用端口的 PID
    if command -v lsof &>/dev/null; then
        pid=$(lsof -ti :"${port}" -sTCP:LISTEN 2>/dev/null | head -1)
    elif command -v ss &>/dev/null; then
        pid=$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | head -1)
    fi

    if [ -n "$pid" ]; then
        echo "[信息] 端口 ${port} 已被占用 (PID: ${pid})，正在关闭旧进程..."
        kill "$pid" 2>/dev/null || true
        # 等待最多 5 秒
        local wait=0
        while kill -0 "$pid" 2>/dev/null && [ $wait -lt 5 ]; do
            sleep 1
            wait=$((wait + 1))
        done
        # 如果还没退出，强制杀掉
        if kill -0 "$pid" 2>/dev/null; then
            echo "[信息] 强制终止旧进程 (PID: ${pid})..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo "[信息] 旧进程已终止，端口 ${port} 已释放"
        sleep 1  # 等待端口完全释放
    fi
}

# 执行端口检查
check_and_kill_port "$VLLM_PORT"

# 构建完整启动命令：激活 conda 环境后 exec 替换为 python 进程
# exec 确保 $! 获取到的 PID 就是 python 进程本身（非中间 shell）
CMD="source '${CONDA_SH}' && conda activate '${CONDA_ENV_NAME}' && exec ${VLLM_ARGS}"

# ----------------------------- 启动服务 --------------------------------------
if [ "$BACKGROUND" = "true" ]; then
    # 后台模式：使用 nohup，日志输出到 logs/vllm_server.log
    LOG_DIR="$SCRIPT_DIR/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/vllm_server.log"

    echo "[信息] 以后台模式启动 vLLM 服务（环境: ${CONDA_ENV_NAME}）..."
    echo "[信息] 日志文件: $LOG_FILE"

    # 通过 bash -c 激活 conda 环境并 exec python
    # exec 替换 shell 进程为 python，使 $! 直接获得 vLLM Python 进程的 PID
    nohup env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    VLLM_PID=$!

    # 保存 PID 到文件，供 start_rag.sh 管理
    PID_DIR="$SCRIPT_DIR/data"
    mkdir -p "$PID_DIR"
    echo "$VLLM_PID" > "$PID_DIR/.vllm.pid"
    echo "[信息] vLLM 服务 PID: $VLLM_PID (已保存到 $PID_DIR/.vllm.pid)"
else
    # 前台模式：在子 shell 中激活 conda 环境并启动 vLLM
    # exec 替换子 shell 为 python 进程，保证 PID 追踪和信号传播正确
    echo "[信息] 以前台模式启动 vLLM 服务（环境: ${CONDA_ENV_NAME}）..."
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" bash -c "$CMD" &
    VLLM_PID=$!
fi

# ----------------------------- 健康检查 --------------------------------------
# 轮询 /health 端点，最多等待 120 秒，每 3 秒检测一次
echo "[信息] 正在等待 vLLM 服务就绪..."
# curl 对 0.0.0.0 的处理在不同操作系统/网络栈下不一致，统一使用 127.0.0.1
HEALTH_URL="http://127.0.0.1:${VLLM_PORT}/health"
MODELS_URL="http://127.0.0.1:${VLLM_PORT}/v1/models"
MAX_WAIT=120
INTERVAL=3
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    # 检查进程是否仍在运行
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[错误] vLLM 进程已退出，请检查日志"
        exit 1
    fi

    # 尝试访问健康检查端点
    if curl -s -o /dev/null -w "%{http_code}" "$HEALTH_URL" 2>/dev/null | grep -q "200"; then
        # 端口就绪，进一步验证模型是否加载完成
        MODEL_RESPONSE=$(curl -s "$MODELS_URL" 2>/dev/null || echo "")
        if echo "$MODEL_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if data.get('data') and len(data['data']) > 0:
        sys.exit(0)
    else:
        sys.exit(1)
except:
    sys.exit(1)
" 2>/dev/null; then
            echo ""
            echo "============================================="
            echo "  ✓ vLLM 服务已就绪！"
            echo "  端点: http://${VLLM_HOST}:${VLLM_PORT}/v1"
            echo "  模型: $VLLM_MODEL_NAME"
            echo "  耗时: ${ELAPSED} 秒"
            echo "============================================="

            # 如果是前台模式，将控制权交还给 vLLM 进程
            if [ "$BACKGROUND" = "false" ]; then
                wait $VLLM_PID
            fi
            exit 0
        else
            # 端口已就绪但模型尚未加载完成
            printf "\r[信息] 已等待 %d/%d 秒（端口就绪，模型加载中）..." $ELAPSED $MAX_WAIT
        fi
    fi

    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
    printf "\r[信息] 已等待 %d/%d 秒..." $ELAPSED $MAX_WAIT
done

# 超时警告
echo ""
echo "============================================="
echo "  ⚠ 警告：vLLM 服务在 ${MAX_WAIT} 秒内未就绪"
echo "  健康检查地址: $HEALTH_URL"
echo "  模型验证地址: $MODELS_URL"
echo "  请手动检查服务状态和日志"
echo "============================================="

# 如果是前台模式，仍然等待进程
if [ "$BACKGROUND" = "false" ]; then
    wait $VLLM_PID
fi
