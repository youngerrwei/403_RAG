#!/bin/bash
set -e

# =============================================================================
# vLLM 模型服务启动脚本
# 适配：单卡 4090 24GB，知识库 QA 场景
# 用法：
#   前台运行：bash start_vllm.sh
#   后台运行：bash start_vllm.sh --background
#   指定 GPU：bash start_vllm.sh --gpu 1
#   多卡运行：bash start_vllm.sh --gpu 0,1
# =============================================================================

# ----------------------------- 读取 .env 配置 --------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

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
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-./models/Qwen3-8B-Instruct}"
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
CMD="python -m vllm.entrypoints.openai.api_server \
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
    CMD="$CMD --enable-prefix-caching"
fi

# ----------------------------- 启动服务 --------------------------------------
if [ "$BACKGROUND" = "true" ]; then
    # 后台模式：使用 nohup，日志输出到 logs/vllm_server.log
    LOG_DIR="$SCRIPT_DIR/logs"
    mkdir -p "$LOG_DIR"
    LOG_FILE="$LOG_DIR/vllm_server.log"

    echo "[信息] 以后台模式启动 vLLM 服务..."
    echo "[信息] 日志文件: $LOG_FILE"

    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    VLLM_PID=$!

    # 保存 PID 到文件，供 start_rag.sh 管理
    PID_DIR="$SCRIPT_DIR/data"
    mkdir -p "$PID_DIR"
    echo "$VLLM_PID" > "$PID_DIR/.vllm.pid"
    echo "[信息] vLLM 服务 PID: $VLLM_PID (已保存到 $PID_DIR/.vllm.pid)"
else
    # 前台模式：直接在当前终端运行
    echo "[信息] 以前台模式启动 vLLM 服务..."
    # 在子进程中启动，以便后续进行健康检查
    bash -c "$CMD" &
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
