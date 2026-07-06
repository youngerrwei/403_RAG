#!/bin/bash
# ==============================================================================
# RAG 系统一键启动/停止/重启脚本
# 组件：vLLM 推理服务 + web_app.py (Flask + RAG Agent)
#
# 环境说明：
#   - vLLM 服务运行在 conda 环境 rag-vllm 中（由 start_vllm.sh 内部处理）
#   - web_app.py 运行在当前环境（rag）中
#
# 用法：
#   bash start_rag.sh start          - 启动全部服务
#   bash start_rag.sh stop           - 只停止 web_app（vLLM 保持运行）
#   bash start_rag.sh stop --all     - 停止 web_app + vLLM
#   bash start_rag.sh restart        - 只重启 web_app（vLLM 保持运行）
#   bash start_rag.sh restart --all  - 重启 web_app + vLLM
#   bash start_rag.sh status         - 查看服务状态
# ==============================================================================
set -e

# ----------------------------- 基础路径 ----------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
DATA_DIR="$SCRIPT_DIR/data"
PID_VLLM="$DATA_DIR/.vllm.pid"
PID_WEBAPP="$DATA_DIR/.web_app.pid"

# 自动创建必要目录
mkdir -p "$LOG_DIR" "$DATA_DIR"

# ----------------------------- 读取 .env 配置 ---------------------------------
# 从 .env 文件加载所有 KEY=VALUE 格式的环境变量（跳过注释和空行）
ENV_FILE="$SCRIPT_DIR/.env"

if [ -f "$ENV_FILE" ]; then
    while IFS='=' read -r key value; do
        # 跳过注释行和空行
        [[ "$key" =~ ^[[:space:]]*# ]] && continue
        [[ -z "$key" ]] && continue
        # 去除首尾空格
        key=$(echo "$key" | xargs)
        value=$(echo "$value" | xargs)
        # 跳过无效的 key（含空格或为空）
        [[ -z "$key" || "$key" == *" "* ]] && continue
        export "$key=$value"
    done < "$ENV_FILE"
else
    echo "[警告] 未找到 .env 文件：$ENV_FILE，将使用默认值"
fi

# 设置默认值（与 .env 推荐值保持一致）
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_API_KEY="${VLLM_API_KEY:-lab-secret-key}"
VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-models/Qwen3-8B-Instruct}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}"
QDRANT_HOST="${QDRANT_HOST:-172.18.216.71}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-models/bge-m3}"
RERANKER_MODEL_NAME="${RERANKER_MODEL_NAME:-models/bge-reranker-v2-m3}"
DOCS_PATH="${DOCS_PATH:-/mnt/cpu_share}"
QDRANT_COLLECTION_NAME="${QDRANT_COLLECTION_NAME:-lab_knowledge_base}"
QDRANT_PARENT_COLLECTION="${QDRANT_PARENT_COLLECTION:-lab_knowledge_base_parents}"
WEBAPP_PORT=5000

# ----------------------------- 工具函数 ----------------------------------------

# 检查端口是否被占用，兼容 Linux(ss) 和 macOS(lsof)
check_port() {
    local port=$1
    if command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${port} " && return 0
    fi
    if command -v lsof &>/dev/null; then
        lsof -i :"${port}" -sTCP:LISTEN &>/dev/null && return 0
    fi
    return 1
}

# 获取占用指定端口的进程 PID
get_pid_by_port() {
    local port=$1
    local pid=""
    if command -v lsof &>/dev/null; then
        pid=$(lsof -ti :"${port}" -sTCP:LISTEN 2>/dev/null | head -1)
    elif command -v ss &>/dev/null; then
        pid=$(ss -tlnp 2>/dev/null | grep ":${port} " | grep -oP 'pid=\K[0-9]+' | head -1)
    fi
    echo "$pid"
}

# 杀掉进程（先 SIGTERM，等待 5 秒后 SIGKILL）
kill_process() {
    local pid=$1
    local name=$2
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[信息] 正在停止 $name (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        # 等待进程退出
        local wait_count=0
        while kill -0 "$pid" 2>/dev/null && [ $wait_count -lt 5 ]; do
            sleep 1
            wait_count=$((wait_count + 1))
        done
        # 如果还没退出，强制杀掉
        if kill -0 "$pid" 2>/dev/null; then
            echo "[信息] 进程未响应，强制杀掉 $name (PID: $pid)..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo "[信息] $name 已停止"
    else
        echo "[信息] $name 未运行 (PID: $pid)"
    fi
}

# ----------------------------- 启动前预检 ----------------------------------------
# 预检函数：检查关键依赖是否就绪，仅在 start 时调用
preflight_check() {
    echo "============================================="
    echo "  启动前预检"
    echo "============================================="
    echo ""

    local has_error=0

    # 颜色定义（兼容无颜色终端）
    local GREEN=""
    local YELLOW=""
    local RED=""
    local NC=""
    if [ -t 1 ] && command -v tput &>/dev/null && [ "$(tput colors 2>/dev/null)" -ge 8 ]; then
        GREEN="\033[0;32m"
        YELLOW="\033[0;33m"
        RED="\033[0;31m"
        NC="\033[0m"
    fi

    # 1. 检查 .env 文件存在性
    if [ -f "$ENV_FILE" ]; then
        printf "  ${GREEN}✓${NC} .env 文件存在\n"
    else
        printf "  ${RED}✗${NC} .env 文件不存在: $ENV_FILE\n"
        has_error=1
    fi

    # 2. 模型路径存在性检查
    local models_to_check="VLLM_MODEL_NAME:$VLLM_MODEL_NAME EMBEDDING_MODEL_NAME:$EMBEDDING_MODEL_NAME RERANKER_MODEL_NAME:$RERANKER_MODEL_NAME"
    for entry in $models_to_check; do
        local var_name="${entry%%:*}"
        local model_path="${entry#*:}"
        # 处理相对路径：相对于脚本目录
        if [[ "$model_path" == ./* ]]; then
            model_path="$SCRIPT_DIR/${model_path#./}"
        fi
        if [ -d "$model_path" ]; then
            printf "  ${GREEN}✓${NC} $var_name 模型目录存在: $model_path\n"
        else
            printf "  ${RED}✗${NC} $var_name 模型目录不存在: $model_path\n"
            has_error=1
        fi
    done

    # 3. Qdrant 连通性检查（失败只警告，不阻止启动）
    local qdrant_ok=0
    if command -v curl &>/dev/null; then
        if curl -sf "http://${QDRANT_HOST}:${QDRANT_PORT}/collections" --connect-timeout 3 &>/dev/null; then
            qdrant_ok=1
        fi
    elif command -v nc &>/dev/null; then
        # nc 参数兼容：macOS 使用 -G，Linux 使用 -w
        if nc -z -w 3 "$QDRANT_HOST" "$QDRANT_PORT" 2>/dev/null || nc -z -G 3 "$QDRANT_HOST" "$QDRANT_PORT" 2>/dev/null; then
            qdrant_ok=1
        fi
    fi

    if [ $qdrant_ok -eq 1 ]; then
        printf "  ${GREEN}✓${NC} Qdrant 服务可达: ${QDRANT_HOST}:${QDRANT_PORT}\n"
    else
        printf "  ${YELLOW}⚠${NC} Qdrant 服务无法连接: ${QDRANT_HOST}:${QDRANT_PORT}（服务可能稍后启动）\n"
    fi

    # 4. 必要目录检查/创建
    for dir in "$LOG_DIR" "$DATA_DIR"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            printf "  ${GREEN}✓${NC} 已创建目录: $dir\n"
        else
            printf "  ${GREEN}✓${NC} 目录存在: $dir\n"
        fi
    done

    echo ""

    # 5. 预检结果汇总
    if [ $has_error -ne 0 ]; then
        printf "  ${RED}[预检失败]${NC} 存在关键错误，请修复后重试\n"
        echo "============================================="
        echo ""
        return 1
    else
        printf "  ${GREEN}[预检通过]${NC} 所有检查项均正常\n"
        echo "============================================="
        echo ""
        return 0
    fi
}

# ----------------------------- start 子命令 ------------------------------------
do_start() {
    # 启动前执行预检
    if ! preflight_check; then
        echo "[错误] 预检未通过，中止启动"
        exit 1
    fi

    echo "============================================="
    echo "  RAG 系统启动流程"
    echo "============================================="
    echo ""

    # -------------------- Step 1: 检查/启动 vLLM --------------------
    # vLLM 运行在独立的 conda 环境 rag-vllm 中，由 start_vllm.sh 内部处理环境切换
    echo ">>> Step 1: 检查/启动 vLLM 推理服务（环境: rag-vllm）"
    echo "    模型: $VLLM_MODEL_NAME"
    echo "    端口: $VLLM_PORT"
    echo ""

    if check_port "$VLLM_PORT"; then
        # 端口已占用，检查模型是否匹配
        echo "[信息] 端口 $VLLM_PORT 已被占用，检查模型配置..."
        # 注意：vLLM 配置了 API Key 认证，curl 需要携带 Authorization 头
        CURRENT_MODEL=$(curl -s -H "Authorization: Bearer ${VLLM_API_KEY}" "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['data'][0]['id'])" 2>/dev/null || echo "")

        # 路径标准化比较：去除 ./ 前缀，避免 './models/x' 与 'models/x' 不匹配
        NORMALIZED_CURRENT=$(echo "$CURRENT_MODEL" | sed 's|^\./||')
        NORMALIZED_EXPECT=$(echo "$VLLM_MODEL_NAME" | sed 's|^\./||')

        if [ "$NORMALIZED_CURRENT" = "$NORMALIZED_EXPECT" ]; then
            echo "[信息] ✓ vLLM 已运行且配置正确，跳过"
        else
            echo "[信息] 当前模型: '$CURRENT_MODEL'，期望模型: '$VLLM_MODEL_NAME'"
            echo "[信息] 模型不匹配，重启 vLLM..."
            # 杀掉旧进程
            local old_pid=$(get_pid_by_port "$VLLM_PORT")
            if [ -n "$old_pid" ]; then
                kill_process "$old_pid" "旧 vLLM 进程"
            fi
            sleep 2
            # 重新启动
            echo "[信息] 调用 start_vllm.sh --background 启动 vLLM..."
            bash "$SCRIPT_DIR/start_vllm.sh" --background
            _wait_vllm
        fi
    else
        # 端口未占用，启动 vLLM
        echo "[信息] 端口 $VLLM_PORT 未被占用，启动 vLLM..."
        bash "$SCRIPT_DIR/start_vllm.sh" --background
        _wait_vllm
    fi

    echo ""

    # -------------------- Step 2: 启动 web_app.py --------------------
    # web_app.py 运行在当前环境（rag）中，无需切换 conda 环境
    echo ">>> Step 2: 检查/启动 web_app.py（当前环境）"
    echo "    端口: $WEBAPP_PORT"
    echo ""

    if check_port "$WEBAPP_PORT"; then
        echo "[信息] ✓ web_app 已运行，跳过"
    else
        echo "[信息] 启动 web_app.py..."
        cd "$SCRIPT_DIR"
        nohup python web_app.py > "$LOG_DIR/web_app.log" 2>&1 &
        local webapp_pid=$!
        echo "$webapp_pid" > "$PID_WEBAPP"
        echo "[信息] web_app PID: $webapp_pid (已保存到 $PID_WEBAPP)"

        # 等待 web_app 就绪（优先使用 /api/health，回退到 /login）
        echo "[信息] 正在等待 web_app 就绪..."
        local max_wait=60
        local interval=3
        local elapsed=0
        local health_ready=0
        while [ $elapsed -lt $max_wait ]; do
            # 尝试 /api/health 端点检查
            local response=$(curl -sf "http://127.0.0.1:${WEBAPP_PORT}/api/health" 2>/dev/null || echo "")
            if [ -n "$response" ]; then
                local status=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null || echo "")
                if [ "$status" = "ok" ]; then
                    echo ""
                    echo "[信息] ✓ web_app 已完全就绪 (耗时: ${elapsed} 秒)"
                    health_ready=1
                    break
                elif [ "$status" = "degraded" ]; then
                    echo ""
                    echo "[警告] web_app 处于降级模式（部分服务不可用），但可以接受请求"
                    health_ready=1
                    break
                fi
            fi

            # 前几次失败时回退到 /login 检查（兼容过渡期）
            if [ $elapsed -le 15 ]; then
                if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${WEBAPP_PORT}/login" 2>/dev/null | grep -q "200"; then
                    # /login 可访问但 /api/health 还未就绪，继续等待 health
                    printf "\r[信息] 已等待 %d/%d 秒（服务已响应，等待完全就绪）..." $elapsed $max_wait
                    sleep $interval
                    elapsed=$((elapsed + interval))
                    continue
                fi
            fi

            sleep $interval
            elapsed=$((elapsed + interval))
            printf "\r[信息] 已等待 %d/%d 秒..." $elapsed $max_wait
        done

        # 超时后的最终确认：尝试回退到 /login
        if [ $health_ready -eq 0 ]; then
            if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:${WEBAPP_PORT}/login" 2>/dev/null | grep -q "200"; then
                echo ""
                echo "[警告] /api/health 未就绪，但 /login 可访问，服务可能部分可用"
            else
                echo ""
                echo "[警告] web_app 在 ${max_wait} 秒内未就绪，请手动检查日志: $LOG_DIR/web_app.log"
            fi
        fi
    fi

    echo ""

    # -------------------- Step 3: 打印启动摘要 --------------------
    echo "============================================="
    echo "  RAG 系统启动摘要"
    echo "============================================="
    echo "  vLLM 服务:    http://127.0.0.1:${VLLM_PORT}/v1"
    echo "  模型名称:     $VLLM_MODEL_NAME"
    echo "  Web 应用:     http://127.0.0.1:${WEBAPP_PORT}"
    echo "  vLLM 日志:    $LOG_DIR/vllm_server.log"
    echo "  Web 日志:     $LOG_DIR/web_app.log"
    echo "============================================="
    echo "  ✓ 所有服务已启动完成！"
    echo "============================================="
}

# 等待 vLLM 就绪（轮询 /health + 验证模型加载）
_wait_vllm() {
    echo "[信息] 正在等待 vLLM 服务就绪..."
    local health_url="http://127.0.0.1:${VLLM_PORT}/health"
    local models_url="http://127.0.0.1:${VLLM_PORT}/v1/models"
    local max_wait=120
    local interval=3
    local elapsed=0

    while [ $elapsed -lt $max_wait ]; do
        if curl -s -o /dev/null -w "%{http_code}" "$health_url" 2>/dev/null | grep -q "200"; then
            # 端口就绪，进一步验证模型是否加载完成
            local model_response=$(curl -s "$models_url" 2>/dev/null || echo "")
            if echo "$model_response" | python3 -c "
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
                echo "[信息] ✓ vLLM 服务已就绪，模型加载完成 (耗时: ${elapsed} 秒)"
                return 0
            else
                # 端口已就绪但模型尚未加载完成
                printf "\r[信息] 已等待 %d/%d 秒（端口就绪，模型加载中）..." $elapsed $max_wait
            fi
        fi
        sleep $interval
        elapsed=$((elapsed + interval))
        printf "\r[信息] 已等待 %d/%d 秒..." $elapsed $max_wait
    done

    echo ""
    echo "[警告] vLLM 服务在 ${max_wait} 秒内未就绪，请手动检查日志: $LOG_DIR/vllm_server.log"
    echo "[警告] 模型验证地址: $models_url"
    return 1
}

# ----------------------------- stop 子命令 -------------------------------------
# vLLM 由 `start_vllm.sh stop` 独立管理，默认 stop 只关闭 web_app
# 如需同时停止 vLLM，请使用 `bash start_rag.sh stop --all`
do_stop() {
    local stop_all=false
    # 检查是否传入 --all 参数
    if [ "${1:-}" = "--all" ]; then
        stop_all=true
    fi

    set +e  # stop 时不因 kill 失败而退出
    echo "============================================="
    echo "  RAG 系统停止流程"
    if [ "$stop_all" = "true" ]; then
        echo "  模式: 停止 web_app + vLLM"
    else
        echo "  模式: 仅停止 web_app（vLLM 保持运行）"
    fi
    echo "============================================="
    echo ""

    # 停止 web_app
    echo ">>> 停止 web_app..."
    if [ -f "$PID_WEBAPP" ]; then
        local pid=$(cat "$PID_WEBAPP")
        kill_process "$pid" "web_app"
        rm -f "$PID_WEBAPP"
    else
        # 通过端口查找
        local pid=$(get_pid_by_port "$WEBAPP_PORT")
        if [ -n "$pid" ]; then
            kill_process "$pid" "web_app"
        else
            echo "[信息] web_app 未运行"
        fi
    fi

    echo ""

    # 仅在 --all 模式下停止 vLLM
    if [ "$stop_all" = "true" ]; then
        echo ">>> 停止 vLLM..."
        if [ -f "$PID_VLLM" ]; then
            local pid=$(cat "$PID_VLLM")
            kill_process "$pid" "vLLM"
            rm -f "$PID_VLLM"
        else
            # 通过端口查找
            local pid=$(get_pid_by_port "$VLLM_PORT")
            if [ -n "$pid" ]; then
                kill_process "$pid" "vLLM"
            else
                echo "[信息] vLLM 未运行"
            fi
        fi
        echo ""
        echo "[信息] ✓ web_app + vLLM 已停止"
    else
        echo "[信息] ✓ web_app 已停止（vLLM 保持运行，如需停止请使用 stop --all）"
    fi
    set -e
}

# ----------------------------- restart 子命令 ----------------------------------
do_restart() {
    local restart_all=""
    if [ "${1:-}" = "--all" ]; then
        restart_all="--all"
        echo "[信息] 正在重启 RAG 系统（包括 vLLM）..."
    else
        echo "[信息] 正在重启 web_app（vLLM 保持运行）..."
    fi
    echo ""
    do_stop $restart_all
    echo ""
    sleep 2
    do_start
}

# ----------------------------- status 子命令 -----------------------------------
do_status() {
    echo "============================================="
    echo "  RAG 系统服务状态"
    echo "============================================="
    echo ""

    # vLLM 状态
    echo ">>> vLLM 推理服务"
    if check_port "$VLLM_PORT"; then
        local vllm_pid=$(get_pid_by_port "$VLLM_PORT")
        echo "  状态:   运行中 (PID: ${vllm_pid:-未知})"
        echo "  端口:   $VLLM_PORT"
        # 尝试获取当前模型名
        local current_model=$(curl -s "http://127.0.0.1:${VLLM_PORT}/v1/models" 2>/dev/null \
            | python3 -c "import sys,json; data=json.load(sys.stdin); print(data['data'][0]['id'])" 2>/dev/null || echo "无法获取")
        echo "  模型:   $current_model"
    else
        echo "  状态:   未运行"
        echo "  端口:   $VLLM_PORT"
    fi
    echo ""

    # web_app 状态
    echo ">>> Web 应用 (web_app.py)"
    if check_port "$WEBAPP_PORT"; then
        local webapp_pid=$(get_pid_by_port "$WEBAPP_PORT")
        echo "  状态:   运行中 (PID: ${webapp_pid:-未知})"
        echo "  端口:   $WEBAPP_PORT"
    else
        echo "  状态:   未运行"
        echo "  端口:   $WEBAPP_PORT"
    fi
    echo ""

    # GPU 使用情况
    echo ">>> GPU 使用情况"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>/dev/null \
            | while IFS=',' read -r idx name mem_used mem_total; do
                echo "  GPU $idx: $name | 显存: $mem_used /$mem_total"
            done
    else
        echo "  nvidia-smi 不可用"
    fi

    echo ""
    echo "============================================="
}

# ----------------------------- 主入口 ------------------------------------------
case "${1:-}" in
    start)
        do_start
        ;;
    stop)
        do_stop "${2:-}"
        ;;
    restart)
        do_restart "${2:-}"
        ;;
    status)
        do_status
        ;;
    *)
        echo "用法: bash $0 {start|stop|restart|status} [--all]"
        echo ""
        echo "子命令说明："
        echo "  start          - 启动全部服务 (vLLM + web_app)"
        echo "  stop           - 只停止 web_app（vLLM 由 start_vllm.sh stop 独立管理）"
        echo "  stop --all     - 停止 web_app + vLLM"
        echo "  restart        - 只重启 web_app（vLLM 保持运行）"
        echo "  restart --all  - 重启 web_app + vLLM"
        echo "  status         - 查看服务状态"
        exit 1
        ;;
esac
