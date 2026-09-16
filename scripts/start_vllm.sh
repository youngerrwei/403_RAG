#!/bin/bash
# vLLM 推理服务安全启动/停止/状态管理（包含认证与模型校验）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/runtime_common.sh"

ENV_FILE="${RAG_ENV_FILE:-$PROJECT_ROOT/.env}"
DATA_DIR="$PROJECT_ROOT/data"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$DATA_DIR/.vllm.pid"
LOCK_FILE="$DATA_DIR/.vllm.lock"
mkdir -p "$DATA_DIR" "$LOG_DIR"

load_env_keys "$ENV_FILE" \
    VLLM_MODEL_NAME VLLM_API_KEY VLLM_CUDA_DEVICES VLLM_HOST VLLM_PORT \
    VLLM_GPU_UTIL VLLM_MAX_MODEL_LEN VLLM_ENABLE_PREFIX_CACHING \
    VLLM_CONDA_ENV VLLM_STARTUP_TIMEOUT VLLM_INFERENCE_PROBE_TIMEOUT \
    STARTUP_PROBE_TIMEOUT || true

VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-./models/Qwen3-8B-Instruct}"
VLLM_API_KEY="${VLLM_API_KEY:-lab-secret-key}"
VLLM_CUDA_DEVICES="${VLLM_CUDA_DEVICES:-3}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_GPU_UTIL="${VLLM_GPU_UTIL:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6000}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-true}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-rag-vllm}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-300}"
VLLM_INFERENCE_PROBE_TIMEOUT="${VLLM_INFERENCE_PROBE_TIMEOUT:-15}"
STARTUP_PROBE_TIMEOUT="${STARTUP_PROBE_TIMEOUT:-5}"
MODEL_PATH="$(resolve_project_path "$VLLM_MODEL_NAME")"
CONDA_CMD="$(find_conda || true)"
PYTHON_BIN=""
[[ -n "$CONDA_CMD" ]] && PYTHON_BIN="$(conda_env_python "$CONDA_CMD" "$VLLM_CONDA_ENV" || true)"

probe_vllm() {
    PROBE_MODEL=""
    [[ -n "$PYTHON_BIN" ]] || return 3
    local response
    http_get_body "$PYTHON_BIN" "http://127.0.0.1:${VLLM_PORT}/health" \
        "$STARTUP_PROBE_TIMEOUT" >/dev/null 2>&1 || return 2
    response="$(http_get_body "$PYTHON_BIN" "http://127.0.0.1:${VLLM_PORT}/v1/models" \
        "$STARTUP_PROBE_TIMEOUT" "$VLLM_API_KEY" 2>/dev/null || true)"
    PROBE_MODEL="$(printf '%s' "$response" | extract_vllm_model_id "$PYTHON_BIN" 2>/dev/null || true)"
    [[ -n "$PROBE_MODEL" ]] || return 2
    model_names_match "$PROBE_MODEL" "$VLLM_MODEL_NAME" || return 1
    probe_vllm_inference "$PYTHON_BIN" \
        "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
        "$VLLM_INFERENCE_PROBE_TIMEOUT" "$VLLM_API_KEY" "$VLLM_MODEL_NAME" \
        >/dev/null 2>&1 || return 4
}

managed_pid() {
    local pid=""
    [[ -f "$PID_FILE" ]] && pid="$(<"$PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && pid_is_vllm "$pid"; then
        printf '%s' "$pid"
        return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$PID_FILE"
    fi
    return 1
}

expected_vllm_listener() {
    local pid="$1" cmd
    cmd="$(pid_cmdline "$pid" 2>/dev/null || true)"
    [[ "$cmd" == *"vllm.entrypoints.openai.api_server"* ]] || return 1
    [[ "$cmd" == *"--port $VLLM_PORT"* ]] || return 1
    [[ "$cmd" == *"--model $MODEL_PATH"* || "$cmd" == *"--model $VLLM_MODEL_NAME"* ]]
}

do_stop() {
    local orphan_mode="${1:-}" pid listener
    pid="$(managed_pid || true)"
    if [[ -z "$pid" ]]; then
        listener="$(port_pid "$VLLM_PORT")"
        if [[ -n "$listener" ]]; then
            if [[ "$orphan_mode" == "--orphan" ]] && expected_vllm_listener "$listener"; then
                echo "[警告] PID 文件缺失，但已确认 PID $listener 是当前模型和端口的 vLLM；正在停止孤儿进程"
                stop_pid_gracefully "$listener" "vLLM 孤儿进程"
                rm -f "$PID_FILE"
                echo "[信息] vLLM 孤儿进程已停止"
                return 0
            fi
            if expected_vllm_listener "$listener"; then
                echo "[错误] 端口 ${VLLM_PORT} 由符合当前配置但未纳管的 vLLM 占用 (PID: $listener)"
                echo "[提示] 核对进程后可执行: bash scripts/start_vllm.sh stop --orphan"
                return 1
            fi
            echo "[错误] 端口 ${VLLM_PORT} 由未纳管进程占用 (PID: $listener)，拒绝停止"
            return 1
        fi
        echo "[信息] vLLM 未运行"
        return 0
    fi
    echo "[信息] 正在停止 vLLM (PID: $pid)..."
    stop_pid_gracefully "$pid" "vLLM"
    rm -f "$PID_FILE"
    echo "[信息] vLLM 已停止"
}

do_status() {
    local pid listener
    pid="$(managed_pid || true)"
    listener="$(port_pid "$VLLM_PORT")"
    echo "============================================="
    if probe_vllm; then
        echo "  vLLM 状态: 健康"
        echo "  模型:      $PROBE_MODEL"
        echo "  PID:       ${pid:-${listener:-未知}}"
        echo "  端口:      $VLLM_PORT"
        echo "============================================="
        return 0
    fi
    if [[ -n "${PROBE_MODEL:-}" ]]; then
        echo "  vLLM 状态: API/模型可达，但真实生成探针失败"
        echo "  模型:      $PROBE_MODEL"
        echo "  PID:       ${pid:-${listener:-未知}}"
        echo "  建议:      检查 GPU 与 logs/vllm_server.log，必要时安全重启"
    elif [[ -n "$listener" ]]; then
        echo "  vLLM 状态: 端口已占用但服务未通过认证健康检查"
        echo "  PID:       $listener"
    else
        echo "  vLLM 状态: 未运行"
    fi
    echo "  端口:      $VLLM_PORT"
    echo "============================================="
    return 1
}

case "${1:-}" in
    stop)
        [[ $# -le 2 && ( $# -eq 1 || "${2:-}" == "--orphan" ) ]] || {
            echo "用法: bash scripts/start_vllm.sh stop [--orphan]"
            exit 1
        }
        do_stop "${2:-}"
        exit $?
        ;;
    status) do_status; exit $? ;;
esac

BACKGROUND=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --background) BACKGROUND=true ;;
        --gpu)
            [[ $# -ge 2 && "$2" != --* ]] || { echo "[错误] --gpu 需要设备号"; exit 1; }
            VLLM_CUDA_DEVICES="$2"
            shift ;;
        *) echo "[错误] 未知参数: $1"; exit 1 ;;
    esac
    shift
done

[[ -f "$ENV_FILE" ]] || { echo "[错误] .env 不存在: $ENV_FILE"; exit 1; }
[[ -d "$MODEL_PATH" ]] || { echo "[错误] 模型路径不存在: $MODEL_PATH"; exit 1; }
command -v flock >/dev/null 2>&1 || { echo "[错误] 未找到 flock"; exit 1; }
[[ -n "$CONDA_CMD" ]] || { echo "[错误] 未找到 conda"; exit 1; }
[[ -n "$PYTHON_BIN" ]] || { echo "[错误] conda 环境不存在或无 Python: $VLLM_CONDA_ENV"; exit 1; }

exec 9>"$LOCK_FILE"
flock -n 9 || { echo "[错误] 另一个 vLLM 管理操作正在执行"; exit 1; }

if probe_vllm; then
    echo "[信息] vLLM 已运行且模型正确: $PROBE_MODEL"
    exit 0
fi

listener="$(port_pid "$VLLM_PORT")"
if [[ -n "$listener" ]]; then
    old_pid="$(managed_pid || true)"
    if [[ -n "$old_pid" && "$old_pid" == "$listener" && "${PROBE_MODEL:-}" != "" ]]; then
        echo "[信息] 受管 vLLM 模型不匹配，安全重启 (PID: $old_pid)"
        stop_pid_gracefully "$old_pid" "vLLM"
        rm -f "$PID_FILE"
    else
        echo "[错误] 端口 ${VLLM_PORT} 已被占用，且无法确认是可安全重启的受管 vLLM (PID: $listener)"
        exit 1
    fi
fi

args=(
    "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server
    --model "$MODEL_PATH"
    --served-model-name "$VLLM_MODEL_NAME"
    --host "$VLLM_HOST" --port "$VLLM_PORT"
    --api-key "$VLLM_API_KEY"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --gpu-memory-utilization "$VLLM_GPU_UTIL"
    --dtype auto --trust-remote-code
)
[[ "$VLLM_ENABLE_PREFIX_CACHING" == "true" ]] && args+=(--enable-prefix-caching)

LOG_FILE="$LOG_DIR/vllm_server.log"
echo "[信息] 启动 vLLM：环境=$VLLM_CONDA_ENV GPU=$VLLM_CUDA_DEVICES 端口=$VLLM_PORT"
if [[ "$BACKGROUND" == "true" ]]; then
    nohup env CUDA_VISIBLE_DEVICES="$VLLM_CUDA_DEVICES" "${args[@]}" 9>&- >"$LOG_FILE" 2>&1 &
else
    env CUDA_VISIBLE_DEVICES="$VLLM_CUDA_DEVICES" "${args[@]}" 9>&- > >(tee -a "$LOG_FILE") 2>&1 &
fi
VLLM_PID=$!
printf '%s\n' "$VLLM_PID" > "$PID_FILE"

started_at=$SECONDS
elapsed=0
while (( SECONDS - started_at < VLLM_STARTUP_TIMEOUT )); do
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        rm -f "$PID_FILE"
        echo "[错误] vLLM 进程提前退出，请检查 $LOG_FILE"
        exit 1
    fi
    if probe_vllm; then
        elapsed=$((SECONDS - started_at))
        echo "[信息] vLLM 已就绪，模型=$PROBE_MODEL，耗时=${elapsed}s"
        if [[ "$BACKGROUND" == "false" ]]; then
            wait "$VLLM_PID"
        fi
        exit 0
    fi
    sleep 3
    elapsed=$((SECONDS - started_at))
done

echo "[错误] vLLM 在 ${VLLM_STARTUP_TIMEOUT}s 内未就绪，停止本次启动的进程"
stop_pid_gracefully "$VLLM_PID" "vLLM" || true
rm -f "$PID_FILE"
exit 1
