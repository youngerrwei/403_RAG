#!/bin/bash
# RAG 系统一键启动/停止/重启/状态管理（使用明确 conda 解释器）。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/runtime_common.sh"

ENV_FILE="${RAG_ENV_FILE:-$SCRIPT_DIR/.env}"
DATA_DIR="$SCRIPT_DIR/data"
RAG_LOG_DIR="$SCRIPT_DIR/logs"
PID_WEBAPP="$DATA_DIR/.web_app.pid"
LOCK_FILE="$DATA_DIR/.rag_manager.lock"
mkdir -p "$DATA_DIR" "$RAG_LOG_DIR"

load_env_keys "$ENV_FILE" \
    VLLM_MODEL_NAME VLLM_API_KEY VLLM_PORT QDRANT_HOST QDRANT_PORT \
    QDRANT_COLLECTION_NAME QDRANT_PARENT_COLLECTION \
    EMBEDDING_MODEL_NAME RERANKER_MODEL_NAME DOCS_PATH FLASK_SECRET_KEY \
    RAG_CONDA_ENV WEBAPP_PORT WEBAPP_STARTUP_TIMEOUT STARTUP_PROBE_TIMEOUT || true

VLLM_MODEL_NAME="${VLLM_MODEL_NAME:-./models/Qwen3-8B-Instruct}"
VLLM_API_KEY="${VLLM_API_KEY:-lab-secret-key}"
VLLM_PORT="${VLLM_PORT:-8000}"
QDRANT_HOST="${QDRANT_HOST:-172.18.216.71}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
QDRANT_COLLECTION_NAME="${QDRANT_COLLECTION_NAME:-lab_knowledge_base}"
QDRANT_PARENT_COLLECTION="${QDRANT_PARENT_COLLECTION:-lab_knowledge_base_parents}"
EMBEDDING_MODEL_NAME="${EMBEDDING_MODEL_NAME:-./models/bge-m3}"
RERANKER_MODEL_NAME="${RERANKER_MODEL_NAME:-./models/bge-reranker-v2-m3}"
DOCS_PATH="${DOCS_PATH:-/mnt/cpu_share}"
FLASK_SECRET_KEY="${FLASK_SECRET_KEY:-}"
RAG_CONDA_ENV="${RAG_CONDA_ENV:-rag}"
WEBAPP_PORT="${WEBAPP_PORT:-5000}"
WEBAPP_STARTUP_TIMEOUT="${WEBAPP_STARTUP_TIMEOUT:-60}"
STARTUP_PROBE_TIMEOUT="${STARTUP_PROBE_TIMEOUT:-5}"
CONDA_CMD="$(find_conda || true)"
RAG_PYTHON=""
[[ -n "$CONDA_CMD" ]] && RAG_PYTHON="$(conda_env_python "$CONDA_CMD" "$RAG_CONDA_ENV" || true)"

web_pid() {
    local pid=""
    [[ -f "$PID_WEBAPP" ]] && pid="$(<"$PID_WEBAPP")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null && pid_is_webapp "$pid"; then
        printf '%s' "$pid"
        return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
        rm -f "$PID_WEBAPP"
    fi
    return 1
}

web_health_status() {
    local response
    WEB_HEALTH_STATUS=""
    [[ -n "$RAG_PYTHON" ]] || return 1
    response="$(http_get_body "$RAG_PYTHON" \
        "http://127.0.0.1:${WEBAPP_PORT}/api/health" "$STARTUP_PROBE_TIMEOUT" 2>/dev/null || true)"
    WEB_HEALTH_STATUS="$(printf '%s' "$response" | extract_json_string status)"
    [[ "$WEB_HEALTH_STATUS" == "ok" || "$WEB_HEALTH_STATUS" == "degraded" ]]
}

preflight_check() {
    local failed=0 configured path
    echo "========== 启动前预检 =========="
    [[ -f "$ENV_FILE" ]] || { echo "[错误] .env 不存在: $ENV_FILE"; failed=1; }
    command -v flock >/dev/null 2>&1 || { echo "[错误] 未找到 flock"; failed=1; }
    [[ -n "$CONDA_CMD" ]] || { echo "[错误] 未找到 conda"; failed=1; }
    [[ -n "$RAG_PYTHON" ]] || { echo "[错误] conda 环境不存在或无 Python: $RAG_CONDA_ENV"; failed=1; }
    is_weak_flask_secret "$FLASK_SECRET_KEY" && { echo "[错误] FLASK_SECRET_KEY 为空、过短或仍是公开占位值"; failed=1; }
    for configured in "$VLLM_MODEL_NAME" "$EMBEDDING_MODEL_NAME" "$RERANKER_MODEL_NAME"; do
        path="$(resolve_project_path "$configured")"
        [[ -d "$path" ]] || { echo "[错误] 模型目录不存在: $path"; failed=1; }
    done
    if [[ ! -d "$DOCS_PATH" ]]; then
        echo "[警告] 文档目录不存在: $DOCS_PATH"
    elif [[ -z "$(find "$DOCS_PATH" -type f -print -quit 2>/dev/null)" ]]; then
        echo "[警告] 文档目录为空: $DOCS_PATH；请确认共享盘已挂载，入库当前不可执行"
    fi
    if [[ -n "$RAG_PYTHON" ]] && qdrant_collections_ready "$RAG_PYTHON" \
        "http://${QDRANT_HOST}:${QDRANT_PORT}/collections" 3 \
        "$QDRANT_COLLECTION_NAME" "$QDRANT_PARENT_COLLECTION" >/dev/null 2>&1; then
        echo "[信息] Qdrant 可达，必要集合完整"
    elif [[ -n "$RAG_PYTHON" ]] && http_get_body "$RAG_PYTHON" \
        "http://${QDRANT_HOST}:${QDRANT_PORT}/collections" 3 >/dev/null 2>&1; then
        echo "[警告] Qdrant 可达但必要集合缺失，请在文档盘挂载后执行: bash auto_ingest.sh --full"
    else
        echo "[警告] Qdrant 当前不可达，Web 将以 degraded 模式启动"
    fi
    (( failed == 0 ))
}

start_web() {
    local pid listener elapsed=0
    listener="$(port_pid "$WEBAPP_PORT")"
    if [[ -n "$listener" ]]; then
        pid="$(web_pid || true)"
        if [[ "$pid" == "$listener" ]] && web_health_status; then
            echo "[信息] Web 已运行，状态=$WEB_HEALTH_STATUS，PID=$pid"
            return 0
        fi
        echo "[错误] 端口 ${WEBAPP_PORT} 已被未知或不健康进程占用 (PID: $listener)"
        return 1
    fi

    [[ -n "$RAG_PYTHON" ]] || { echo "[错误] conda 环境不存在或无 Python: $RAG_CONDA_ENV"; return 1; }
    echo "[信息] 使用 $RAG_PYTHON 启动 Web"
    (
        cd "$SCRIPT_DIR"
        nohup "$RAG_PYTHON" web_app.py 9>&- >"$RAG_LOG_DIR/web_app.log" 2>&1 &
        printf '%s\n' "$!" > "$PID_WEBAPP"
    )
    pid="$(<"$PID_WEBAPP")"
    while (( elapsed < WEBAPP_STARTUP_TIMEOUT )); do
        if ! kill -0 "$pid" 2>/dev/null; then
            rm -f "$PID_WEBAPP"
            echo "[错误] Web 进程提前退出，请检查 $RAG_LOG_DIR/web_app.log"
            return 1
        fi
        if web_health_status; then
            echo "[信息] Web 已就绪，状态=$WEB_HEALTH_STATUS，耗时=${elapsed}s"
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done
    echo "[错误] Web 在 ${WEBAPP_STARTUP_TIMEOUT}s 内未通过 /api/health"
    if pid_is_webapp "$pid"; then stop_pid_gracefully "$pid" "Web" || true; fi
    rm -f "$PID_WEBAPP"
    return 1
}

stop_web() {
    local pid listener
    pid="$(web_pid || true)"
    if [[ -z "$pid" ]]; then
        listener="$(port_pid "$WEBAPP_PORT")"
        if [[ -n "$listener" ]]; then
            echo "[错误] Web 端口由未纳管进程占用 (PID: $listener)，拒绝停止"
            return 1
        fi
        echo "[信息] Web 未运行"
        return 0
    fi
    echo "[信息] 正在停止 Web (PID: $pid)..."
    stop_pid_gracefully "$pid" "Web"
    rm -f "$PID_WEBAPP"
}

do_start() {
    preflight_check || return 1
    bash "$SCRIPT_DIR/start_vllm.sh" --background || return 1
    start_web || return 1
    echo "============================================="
    echo "  RAG 系统启动成功"
    echo "  vLLM: http://127.0.0.1:${VLLM_PORT}/v1"
    echo "  Web:  http://127.0.0.1:${WEBAPP_PORT}"
    echo "============================================="
}

do_stop() {
    local stop_all="${1:-}"
    stop_web || return 1
    if [[ "$stop_all" == "--all" ]]; then
        bash "$SCRIPT_DIR/start_vllm.sh" stop || return 1
    fi
}

do_status() {
    local failed=0 pid listener
    bash "$SCRIPT_DIR/start_vllm.sh" status || failed=1
    pid="$(web_pid || true)"
    listener="$(port_pid "$WEBAPP_PORT")"
    if [[ -n "$pid" && "$pid" == "$listener" ]] && web_health_status; then
        echo "Web 状态: 健康（$WEB_HEALTH_STATUS），PID=$pid，端口=$WEBAPP_PORT"
    elif [[ -n "$listener" ]]; then
        echo "Web 状态: 端口占用但身份或健康检查失败，PID=$listener"
        failed=1
    else
        echo "Web 状态: 未运行"
        failed=1
    fi
    return "$failed"
}

command="${1:-start}"
shift || true
command -v flock >/dev/null 2>&1 || { echo "[错误] 未找到 flock"; exit 1; }
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "[错误] 另一个 RAG 管理操作正在执行"; exit 1; }

case "$command" in
    start) do_start ;;
    stop) do_stop "${1:-}" ;;
    restart)
        restart_all="${1:-}"
        do_stop "$restart_all"
        if [[ "$restart_all" == "--all" ]]; then
            sleep 2
        fi
        do_start ;;
    status) do_status ;;
    *) echo "用法: bash start_rag.sh {start|stop [--all]|restart [--all]|status}"; exit 1 ;;
esac
