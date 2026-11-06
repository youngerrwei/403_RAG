#!/bin/bash
# 知识库变更触发式覆盖、全量入库与安全销毁；失败时不推进本地状态。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/runtime_common.sh"

ENV_FILE="${RAG_ENV_FILE:-$PROJECT_ROOT/.env}"
STATE_FILE="$PROJECT_ROOT/data/.ingest_state"
MANIFEST_FILE="$PROJECT_ROOT/data/.ingest_manifest"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/auto_ingest.log"
LOCK_FILE="$PROJECT_ROOT/data/.auto_ingest.lock"
mkdir -p "$LOG_DIR" "$PROJECT_ROOT/data"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE"
}

report_lock_holders() {
    local holders="" pid details
    if command -v lsof >/dev/null 2>&1; then
        holders="$(lsof -t "$LOCK_FILE" 2>/dev/null | sort -u | tr '\n' ' ')"
    elif command -v fuser >/dev/null 2>&1; then
        holders="$(fuser "$LOCK_FILE" 2>/dev/null | tr ' ' '\n' | sort -u | tr '\n' ' ')"
    fi
    if [[ -z "${holders// }" ]]; then
        log "INFO: 未找到锁持有者详情；可执行 lsof $LOCK_FILE 或 fuser $LOCK_FILE"
        return
    fi
    for pid in $holders; do
        [[ "$pid" =~ ^[0-9]+$ ]] || continue
        details="$(ps -p "$pid" -o pid=,ppid=,etime=,stat=,args= 2>/dev/null || true)"
        [[ -n "$details" ]] && log "INFO: 锁持有者: $details"
    done
}

[[ -f "$ENV_FILE" ]] || { log "ERROR: .env 不存在: $ENV_FILE"; exit 1; }
load_env_keys "$ENV_FILE" DOCS_PATH QDRANT_HOST QDRANT_PORT QDRANT_COLLECTION_NAME \
    QDRANT_PARENT_COLLECTION RAG_CONDA_ENV STARTUP_PROBE_TIMEOUT || true
DOCS_PATH="${DOCS_PATH:-/mnt/cpu_share}"
[[ "$DOCS_PATH" = /* ]] || DOCS_PATH="$(resolve_project_path "$DOCS_PATH")"
QDRANT_HOST="${QDRANT_HOST:-172.18.216.71}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
COLLECTION_NAME="${QDRANT_COLLECTION_NAME:-lab_knowledge_base}"
PARENT_COLLECTION="${QDRANT_PARENT_COLLECTION:-lab_knowledge_base_parents}"
RAG_CONDA_ENV="${RAG_CONDA_ENV:-rag}"
STARTUP_PROBE_TIMEOUT="${STARTUP_PROBE_TIMEOUT:-5}"
CONDA_CMD="$(find_conda || true)"
RAG_PYTHON=""
[[ -n "$CONDA_CMD" ]] && RAG_PYTHON="$(conda_env_python "$CONDA_CMD" "$RAG_CONDA_ENV" || true)"

command -v flock >/dev/null 2>&1 || { log "ERROR: 未找到 flock"; exit 1; }
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    log "ERROR: 另一个入库实例正在运行"
    report_lock_holders
    exit 1
fi

save_manifest() {
    local tmp
    tmp="$(mktemp "${MANIFEST_FILE}.tmp.XXXXXX")"
    find "$DOCS_PATH" -name '*.md' -type f -print | sort > "$tmp"
    mv -f "$tmp" "$MANIFEST_FILE"
    log "INFO: 已更新文件清单 ($(wc -l < "$MANIFEST_FILE" | tr -d ' ') 个文件)"
}

check_deleted_files() {
    [[ -f "$MANIFEST_FILE" ]] || return 0
    local file deleted=0
    while IFS= read -r file; do
        if [[ -n "$file" && ! -f "$file" ]]; then
            log "WARNING: 文件已删除: $file"
            deleted=1
        fi
    done < "$MANIFEST_FILE"
    (( deleted == 0 ))
}

run_ingest() {
    local recreate="$1" started_at ingest_status
    [[ -n "$RAG_PYTHON" ]] || { log "ERROR: conda 环境不存在或无 Python: $RAG_CONDA_ENV"; return 1; }
    log "INFO: 使用解释器 $RAG_PYTHON 执行入库 (recreate=$recreate)"
    log "INFO: 入库阶段日志将实时显示，并同步写入 $LOG_FILE"
    started_at=$SECONDS
    (
        cd "$PROJECT_ROOT"
        export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
        PYTHONUNBUFFERED=1 QDRANT_RECREATE_COLLECTION="$recreate" \
            "$RAG_PYTHON" -u -m lab_rag.ingest
    ) 2>&1 | tee -a "$LOG_FILE"
    ingest_status=${PIPESTATUS[0]}
    if (( ingest_status == 0 )); then
        log "INFO: 入库子进程结束 (exit=$ingest_status, elapsed=$((SECONDS - started_at))s)"
    else
        log "ERROR: 入库子进程失败 (exit=$ingest_status, elapsed=$((SECONDS - started_at))s)"
    fi
    return "$ingest_status"
}

destroy_collection() {
    local collection="$1" code
    [[ -n "$RAG_PYTHON" ]] || { log "ERROR: conda 环境不存在或无 Python: $RAG_CONDA_ENV"; return 1; }
    code="$(http_request_status "$RAG_PYTHON" DELETE \
        "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${collection}" \
        "$STARTUP_PROBE_TIMEOUT" 2>/dev/null || true)"
    if [[ "$code" == "200" || "$code" == "404" ]]; then
        log "INFO: 集合已删除或不存在: $collection (HTTP $code)"
        return 0
    fi
    log "ERROR: 删除集合失败: $collection (HTTP ${code:-000})"
    return 1
}

case "${1:-}" in
    --destroy)
        destroy_failed=0
        if [[ "${2:-}" != "--force" ]]; then
            read -r -p "确认销毁集合 ${COLLECTION_NAME} 和 ${PARENT_COLLECTION}？(y/N): " confirm
            [[ "$confirm" == "y" || "$confirm" == "Y" ]] || { echo "已取消"; exit 0; }
        fi
        destroy_collection "$COLLECTION_NAME" || destroy_failed=1
        destroy_collection "$PARENT_COLLECTION" || destroy_failed=1
        if (( destroy_failed != 0 )); then
            log "ERROR: 集合未全部删除，本地状态保留"
            exit 1
        fi
        rm -f "$STATE_FILE" "$MANIFEST_FILE"
        log "SUCCESS: 知识库集合及本地状态已销毁"
        exit 0
        ;;
    --full)
        [[ -d "$DOCS_PATH" ]] || { log "ERROR: DOCS_PATH 不存在: $DOCS_PATH"; exit 1; }
        if run_ingest true; then
            touch "$STATE_FILE"
            save_manifest
            log "SUCCESS: 全量入库完成"
            exit 0
        fi
        log "ERROR: 全量入库失败，本地状态未更新"
        exit 1
        ;;
    "") ;;
    *) echo "用法: bash scripts/auto_ingest.sh [--full|--destroy [--force]]"; exit 1 ;;
esac

[[ -d "$DOCS_PATH" ]] || { log "ERROR: DOCS_PATH 不存在: $DOCS_PATH"; exit 1; }
if [[ ! -f "$STATE_FILE" ]]; then
    log "ERROR: 首次运行必须执行: bash scripts/auto_ingest.sh --full"
    exit 1
fi

if ! check_deleted_files; then
    log "INFO: 检测到文件删除，按既有策略执行全量重建"
    if run_ingest true; then
        touch "$STATE_FILE"
        save_manifest
        log "SUCCESS: 删除文件触发的全量重建完成"
        exit 0
    fi
    log "ERROR: 全量重建失败，本地状态未更新"
    exit 1
fi

mapfile -d '' changed_files < <(find "$DOCS_PATH" -name '*.md' -type f -newer "$STATE_FILE" -print0)

# 仅依赖 mtime 会漏掉“保留旧时间戳后复制进来”的新文件；必须同时与上次清单比对路径。
declare -A changed_seen=()
declare -A manifest_paths=()
for file in "${changed_files[@]}"; do
    changed_seen["$file"]=1
done
if [[ -f "$MANIFEST_FILE" ]]; then
    while IFS= read -r file; do
        [[ -n "$file" ]] && manifest_paths["$file"]=1
    done < "$MANIFEST_FILE"
fi
while IFS= read -r -d '' file; do
    if [[ -z "${manifest_paths[$file]+x}" && -z "${changed_seen[$file]+x}" ]]; then
        changed_files+=("$file")
        changed_seen["$file"]=1
    fi
done < <(find "$DOCS_PATH" -name '*.md' -type f -print0)

if (( ${#changed_files[@]} == 0 )); then
    log "INFO: 无新增或修改文件，跳过入库"
    exit 0
fi
log "INFO: 检测到 ${#changed_files[@]} 个新增/修改文件"
for file in "${changed_files[@]}"; do log "  - $file"; done

if run_ingest false; then
    touch "$STATE_FILE"
    save_manifest
    log "SUCCESS: 变更触发式幂等覆盖完成"
    exit 0
fi
log "ERROR: 增量入库失败，本地状态未更新，下次将重试"
exit 1
