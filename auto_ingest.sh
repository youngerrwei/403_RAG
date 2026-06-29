#!/bin/bash
# 知识库自动入库脚本
# 用法:
#   bash auto_ingest.sh            # 增量入库（检测新增/修改文件）
#   bash auto_ingest.sh --full     # 全量入库（重建集合 + 入库）
#   bash auto_ingest.sh --destroy  # 仅销毁知识库（删除 Qdrant 集合 + 清理状态）
#   bash auto_ingest.sh --destroy --force  # 跳过确认直接销毁

set -euo pipefail

# ========== 基础配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
STATE_FILE="${SCRIPT_DIR}/data/.ingest_state"
MANIFEST_FILE="${SCRIPT_DIR}/data/.ingest_manifest"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/auto_ingest.log"
LOCK_FILE="/tmp/auto_ingest.lock"

# ========== 日志函数 ==========
log() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo "[${timestamp}] $*" | tee -a "${LOG_FILE}"
}

# ========== 初始化日志目录 ==========
mkdir -p "${LOG_DIR}"
mkdir -p "${SCRIPT_DIR}/data"

# ========== 文件锁防止并发执行 ==========
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    log "ERROR: 另一个实例正在运行，退出"
    exit 1
fi

# ========== 从 .env 读取配置 ==========
if [[ ! -f "${ENV_FILE}" ]]; then
    log "ERROR: .env 文件不存在: ${ENV_FILE}"
    exit 1
fi

# 解析 .env 文件中的配置
DOCS_PATH=""
QDRANT_HOST=""
QDRANT_PORT=""
COLLECTION_NAME=""
PARENT_COLLECTION=""
while IFS='=' read -r key value; do
    # 跳过注释和空行
    [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
    # 去除首尾空格
    key="$(echo "$key" | xargs)"
    value="$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
    case "$key" in
        DOCS_PATH) DOCS_PATH="$value" ;;
        QDRANT_HOST) QDRANT_HOST="$value" ;;
        QDRANT_PORT) QDRANT_PORT="$value" ;;
        QDRANT_COLLECTION_NAME) COLLECTION_NAME="$value" ;;
        QDRANT_PARENT_COLLECTION) PARENT_COLLECTION="$value" ;;
    esac
done < "${ENV_FILE}"

# 设置默认值
DOCS_PATH="${DOCS_PATH:-./data}"
QDRANT_HOST="${QDRANT_HOST:-127.0.0.1}"
QDRANT_PORT="${QDRANT_PORT:-6333}"
COLLECTION_NAME="${COLLECTION_NAME:-lab_knowledge_base}"
PARENT_COLLECTION="${PARENT_COLLECTION:-lab_knowledge_base_parents}"

# ========== 文件删除检测函数 ==========
# 对比 manifest 文件与当前文件列表，检测是否有文件被删除
check_deleted_files() {
    if [[ ! -f "$MANIFEST_FILE" ]]; then
        return 0  # 无历史记录，跳过检测
    fi

    local deleted_files=()
    while IFS= read -r file; do
        if [[ ! -f "$file" ]]; then
            deleted_files+=("$file")
        fi
    done < "$MANIFEST_FILE"

    if [[ ${#deleted_files[@]} -gt 0 ]]; then
        log "WARNING: 检测到 ${#deleted_files[@]} 个文件已被删除："
        for f in "${deleted_files[@]}"; do
            log "  - $f"
        done
        log "INFO: 将触发全量重建以清除残留向量"
        return 1  # 需要全量重建
    fi
    return 0
}

# 保存当前文件清单到 manifest 文件
save_manifest() {
    find "${DOCS_PATH}" -name "*.md" -type f | sort > "$MANIFEST_FILE"
    log "INFO: 已更新文件清单 ($(wc -l < "$MANIFEST_FILE" | xargs) 个文件)"
}

# ========== --destroy 参数：销毁知识库 ==========
if [[ "${1:-}" == "--destroy" ]]; then
    FORCE=false
    if [[ "${2:-}" == "--force" ]]; then
        FORCE=true
    fi

    echo "⚠️  即将销毁知识库，删除以下集合："
    echo "   - ${COLLECTION_NAME}"
    echo "   - ${PARENT_COLLECTION}"
    echo ""

    if [[ "$FORCE" != "true" ]]; then
        read -p "确认销毁？(y/N): " confirm
        if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
            echo "已取消"
            exit 0
        fi
    fi

    echo "[1/3] 删除子块集合..."
    response=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${COLLECTION_NAME}")
    if [[ "$response" == "200" || "$response" == "404" ]]; then
        echo "  ✓ 子块集合已删除"
    else
        echo "  ✗ 删除失败 (HTTP $response)"
    fi

    echo "[2/3] 删除父块集合..."
    response=$(curl -s -o /dev/null -w "%{http_code}" -X DELETE "http://${QDRANT_HOST}:${QDRANT_PORT}/collections/${PARENT_COLLECTION}")
    if [[ "$response" == "200" || "$response" == "404" ]]; then
        echo "  ✓ 父块集合已删除"
    else
        echo "  ✗ 删除失败 (HTTP $response)"
    fi

    echo "[3/3] 清理本地状态..."
    rm -f "${STATE_FILE}"
    rm -f "${MANIFEST_FILE}"
    echo "  ✓ 入库状态已清理"

    echo ""
    echo "✅ 知识库已销毁。如需重新入库，请运行："
    echo "   bash auto_ingest.sh --full"
    exit 0
fi

# ========== 验证 DOCS_PATH ==========
if [[ -z "${DOCS_PATH}" ]]; then
    log "ERROR: .env 中未找到 DOCS_PATH 配置"
    exit 1
fi

if [[ ! -d "${DOCS_PATH}" ]]; then
    log "ERROR: DOCS_PATH 目录不存在: ${DOCS_PATH}"
    exit 1
fi

log "========== 开始自动入库检测 =========="
log "DOCS_PATH: ${DOCS_PATH}"

# ========== 首次运行处理 ==========
if [[ ! -f "${STATE_FILE}" ]]; then
    log "INFO: 状态文件不存在（首次运行）"

    if [[ "${1:-}" == "--full" ]]; then
        log "INFO: 检测到 --full 参数，执行全量入库"
        export QDRANT_RECREATE_COLLECTION=true

        cd "${SCRIPT_DIR}"
        if python ingest.py >> "${LOG_FILE}" 2>&1; then
            touch "${STATE_FILE}"
            save_manifest
            log "SUCCESS: 全量入库完成，状态文件已创建"
        else
            log "ERROR: 全量入库失败，状态文件未更新"
            exit 1
        fi
    else
        log "INFO: 首次运行未指定 --full 参数，创建状态文件并跳过本次入库"
        log "INFO: 如需全量入库，请运行: $0 --full"
        touch "${STATE_FILE}"
    fi

    log "========== 本次运行结束 =========="
    exit 0
fi

# ========== 增量入库检测 ==========
# 检测文件删除，如有删除则自动切换为全量重建模式
if ! check_deleted_files; then
    log "INFO: 检测到文件删除，切换为全量重建模式"
    export QDRANT_RECREATE_COLLECTION=true

    cd "${SCRIPT_DIR}"
    if python ingest.py >> "${LOG_FILE}" 2>&1; then
        touch "${STATE_FILE}"
        save_manifest
        log "SUCCESS: 全量重建完成（由文件删除触发）"
    else
        log "ERROR: 全量重建失败"
        exit 1
    fi

    log "========== 本次运行结束 =========="
    exit 0
fi

NEW_FILES=$(find "${DOCS_PATH}" -name "*.md" -newer "${STATE_FILE}" 2>/dev/null)
NEW_FILE_COUNT=$(echo "${NEW_FILES}" | grep -c '.' 2>/dev/null || echo "0")

if [[ -z "${NEW_FILES}" ]]; then
    NEW_FILE_COUNT=0
fi

log "INFO: 检测到 ${NEW_FILE_COUNT} 个新增/修改的 .md 文件"

if [[ ${NEW_FILE_COUNT} -eq 0 ]]; then
    log "INFO: 无新文件，跳过入库"
    log "========== 本次运行结束 =========="
    exit 0
fi

# 列出新文件
log "INFO: 新增/修改的文件列表:"
echo "${NEW_FILES}" | while read -r f; do
    log "  - ${f}"
done

# ========== 执行增量入库 ==========
# 覆盖环境变量，确保增量模式（不删除旧数据）
export QDRANT_RECREATE_COLLECTION=false

log "INFO: 开始执行增量入库 (QDRANT_RECREATE_COLLECTION=false)..."

cd "${SCRIPT_DIR}"
if python ingest.py >> "${LOG_FILE}" 2>&1; then
    # 入库成功，更新状态文件时间戳
    touch "${STATE_FILE}"
    save_manifest
    log "SUCCESS: 增量入库完成，状态文件已更新"
else
    # 入库失败，不更新状态文件（下次仍会重试）
    log "ERROR: 增量入库失败，状态文件未更新（下次将重试）"
    exit 1
fi

log "========== 本次运行结束 =========="

# ========== Cron 配置说明 ==========
# cron 配置示例（每天凌晨 3 点执行）：
# 0 3 * * * /Users/weiziyang/Documents/code/403_RAG/auto_ingest.sh >> /Users/weiziyang/Documents/code/403_RAG/logs/auto_ingest_cron.log 2>&1
