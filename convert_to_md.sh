#!/bin/bash
# 文件格式转换脚本：使用 MinerU/Marker/Docling 将文档转换为 Markdown
# 用法:
#   bash convert_to_md.sh                        # 增量转换（仅处理新增/修改文件）
#   bash convert_to_md.sh --full                 # 全量转换
#   bash convert_to_md.sh --engine mineru        # 指定转换引擎
#   bash convert_to_md.sh --backend vlm          # 指定 MinerU 后端
#   bash convert_to_md.sh --device cpu           # 指定运行设备
#   bash convert_to_md.sh --dry-run              # 仅预览，不执行
#   bash convert_to_md.sh --help                 # 显示帮助信息

set -euo pipefail

# ========== 基础配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
STATE_FILE="${SCRIPT_DIR}/data/.convert_state"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/convert_to_md.log"
LOCK_FILE="/tmp/convert_to_md.lock"

# ========== 颜色定义 ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ========== 日志函数 ==========
log_info() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${GREEN}[INFO]${NC} [${timestamp}] $*"
    echo "[INFO] [${timestamp}] $*" >> "${LOG_FILE}"
}

log_warn() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${YELLOW}[WARN]${NC} [${timestamp}] $*"
    echo "[WARN] [${timestamp}] $*" >> "${LOG_FILE}"
}

log_error() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${RED}[ERROR]${NC} [${timestamp}] $*" >&2
    echo "[ERROR] [${timestamp}] $*" >> "${LOG_FILE}"
}

log_debug() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${BLUE}[DEBUG]${NC} [${timestamp}] $*"
    echo "[DEBUG] [${timestamp}] $*" >> "${LOG_FILE}"
}

# ========== 初始化目录 ==========
mkdir -p "${LOG_DIR}"
mkdir -p "${SCRIPT_DIR}/data"

# ========== 帮助信息 ==========
show_help() {
    cat << 'EOF'
文件格式转换脚本 - 使用 MinerU/Marker/Docling 将文档转换为 Markdown

用法:
  bash convert_to_md.sh [选项]

选项:
  --source DIR       源目录（默认从 .env 读取 DOCS_PATH）
  --output DIR       输出目录（默认与源目录相同，.md 文件放在源文件旁边）
  --engine ENGINE    指定转换引擎：mineru|marker|docling（默认自动选择）
  --backend BACKEND  MinerU 后端：pipeline|hybrid|vlm（默认 pipeline）
  --device DEVICE    运行设备：cuda:0|cpu（默认 cuda:0）
  --full             强制全量转换（忽略时间戳，重新转换所有文件）
  --dry-run          仅预览将要转换的文件，不实际执行
  --help             显示帮助信息

转换引擎优先级（自动选择模式）:
  1. MinerU  - 精度最高，支持 PDF/DOCX/PPTX/XLSX/图片
  2. Marker  - PDF 专用高质量转换
  3. Docling - 多格式通用转换

支持的文件格式:
  .pdf / .docx / .doc / .pptx / .ppt

增量逻辑:
  - 通过状态文件 data/.convert_state 记录每个文件的转换时间戳
  - 如果源文件未修改且对应 .md 已存在，则跳过
  - 使用 --full 可强制重新转换所有文件

示例:
  bash convert_to_md.sh                                    # 增量转换（自动选引擎）
  bash convert_to_md.sh --full                             # 全量转换
  bash convert_to_md.sh --engine mineru --backend vlm      # 使用 MinerU VLM 后端
  bash convert_to_md.sh --source ./data --full             # 指定目录全量转换
  bash convert_to_md.sh --dry-run                          # 预览将要转换的文件
  bash convert_to_md.sh --device cpu                       # 使用 CPU 转换
EOF
}

# ========== 参数解析 ==========
SOURCE_DIR=""
OUTPUT_DIR=""
ENGINE=""
BACKEND="pipeline"
DEVICE="cuda:0"
FULL_MODE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            SOURCE_DIR="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --engine)
            ENGINE="$2"
            shift 2
            ;;
        --backend)
            BACKEND="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --full)
            FULL_MODE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# ========== 从 .env 读取配置 ==========
if [[ -z "${SOURCE_DIR}" ]]; then
    if [[ -f "${ENV_FILE}" ]]; then
        while IFS='=' read -r key value; do
            [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]] && continue
            key="$(echo "$key" | xargs)"
            value="$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")"
            case "$key" in
                DOCS_PATH) SOURCE_DIR="$value" ;;
            esac
        done < "${ENV_FILE}"
    fi
    SOURCE_DIR="${SOURCE_DIR:-/mnt/cpu_share}"
fi

# 输出目录默认与源目录相同
OUTPUT_DIR="${OUTPUT_DIR:-${SOURCE_DIR}}"

# ========== GPU 环境设置 ==========
setup_gpu_env() {
    if [[ "$DEVICE" == "cpu" ]]; then
        export CUDA_VISIBLE_DEVICES=""
        log_info "运行设备: CPU"
    else
        # 从 cuda:X 格式提取 GPU 编号
        local gpu_id="${DEVICE#cuda:}"
        gpu_id="${gpu_id:-0}"
        export CUDA_VISIBLE_DEVICES="$gpu_id"
        log_info "运行设备: GPU ${gpu_id} (CUDA_VISIBLE_DEVICES=${gpu_id})"
    fi
}

# ========== 文件锁防止并发执行 ==========
exec 200>"${LOCK_FILE}"
if ! flock -n 200 2>/dev/null; then
    log_error "另一个转换实例正在运行，退出"
    exit 1
fi

# ========== 依赖检测与引擎选择 ==========
AVAILABLE_ENGINES=()

detect_engines() {
    # 检测 MinerU
    if command -v mineru &>/dev/null; then
        AVAILABLE_ENGINES+=("mineru")
        log_debug "检测到引擎: MinerU ($(command -v mineru))"
    fi

    # 检测 Marker
    if command -v marker_single &>/dev/null || command -v marker &>/dev/null; then
        AVAILABLE_ENGINES+=("marker")
        local marker_cmd=""
        if command -v marker_single &>/dev/null; then
            marker_cmd="$(command -v marker_single)"
        else
            marker_cmd="$(command -v marker)"
        fi
        log_debug "检测到引擎: Marker (${marker_cmd})"
    fi

    # 检测 Docling
    if command -v docling &>/dev/null; then
        AVAILABLE_ENGINES+=("docling")
        log_debug "检测到引擎: Docling ($(command -v docling))"
    fi
}

select_engine() {
    # 如果用户指定了引擎，验证是否可用
    if [[ -n "$ENGINE" ]]; then
        local found=false
        for e in "${AVAILABLE_ENGINES[@]}"; do
            if [[ "$e" == "$ENGINE" ]]; then
                found=true
                break
            fi
        done
        if [[ "$found" != "true" ]]; then
            log_error "指定的引擎 '${ENGINE}' 不可用"
            show_install_hints "$ENGINE"
            exit 1
        fi
        return
    fi

    # 自动选择：按优先级 mineru > marker > docling
    if [[ ${#AVAILABLE_ENGINES[@]} -eq 0 ]]; then
        log_error "未检测到任何可用的转换引擎！"
        echo ""
        echo -e "${YELLOW}请安装以下任一转换引擎：${NC}"
        echo ""
        echo "  MinerU（推荐，精度最高）："
        echo "    pip install uv && uv pip install -U \"mineru[all]\""
        echo ""
        echo "  Marker（PDF 专用）："
        echo "    pip install marker-pdf[full]"
        echo ""
        echo "  Docling（通用）："
        echo "    pip install docling"
        echo ""
        exit 1
    fi

    # 选择第一个可用引擎（已按优先级排列）
    ENGINE="${AVAILABLE_ENGINES[0]}"
}

show_install_hints() {
    local engine="$1"
    echo ""
    echo -e "${YELLOW}安装方法：${NC}"
    case "$engine" in
        mineru)
            echo "  pip install uv && uv pip install -U \"mineru[all]\""
            ;;
        marker)
            echo "  pip install marker-pdf[full]"
            ;;
        docling)
            echo "  pip install docling"
            ;;
    esac
    echo ""
}

# ========== 转换函数 ==========

# MinerU 转换单个文件
convert_with_mineru() {
    local source_file="$1"
    local target_md="$2"

    local tmp_dir
    tmp_dir="$(mktemp -d)"

    local basename_noext
    basename_noext="$(basename "$source_file" | sed 's/\.[^.]*$//')"

    # 调用 MinerU 转换
    if ! mineru -p "$source_file" -o "$tmp_dir" -b "$BACKEND" >> "${LOG_FILE}" 2>&1; then
        rm -rf "$tmp_dir"
        return 1
    fi

    # MinerU 输出结构：$tmp_dir/<filename>/<filename>.md
    local md_file="${tmp_dir}/${basename_noext}/${basename_noext}.md"

    # 如果标准路径不存在，尝试搜索
    if [[ ! -f "$md_file" ]]; then
        md_file="$(find "$tmp_dir" -name "*.md" -type f | head -1)"
    fi

    if [[ -z "$md_file" || ! -f "$md_file" ]]; then
        log_error "MinerU 未生成 .md 文件: ${source_file}"
        rm -rf "$tmp_dir"
        return 1
    fi

    # 确保目标目录存在
    mkdir -p "$(dirname "$target_md")"

    # 移动到目标位置
    mv "$md_file" "$target_md"

    # 清理临时目录
    rm -rf "$tmp_dir"
    return 0
}

# Marker 转换单个文件
convert_with_marker() {
    local source_file="$1"
    local target_md="$2"

    local tmp_dir
    tmp_dir="$(mktemp -d)"

    local basename_noext
    basename_noext="$(basename "$source_file" | sed 's/\.[^.]*$//')"

    # 设置 Marker GPU 环境
    export TORCH_DEVICE="${DEVICE}"

    # 优先使用 marker_single
    local marker_cmd=""
    if command -v marker_single &>/dev/null; then
        marker_cmd="marker_single"
    elif command -v marker &>/dev/null; then
        marker_cmd="marker"
    fi

    if [[ -z "$marker_cmd" ]]; then
        rm -rf "$tmp_dir"
        return 1
    fi

    # 调用 Marker
    if ! "$marker_cmd" "$source_file" --output_dir "$tmp_dir" --output_format markdown >> "${LOG_FILE}" 2>&1; then
        rm -rf "$tmp_dir"
        return 1
    fi

    # Marker 输出结构：$tmp_dir/<filename>/<filename>.md
    local md_file="${tmp_dir}/${basename_noext}/${basename_noext}.md"

    # 如果标准路径不存在，尝试搜索
    if [[ ! -f "$md_file" ]]; then
        md_file="$(find "$tmp_dir" -name "*.md" -type f | head -1)"
    fi

    if [[ -z "$md_file" || ! -f "$md_file" ]]; then
        log_error "Marker 未生成 .md 文件: ${source_file}"
        rm -rf "$tmp_dir"
        return 1
    fi

    mkdir -p "$(dirname "$target_md")"
    mv "$md_file" "$target_md"
    rm -rf "$tmp_dir"
    return 0
}

# Docling 转换单个文件
convert_with_docling() {
    local source_file="$1"
    local target_md="$2"

    local tmp_dir
    tmp_dir="$(mktemp -d)"

    local basename_noext
    basename_noext="$(basename "$source_file" | sed 's/\.[^.]*$//')"

    # 调用 Docling
    if ! docling "$source_file" --to md --output "$tmp_dir" >> "${LOG_FILE}" 2>&1; then
        rm -rf "$tmp_dir"
        return 1
    fi

    # Docling 输出结构：$tmp_dir/<filename>.md
    local md_file="${tmp_dir}/${basename_noext}.md"

    # 如果标准路径不存在，尝试搜索
    if [[ ! -f "$md_file" ]]; then
        md_file="$(find "$tmp_dir" -name "*.md" -type f | head -1)"
    fi

    if [[ -z "$md_file" || ! -f "$md_file" ]]; then
        log_error "Docling 未生成 .md 文件: ${source_file}"
        rm -rf "$tmp_dir"
        return 1
    fi

    mkdir -p "$(dirname "$target_md")"
    mv "$md_file" "$target_md"
    rm -rf "$tmp_dir"
    return 0
}

# 统一转换入口
convert_file() {
    local source_file="$1"
    local target_md="$2"

    case "$ENGINE" in
        mineru)
            convert_with_mineru "$source_file" "$target_md"
            ;;
        marker)
            convert_with_marker "$source_file" "$target_md"
            ;;
        docling)
            convert_with_docling "$source_file" "$target_md"
            ;;
        *)
            log_error "未知引擎: ${ENGINE}"
            return 1
            ;;
    esac
}

# ========== 增量判断 ==========
needs_conversion() {
    local source_file="$1"
    local target_file="$2"

    # 全量模式：总是转换
    if [[ "$FULL_MODE" == "true" ]]; then
        return 0
    fi

    # 目标文件不存在：需要转换
    if [[ ! -f "$target_file" ]]; then
        return 0
    fi

    # 检查状态文件中的记录
    if [[ -f "$STATE_FILE" ]]; then
        local source_mtime
        source_mtime="$(stat -c '%Y' "$source_file" 2>/dev/null || stat -f '%m' "$source_file" 2>/dev/null)"
        local recorded_mtime
        recorded_mtime="$(grep -F "$source_file" "$STATE_FILE" 2>/dev/null | cut -d'|' -f2)"

        if [[ -n "$recorded_mtime" && "$source_mtime" == "$recorded_mtime" ]]; then
            # 文件未修改，跳过
            return 1
        fi
    fi

    # 比较修改时间：源文件比目标文件新则需要转换
    if [[ "$source_file" -nt "$target_file" ]]; then
        return 0
    fi

    # 不需要转换
    return 1
}

# 更新状态文件中的记录
update_state() {
    local source_file="$1"
    local source_mtime
    source_mtime="$(stat -c '%Y' "$source_file" 2>/dev/null || stat -f '%m' "$source_file" 2>/dev/null)"

    # 移除旧记录（如果有）
    if [[ -f "$STATE_FILE" ]]; then
        grep -v -F "$source_file" "$STATE_FILE" > "${STATE_FILE}.tmp" 2>/dev/null || true
        mv "${STATE_FILE}.tmp" "$STATE_FILE"
    fi

    # 添加新记录
    echo "${source_file}|${source_mtime}" >> "$STATE_FILE"
}

# ========== 主流程 ==========

# 检测可用引擎
detect_engines

# 选择引擎
select_engine

# 设置 GPU 环境
setup_gpu_env

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          文档转 Markdown 工具 (MinerU/Marker/Docling)    ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

log_info "========== 开始文件格式转换 =========="
log_info "源目录: ${SOURCE_DIR}"
log_info "输出目录: ${OUTPUT_DIR}"
log_info "转换引擎: ${ENGINE}"
[[ "$ENGINE" == "mineru" ]] && log_info "MinerU 后端: ${BACKEND}"
log_info "运行设备: ${DEVICE}"
log_info "模式: $([ "$FULL_MODE" == "true" ] && echo "全量转换" || echo "增量转换")"
[[ "$DRY_RUN" == "true" ]] && log_info "DRY-RUN 模式：仅预览，不实际转换"

echo ""
echo -e "  引擎:   ${GREEN}${ENGINE}${NC}"
[[ "$ENGINE" == "mineru" ]] && echo -e "  后端:   ${GREEN}${BACKEND}${NC}"
echo -e "  设备:   ${GREEN}${DEVICE}${NC}"
echo -e "  模式:   ${GREEN}$([ "$FULL_MODE" == "true" ] && echo "全量" || echo "增量")${NC}"
echo ""

# 验证源目录
if [[ ! -d "${SOURCE_DIR}" ]]; then
    log_error "源目录不存在: ${SOURCE_DIR}"
    exit 1
fi

# 显示可用引擎信息
if [[ ${#AVAILABLE_ENGINES[@]} -gt 1 ]]; then
    log_debug "可用引擎: ${AVAILABLE_ENGINES[*]}"
fi

# 创建输出目录（如果与源目录不同）
if [[ "${OUTPUT_DIR}" != "${SOURCE_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi

# ========== 收集需要转换的文件 ==========
declare -a files_to_convert=()
declare -a targets=()

while IFS= read -r -d '' file; do
    # 计算相对路径
    local_relative="${file#${SOURCE_DIR}/}"
    # 计算目标文件路径（替换扩展名为 .md）
    target_relative="${local_relative%.*}.md"
    target_file="${OUTPUT_DIR}/${target_relative}"

    # 检查是否需要转换
    if needs_conversion "$file" "$target_file"; then
        files_to_convert+=("$file")
        targets+=("$target_file")
    fi
done < <(find "${SOURCE_DIR}" -type f \( \
    -iname "*.pdf" -o \
    -iname "*.docx" -o \
    -iname "*.doc" -o \
    -iname "*.pptx" -o \
    -iname "*.ppt" \
\) -print0 2>/dev/null)

# 统计信息
total_found=${#files_to_convert[@]}

if [[ ${total_found} -eq 0 ]]; then
    log_info "没有需要转换的文件（所有文件已是最新）"
    log_info "========== 转换结束 =========="
    exit 0
fi

log_info "发现 ${total_found} 个文件需要转换"

# ========== DRY-RUN 模式 ==========
if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo -e "${BLUE}将要转换的文件（共 ${total_found} 个）：${NC}"
    echo ""
    for i in "${!files_to_convert[@]}"; do
        local_relative="${files_to_convert[$i]#${SOURCE_DIR}/}"
        target_relative="${targets[$i]#${OUTPUT_DIR}/}"
        echo -e "  ${GREEN}→${NC} ${local_relative}"
        echo -e "    ${BLUE}输出:${NC} ${target_relative}"
    done
    echo ""
    echo -e "${CYAN}引擎: ${ENGINE} | 设备: ${DEVICE}${NC}"
    echo ""
    echo "如需执行转换，请去掉 --dry-run 参数"
    exit 0
fi

# ========== 执行转换 ==========
success_count=0
fail_count=0
skip_count=0
declare -a failed_files=()

log_info "开始转换，使用引擎: ${ENGINE}"
echo ""

for i in "${!files_to_convert[@]}"; do
    source_file="${files_to_convert[$i]}"
    target_file="${targets[$i]}"
    local_relative="${source_file#${SOURCE_DIR}/}"

    # 确保目标目录存在
    target_dir="$(dirname "$target_file")"
    mkdir -p "$target_dir"

    echo -e "  [$(( i + 1 ))/${total_found}] ${BLUE}转换:${NC} ${local_relative}"

    # 执行转换（在子 shell 中运行，避免 set -e 导致整体退出）
    if ( convert_file "$source_file" "$target_file" ); then
        if [[ -f "$target_file" && -s "$target_file" ]]; then
            success_count=$(( success_count + 1 ))
            # 更新状态记录
            update_state "$source_file"
            echo -e "           ${GREEN}✓ 成功${NC} → $(basename "$target_file")"
            log_info "[$(( i + 1 ))/${total_found}] 成功: ${local_relative}"
        else
            fail_count=$(( fail_count + 1 ))
            failed_files+=("$local_relative")
            echo -e "           ${RED}✗ 转换产生空文件${NC}"
            log_error "[$(( i + 1 ))/${total_found}] 空文件: ${local_relative}"
            # 删除空文件
            rm -f "$target_file"
        fi
    else
        fail_count=$(( fail_count + 1 ))
        failed_files+=("$local_relative")
        echo -e "           ${RED}✗ 转换失败${NC}"
        log_error "[$(( i + 1 ))/${total_found}] 失败: ${local_relative}"
    fi
done

# ========== 输出汇总 ==========
echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                     转换完成汇总                         ${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  引擎:   ${ENGINE}"
echo -e "  设备:   ${DEVICE}"
echo -e "  ${GREEN}成功:   ${success_count} 个文件${NC}"
if [[ ${fail_count} -gt 0 ]]; then
    echo -e "  ${RED}失败:   ${fail_count} 个文件${NC}"
fi
echo -e "  总计:   ${total_found} 个文件"
echo ""

if [[ ${fail_count} -gt 0 ]]; then
    echo -e "${RED}失败文件列表：${NC}"
    for f in "${failed_files[@]}"; do
        echo -e "  ${RED}✗${NC} ${f}"
    done
    echo ""
fi

log_info "转换汇总: 成功=${success_count}, 失败=${fail_count}, 总计=${total_found}"
log_info "========== 转换结束 =========="

# ========== 后续操作提示 ==========
if [[ ${success_count} -gt 0 ]]; then
    echo -e "${GREEN}转换完成！可执行以下命令进行知识入库：${NC}"
    echo "  bash auto_ingest.sh        # 增量入库（推荐）"
    echo "  bash auto_ingest.sh --full # 全量入库"
    echo ""
fi

# 如果有失败但也有成功，返回 0；如果全部失败返回 1
if [[ ${success_count} -eq 0 && ${fail_count} -gt 0 ]]; then
    exit 1
fi

exit 0
