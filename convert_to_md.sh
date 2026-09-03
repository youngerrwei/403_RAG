#!/bin/bash
# 文件格式转换脚本：使用 MinerU/Marker/Docling 原子转换为 Markdown
# 运行环境：MinerU 运行在独立的 conda 环境 rag-mineru（避免与 vLLM 的依赖冲突）
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
source "${SCRIPT_DIR}/scripts/runtime_common.sh"
load_env_keys "${ENV_FILE}" MINERU_CONDA_ENV CONVERT_TIMEOUT CONVERT_MAX_RETRIES MAX_FILE_SIZE_MB MIN_OUTPUT_SIZE || true

# ========== 兜底机制配置 ==========
CONVERT_TIMEOUT="${CONVERT_TIMEOUT:-300}"          # 单文件转换超时（秒）
CONVERT_MAX_RETRIES="${CONVERT_MAX_RETRIES:-2}"    # 最大重试次数
MAX_FILE_SIZE_MB="${MAX_FILE_SIZE_MB:-100}"         # 最大文件大小（MB）
MIN_OUTPUT_SIZE="${MIN_OUTPUT_SIZE:-10}"            # 输出文件最小有效大小（字节）

# ========== 文件跳过模式 ==========
# 以下模式的文件会在文件发现阶段被过滤掉
SKIP_PATTERNS=(
    '~$*'           # Office 临时/锁文件
    '.~*'           # 其他临时文件（如 .~lock.*）
    '*.tmp'         # 通用临时文件
    'Thumbs.db'     # Windows 缩略图缓存
    '.DS_Store'     # macOS 目录元数据
)

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

# ========== 检查 rag-mineru conda 环境 ==========
# MinerU 运行在独立的 conda 环境中，避免与 vLLM/RAG 主环境的 PyTorch 版本冲突
CONDA_ENV_NAME="${MINERU_CONDA_ENV:-rag-mineru}"

# 查找 conda 命令
CONDA_CMD="$(find_conda || true)"

if [[ -z "$CONDA_CMD" ]]; then
    log_error "未找到 conda 命令，请先安装 Miniconda/Anaconda"
    exit 1
fi

# 检查 rag-mineru 环境是否存在
if ! $CONDA_CMD env list 2>/dev/null | grep -qE "^${CONDA_ENV_NAME}\s"; then
    log_error "conda 环境 '${CONDA_ENV_NAME}' 不存在"
    echo -e "${YELLOW}[提示] 请先运行安装脚本创建环境:${NC}"
    echo "        bash setup_env.sh --mineru"
    exit 1
fi

log_info "使用 conda 环境: ${CONDA_ENV_NAME}"

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
  文档: .pdf / .docx / .doc / .pptx / .ppt
  代码: .py .pyx .pyi .m .c .h .cpp .hpp .cc .cxx .java .js .jsx .ts .tsx
        .go .rs .rb .sh .bash .zsh .pl .pm .r .R .scala .kt .kts .swift
        .lua .sql .html .htm .css .scss .less .json .yaml .yml .xml .toml
        .ini .cfg .conf .tex .php .cs .dart .hs .ex .exs .proto .v .vhd
        .vhdl 以及 Makefile / Dockerfile
  Markdown: .md（直接复制）

环境变量:
  CONVERT_TIMEOUT       单文件转换超时秒数（默认 300）
  CONVERT_MAX_RETRIES   转换失败最大重试次数（默认 2）
  MAX_FILE_SIZE_MB      允许的最大文件大小 MB（默认 100）
  MIN_OUTPUT_SIZE       输出文件最小有效字节数（默认 10）

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

# ========== 并发保护（内核文件锁） ==========
command -v flock &>/dev/null || { log_error "未找到 flock，无法保证转换状态一致性"; exit 1; }
exec 200>"${LOCK_FILE}"
if ! flock -n 200; then
    log_error "另一个转换实例正在运行，退出"
    exit 1
fi

# ========== 代码文件扩展名与语言映射 ==========
# 返回扩展名对应的 Markdown 代码块语言标识
get_code_language() {
    local filename="$1"
    local basename_file
    basename_file="$(basename "$filename")"
    local ext="${basename_file##*.}"

    # 处理无扩展名的特殊文件
    if [[ "$basename_file" == "Makefile" || "$basename_file" == "makefile" || "$basename_file" == "GNUmakefile" ]]; then
        echo "makefile"
        return 0
    fi
    if [[ "$basename_file" == "Dockerfile" || "$basename_file" =~ ^Dockerfile\. ]]; then
        echo "dockerfile"
        return 0
    fi

    local ext_lower
    ext_lower="$(echo "$ext" | tr '[:upper:]' '[:lower:]')"
    case "${ext_lower}" in
        py|pyx|pyi)     echo "python" ;;
        m)              echo "matlab" ;;
        c)              echo "c" ;;
        h)              echo "c" ;;
        cpp|cc|cxx)     echo "cpp" ;;
        hpp)            echo "cpp" ;;
        java)           echo "java" ;;
        js|jsx)         echo "javascript" ;;
        ts|tsx)         echo "typescript" ;;
        go)             echo "go" ;;
        rs)             echo "rust" ;;
        rb)             echo "ruby" ;;
        sh|bash|zsh)    echo "bash" ;;
        pl|pm)          echo "perl" ;;
        r)              echo "r" ;;
        scala)          echo "scala" ;;
        kt|kts)         echo "kotlin" ;;
        swift)          echo "swift" ;;
        lua)            echo "lua" ;;
        sql)            echo "sql" ;;
        html|htm)       echo "html" ;;
        css)            echo "css" ;;
        scss)           echo "scss" ;;
        less)           echo "less" ;;
        json)           echo "json" ;;
        yaml|yml)       echo "yaml" ;;
        xml)            echo "xml" ;;
        toml)           echo "toml" ;;
        ini|cfg|conf)   echo "ini" ;;
        tex)            echo "latex" ;;
        php)            echo "php" ;;
        cs)             echo "csharp" ;;
        dart)           echo "dart" ;;
        hs)             echo "haskell" ;;
        ex|exs)         echo "elixir" ;;
        proto)          echo "protobuf" ;;
        v)              echo "verilog" ;;
        vhd|vhdl)       echo "vhdl" ;;
        *)              echo "" ;;  # 不支持的扩展名
    esac
}

# 判断文件是否为代码文件
is_code_file() {
    local filename="$1"
    local lang
    lang="$(get_code_language "$filename")"
    [[ -n "$lang" ]]
}

# 判断文件是否为 Markdown 文件
is_markdown_file() {
    local filename="$1"
    local ext="${filename##*.}"
    local ext_lower
    ext_lower="$(echo "$ext" | tr '[:upper:]' '[:lower:]')"
    [[ "${ext_lower}" == "md" ]]
}

# 判断文件是否为文档文件（需要引擎转换）
is_document_file() {
    local filename="$1"
    local ext="${filename##*.}"
    local ext_lower
    ext_lower="$(echo "$ext" | tr '[:upper:]' '[:lower:]')"
    case "${ext_lower}" in
        pdf|docx|doc|pptx|ppt) return 0 ;;
        *) return 1 ;;
    esac
}

# 获取文件类型描述
get_file_type_label() {
    local filename="$1"
    if is_document_file "$filename"; then
        echo "document"
    elif is_code_file "$filename"; then
        echo "code"
    elif is_markdown_file "$filename"; then
        echo "markdown"
    else
        echo "unknown"
    fi
}

# 获取文件大小（字节，跨平台兼容）
# 返回值：成功返回字节数，失败返回 -1 表示未知
get_file_size_bytes() {
    local file="$1"
    local size
    if [[ "$(uname)" == "Darwin" ]]; then
        size=$(stat -f %z "$file" 2>/dev/null)
    else
        size=$(stat -c %s "$file" 2>/dev/null)
    fi
    # 如果 stat 失败，用 wc -c 作为兜底
    if [[ -z "$size" ]]; then
        size=$(wc -c < "$file" 2>/dev/null | tr -d ' ')
    fi
    # 最终兜底：如果仍然获取失败，返回 -1 表示未知
    echo "${size:-"-1"}"
}

# 获取文件大小（MB，保留两位小数）
get_file_size_mb() {
    local file="$1"
    local size_bytes
    size_bytes="$(get_file_size_bytes "$file")"
    # 如果获取失败（-1），返回 0.00 避免后续计算出错
    if [[ "$size_bytes" == "-1" ]]; then
        echo "0.00"
        return
    fi
    echo "$size_bytes" | awk '{printf "%.2f", $1/1048576}'
}

# ========== 文件跳过判断 ==========
# 检查文件是否应该被跳过（临时文件、隐藏文件等）
should_skip_file() {
    local file="$1"
    local basename_file
    basename_file="$(basename "$file")"

    # Office 临时/锁文件：以 ~$ 开头
    if [[ "$basename_file" == '~$'* ]]; then
        log_debug "跳过 Office 临时文件: ${basename_file}"
        return 0
    fi

    # 以 .~ 开头的临时文件（如 .~lock.xxx）
    if [[ "$basename_file" == '.~'* ]]; then
        log_debug "跳过临时文件: ${basename_file}"
        return 0
    fi

    # 以 ~ 开头且以 .tmp 结尾的文件
    if [[ "$basename_file" == '~'* && "$basename_file" == *.tmp ]]; then
        log_debug "跳过临时文件: ${basename_file}"
        return 0
    fi

    # 隐藏文件（以 . 开头）
    if [[ "$basename_file" == .* ]]; then
        log_debug "跳过隐藏文件: ${basename_file}"
        return 0
    fi

    # 通用临时文件
    if [[ "$basename_file" == *.tmp ]]; then
        log_debug "跳过临时文件: ${basename_file}"
        return 0
    fi

    # Windows/macOS 系统文件
    if [[ "$basename_file" == "Thumbs.db" || "$basename_file" == ".DS_Store" ]]; then
        log_debug "跳过系统文件: ${basename_file}"
        return 0
    fi

    # 0 字节源文件只跳过，转换工具绝不能删除用户原始资料。
    local file_size
    file_size=$(get_file_size_bytes "$file")
    if [[ "$file_size" == "0" ]]; then
        log_warn "跳过 0 字节源文件: ${file}"
        return 0
    fi
    # 如果 file_size 为 -1 或获取失败，不跳过，继续处理

    return 1  # 不跳过
}

# ========== 文件大小检查 ==========
check_file_size() {
    local file="$1"
    local size_bytes
    size_bytes="$(get_file_size_bytes "$file")"
    # 如果无法获取文件大小，不阻止处理
    if [[ "$size_bytes" == "-1" ]]; then
        return 0
    fi
    local max_bytes=$(( MAX_FILE_SIZE_MB * 1048576 ))

    if [[ "$size_bytes" -gt "$max_bytes" ]]; then
        return 1  # 文件过大
    fi
    return 0
}

# ========== 输出文件有效性检查 ==========
validate_output_file() {
    local output_file="$1"

    # 文件不存在
    if [[ ! -f "$output_file" ]]; then
        return 1
    fi

    # 文件大小检查
    local size_bytes
    size_bytes="$(get_file_size_bytes "$output_file")"
    # 如果获取大小失败，假定文件有效（不误判）
    if [[ "$size_bytes" == "-1" ]]; then
        return 0
    fi
    if [[ "$size_bytes" -lt "$MIN_OUTPUT_SIZE" ]]; then
        return 1
    fi

    return 0
}

# ========== 依赖检测与引擎选择 ==========
AVAILABLE_ENGINES=()

detect_engines() {
    # 检测 MinerU（在 rag-mineru conda 环境中检查）
    if $CONDA_CMD run -n "$CONDA_ENV_NAME" mineru --help &>/dev/null 2>&1; then
        AVAILABLE_ENGINES+=("mineru")
        log_debug "检测到引擎: MinerU (在 ${CONDA_ENV_NAME} 环境中)"
    fi

    # 检测 Marker（在 rag-mineru conda 环境中检查）
    if $CONDA_CMD run -n "$CONDA_ENV_NAME" bash -c "command -v marker_single || command -v marker" &>/dev/null 2>&1; then
        AVAILABLE_ENGINES+=("marker")
        log_debug "检测到引擎: Marker (在 ${CONDA_ENV_NAME} 环境中)"
    fi

    # 检测 Docling（在 rag-mineru conda 环境中检查）
    if $CONDA_CMD run -n "$CONDA_ENV_NAME" docling --help &>/dev/null 2>&1; then
        AVAILABLE_ENGINES+=("docling")
        log_debug "检测到引擎: Docling (在 ${CONDA_ENV_NAME} 环境中)"
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
        log_warn "未检测到任何可用的文档转换引擎（代码文件和 MD 文件仍可处理）"
        ENGINE="none"
        return
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

# ========== 路径安全化辅助函数 ==========
# conda run 内部创建临时脚本时不会正确引用路径中的特殊字符（括号、空格等），
# 导致 bash 解析错误。此函数创建一个不含特殊字符的符号链接作为替代。
safe_source_path() {
    local source_file="$1"
    local basename_file
    basename_file="$(basename "$source_file")"

    # 检查文件名是否包含 shell 特殊字符
    if [[ "$basename_file" =~ [\(\)\[\]\{\}\$\!\#\&\;\|\<\>\`\'\"\ ] ]]; then
        # 创建安全的临时符号链接
        local ext="${source_file##*.}"
        local safe_name="doc_$(date +%s%N)_$$.${ext}"
        local safe_link="/tmp/${safe_name}"
        ln -sf "$(realpath "$source_file" 2>/dev/null || echo "$source_file")" "$safe_link"
        echo "$safe_link"
    else
        echo "$source_file"
    fi
}

# ========== 转换函数 ==========

# MinerU 转换单个文件（通过 conda run 在 rag-mineru 环境中执行）
convert_with_mineru() {
    local source_file="$1"
    local target_md="$2"

    # 安全化路径（处理文件名含括号等特殊字符的情况）
    local safe_path
    safe_path="$(safe_source_path "$source_file")"
    local need_cleanup_link=false
    [[ "$safe_path" != "$source_file" ]] && need_cleanup_link=true

    local tmp_dir
    tmp_dir="$(mktemp -d)"

    local basename_noext
    basename_noext="$(basename "$source_file" | sed 's/\.[^.]*$//')"

    # 通过 conda run 在 rag-mineru 环境中调用 MinerU（使用安全路径）
    if ! $CONDA_CMD run -n "$CONDA_ENV_NAME" mineru -p "$safe_path" -o "$tmp_dir" -b "$BACKEND" >> "${LOG_FILE}" 2>&1; then
        [[ "$need_cleanup_link" == true ]] && rm -f "$safe_path"
        rm -rf "$tmp_dir"
        return 1
    fi

    # 清理符号链接
    [[ "$need_cleanup_link" == true ]] && rm -f "$safe_path"

    # MinerU 输出结构：$tmp_dir/<filename>/<filename>.md
    local md_file="${tmp_dir}/${basename_noext}/${basename_noext}.md"

    # 如果标准路径不存在，递归查找所有 .md 文件
    if [[ ! -f "$md_file" ]]; then
        md_file="$(find "$tmp_dir" -name "*.md" -type f 2>/dev/null | head -1)"
    fi

    # 额外兜底：搜索含 content 的子路径
    if [[ -z "$md_file" || ! -f "$md_file" ]]; then
        md_file="$(find "$tmp_dir" -name "*.md" -path "*content*" -type f 2>/dev/null | head -1)"
    fi

    if [[ -z "$md_file" || ! -f "$md_file" ]]; then
        log_error "MinerU 未生成 .md 文件: ${source_file}"
        log_error "tmp_dir 内容: $(ls -R "$tmp_dir" 2>/dev/null | head -20)"
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

# Marker 转换单个文件（通过 conda run 在 rag-mineru 环境中执行）
convert_with_marker() {
    local source_file="$1"
    local target_md="$2"

    # 安全化路径（处理文件名含括号等特殊字符的情况）
    local safe_path
    safe_path="$(safe_source_path "$source_file")"
    local need_cleanup_link=false
    [[ "$safe_path" != "$source_file" ]] && need_cleanup_link=true

    local tmp_dir
    tmp_dir="$(mktemp -d)"

    local basename_noext
    basename_noext="$(basename "$source_file" | sed 's/\.[^.]*$//')"

    # 调用 Marker（通过 conda run 在 rag-mineru 环境中执行）
    # 优先使用 marker_single（使用安全路径）
    if ! $CONDA_CMD run -n "$CONDA_ENV_NAME" \
        env TORCH_DEVICE="${DEVICE}" bash -c "
            if command -v marker_single &>/dev/null; then
                marker_single '$safe_path' --output_dir '$tmp_dir' --output_format markdown
            elif command -v marker &>/dev/null; then
                marker '$safe_path' --output_dir '$tmp_dir' --output_format markdown
            else
                exit 1
            fi
        " >> "${LOG_FILE}" 2>&1; then
        [[ "$need_cleanup_link" == true ]] && rm -f "$safe_path"
        rm -rf "$tmp_dir"
        return 1
    fi

    # 清理符号链接
    [[ "$need_cleanup_link" == true ]] && rm -f "$safe_path"

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

# Docling 转换单个文件（通过 conda run 在 rag-mineru 环境中执行）
convert_with_docling() {
    local source_file="$1"
    local target_md="$2"

    # 安全化路径（处理文件名含括号等特殊字符的情况）
    local safe_path
    safe_path="$(safe_source_path "$source_file")"
    local need_cleanup_link=false
    [[ "$safe_path" != "$source_file" ]] && need_cleanup_link=true

    local tmp_dir
    tmp_dir="$(mktemp -d)"

    local basename_noext
    basename_noext="$(basename "$source_file" | sed 's/\.[^.]*$//')"

    # 通过 conda run 在 rag-mineru 环境中调用 Docling（使用安全路径）
    if ! $CONDA_CMD run -n "$CONDA_ENV_NAME" docling "$safe_path" --to md --output "$tmp_dir" >> "${LOG_FILE}" 2>&1; then
        [[ "$need_cleanup_link" == true ]] && rm -f "$safe_path"
        rm -rf "$tmp_dir"
        return 1
    fi

    # 清理符号链接
    [[ "$need_cleanup_link" == true ]] && rm -f "$safe_path"

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

# DOC 文件专用转换函数（MinerU 不支持 .doc 格式）
convert_doc_file() {
    local source_file="$1"
    local target_file="$2"
    local basename_file
    basename_file="$(basename "$source_file")"

    log_info "DOC 文件使用专用转换: ${basename_file}"
    mkdir -p "$(dirname "$target_file")"

    # 方案1: 使用 libreoffice 转换 doc → docx → 再用引擎处理
    if command -v libreoffice &>/dev/null; then
        log_debug "尝试 libreoffice 转换 DOC → DOCX: ${basename_file}"
        local tmp_dir
        tmp_dir=$(mktemp -d)
        if libreoffice --headless --convert-to docx --outdir "$tmp_dir" "$source_file" >> "${LOG_FILE}" 2>&1; then
            local docx_file="$tmp_dir/$(basename "${source_file%.doc}.docx")"
            # 兼容大写扩展名
            if [[ ! -f "$docx_file" ]]; then
                docx_file="$tmp_dir/$(basename "${source_file%.DOC}.docx")"
            fi
            if [[ -f "$docx_file" ]]; then
                # 安全化 docx 路径（libreoffice 转出的文件名继承原文件名，可能含括号）
                local safe_docx
                safe_docx="$(safe_source_path "$docx_file")"
                local need_cleanup_docx=false
                [[ "$safe_docx" != "$docx_file" ]] && need_cleanup_docx=true

                # 对 docx 调用正常引擎转换
                if convert_document_once "$safe_docx" "$target_file"; then
                    [[ "$need_cleanup_docx" == true ]] && rm -f "$safe_docx"
                    if validate_output_file "$target_file"; then
                        log_info "DOC → DOCX → 引擎转换成功: ${basename_file}"
                        rm -rf "$tmp_dir"
                        return 0
                    fi
                fi
                [[ "$need_cleanup_docx" == true ]] && rm -f "$safe_docx"
            fi
        fi
        rm -rf "$tmp_dir"
    fi

    # 方案2: 使用 textutil (macOS)
    if command -v textutil &>/dev/null; then
        log_debug "尝试 textutil 转换: ${basename_file}"
        local tmp_txt
        tmp_txt=$(mktemp)
        if textutil -convert txt -output "$tmp_txt" "$source_file" 2>/dev/null; then
            if [[ -s "$tmp_txt" ]]; then
                {
                    echo "# $(basename "${source_file%.*}")"
                    echo ""
                    cat "$tmp_txt"
                } > "$target_file"
                rm -f "$tmp_txt"
                log_info "DOC textutil 转换成功: ${basename_file}"
                return 0
            fi
        fi
        rm -f "$tmp_txt"
    fi

    # 方案3: 使用 catdoc 提取纯文本
    if command -v catdoc &>/dev/null; then
        log_debug "尝试 catdoc 提取: ${basename_file}"
        local content
        content=$(catdoc "$source_file" 2>/dev/null)
        if [[ -n "$content" && ${#content} -gt 10 ]]; then
            {
                echo "# $(basename "${source_file%.*}")"
                echo ""
                echo "$content"
            } > "$target_file"
            log_info "DOC catdoc 提取成功: ${basename_file}"
            return 0
        fi
    fi

    # 方案4: 使用 antiword
    if command -v antiword &>/dev/null; then
        log_debug "尝试 antiword 提取: ${basename_file}"
        local content
        content=$(antiword "$source_file" 2>/dev/null)
        if [[ -n "$content" && ${#content} -gt 10 ]]; then
            {
                echo "# $(basename "${source_file%.*}")"
                echo ""
                echo "$content"
            } > "$target_file"
            log_info "DOC antiword 提取成功: ${basename_file}"
            return 0
        fi
    fi

    # 所有方案都失败
    log_warn "DOC 文件无可用转换工具，跳过: ${basename_file}"
    log_warn "提示: 安装 libreoffice 或 catdoc 或 antiword 以支持 .doc 格式"
    return 1
}

# PPTX/PPT 文件专用转换函数（MinerU 不支持 .pptx/.ppt 格式）
convert_pptx_file() {
    local source_file="$1"
    local target_md="$2"
    local basename_file
    basename_file="$(basename "$source_file")"

    log_info "PPTX/PPT 文件使用专用转换: ${basename_file}"
    mkdir -p "$(dirname "$target_md")"

    # 检测扩展名：如果是旧版 .ppt 格式，需要先转为 .pptx
    local ext="${source_file##*.}"
    local ext_lower
    ext_lower="$(echo "$ext" | tr '[:upper:]' '[:lower:]')"

    local actual_pptx="$source_file"
    local ppt_tmp_dir=""
    if [[ "$ext_lower" == "ppt" ]]; then
        # 旧版 PPT 格式，python-pptx 不支持，需要先用 libreoffice 转换
        if command -v libreoffice &>/dev/null; then
            ppt_tmp_dir=$(mktemp -d)
            log_debug "旧版 PPT 格式，使用 libreoffice 转换为 PPTX: ${basename_file}"
            if libreoffice --headless --convert-to pptx --outdir "$ppt_tmp_dir" "$source_file" >> "${LOG_FILE}" 2>&1; then
                local converted_pptx
                converted_pptx=$(find "$ppt_tmp_dir" -name "*.pptx" -type f | head -1)
                if [[ -n "$converted_pptx" && -f "$converted_pptx" ]]; then
                    actual_pptx="$converted_pptx"
                    log_debug "PPT → PPTX 转换成功: $converted_pptx"
                else
                    log_warn "libreoffice 转换未产生 .pptx 文件，尝试直接使用 python-pptx"
                fi
            else
                log_warn "libreoffice 转换失败，尝试直接使用 python-pptx"
            fi
        else
            log_debug "libreoffice 不可用，尝试直接用 python-pptx 打开 .ppt（可能失败）"
        fi
    fi

    # 方案1: 使用 python-pptx 提取文本（主方案）
    log_debug "使用 python-pptx 提取: ${basename_file}"

    local py_script='
import sys
from pptx import Presentation

try:
    prs = Presentation(sys.argv[1])
    output_lines = []
    for slide_num, slide in enumerate(prs.slides, 1):
        output_lines.append(f"## 第 {slide_num} 页")
        output_lines.append("")
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    text = paragraph.text.strip()
                    if text:
                        output_lines.append(text)
            if hasattr(shape, "table"):
                table = shape.table
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells)
                    if row_text.strip(" |\t"):
                        output_lines.append(f"| {row_text} |")
        output_lines.append("")
    result = "\n".join(output_lines)
    if not result.strip():
        sys.exit(1)
    print(result)
except Exception as e:
    print(f"python-pptx error: {e}", file=sys.stderr)
    sys.exit(1)
'

    local content
    # 尝试在 conda 环境中执行（用 actual_pptx 而不是原始 source_file）
    content=$($CONDA_CMD run -n "$CONDA_ENV_NAME" python3 -c "$py_script" "$actual_pptx" 2>/dev/null)
    if [[ -z "$content" || ${#content} -le 10 ]]; then
        # conda 环境失败，尝试系统 Python
        content=$(python3 -c "$py_script" "$actual_pptx" 2>/dev/null)
    fi

    if [[ -n "$content" && ${#content} -gt 10 ]]; then
        {
            echo "# $(basename "${source_file%.*}")"
            echo ""
            echo "$content"
        } > "$target_md"
        if validate_output_file "$target_md"; then
            log_info "PPTX python-pptx 提取成功: ${basename_file}"
            [[ -n "$ppt_tmp_dir" ]] && rm -rf "$ppt_tmp_dir"
            return 0
        fi
    fi

    # 方案2: python-pptx 失败，尝试 libreoffice 转为 PDF 再用引擎处理
    if command -v libreoffice &>/dev/null; then
        log_debug "python-pptx 失败，尝试 libreoffice → PDF → 引擎: ${basename_file}"
        local tmp_dir
        tmp_dir=$(mktemp -d)
        if libreoffice --headless --convert-to pdf --outdir "$tmp_dir" "$source_file" >> "${LOG_FILE}" 2>&1; then
            local pdf_file="$tmp_dir/$(basename "${source_file%.*}.pdf")"
            if [[ -f "$pdf_file" ]]; then
                if convert_with_mineru "$pdf_file" "$target_md" 2>/dev/null || \
                   convert_with_marker "$pdf_file" "$target_md" 2>/dev/null || \
                   convert_with_docling "$pdf_file" "$target_md" 2>/dev/null; then
                    if validate_output_file "$target_md"; then
                        log_info "PPTX libreoffice→PDF→引擎转换成功: ${basename_file}"
                        rm -rf "$tmp_dir"
                        return 0
                    fi
                fi
            fi
        fi
        rm -rf "$tmp_dir"
    fi

    log_warn "PPTX/PPT 文件转换失败，无法提取内容: ${basename_file}"
    rm -f "$target_md"
    [[ -n "$ppt_tmp_dir" ]] && rm -rf "$ppt_tmp_dir"
    return 1
}

# PPTX 兜底方案（保留向后兼容）：使用 python-pptx 提取文本内容
convert_pptx_fallback() {
    local source_file="$1"
    local target_md="$2"
    convert_pptx_file "$source_file" "$target_md"
    return $?
}

# 统一文档转换入口（带超时控制）
convert_document_once() {
    local source_file="$1"
    local target_md="$2"

    if [[ "$ENGINE" == "none" ]]; then
        log_error "无可用的文档转换引擎，跳过: $(basename "$source_file")"
        return 1
    fi

    # 导出环境变量，让 timeout 子进程自动继承
    export CONDA_CMD CONDA_ENV_NAME BACKEND DEVICE LOG_FILE
    export RED GREEN YELLOW BLUE CYAN NC
    export MIN_OUTPUT_SIZE

    # 使用 timeout 命令执行转换
    local timeout_cmd="timeout"
    # macOS 使用 gtimeout（需安装 coreutils）
    if ! command -v timeout &>/dev/null; then
        if command -v gtimeout &>/dev/null; then
            timeout_cmd="gtimeout"
        else
            # 无 timeout 命令可用，直接执行（不限时）
            timeout_cmd=""
        fi
    fi

    local convert_result=1
    if [[ -n "$timeout_cmd" ]]; then
        # 带超时执行：通过临时脚本避免 bash -c 引号/展开问题
        local tmp_script
        tmp_script=$(mktemp /tmp/convert_XXXXXX.sh)

        local convert_func=""
        case "$ENGINE" in
            mineru)  convert_func="convert_with_mineru" ;;
            marker)  convert_func="convert_with_marker" ;;
            docling) convert_func="convert_with_docling" ;;
            *)
                log_error "未知引擎: ${ENGINE}"
                return 1
                ;;
        esac

        # 写入临时脚本：函数定义通过 declare -f 追加（避免引号冲突）
        {
            echo '#!/bin/bash'
            echo '# 自动生成的转换临时脚本'
            echo '# 环境变量已通过 export 从父进程继承'
            echo ''
            # 导出辅助函数
            declare -f log_info log_warn log_error log_debug
            declare -f get_file_size_bytes validate_output_file
            declare -f safe_source_path
            # 导出对应引擎的转换函数
            declare -f "$convert_func"
            echo ''
            echo "$convert_func \"\$1\" \"\$2\""
        } > "$tmp_script"
        chmod +x "$tmp_script"

        $timeout_cmd "${CONVERT_TIMEOUT}" bash "$tmp_script" "$source_file" "$target_md"
        convert_result=$?
        rm -f "$tmp_script"

        # timeout 返回 124 表示超时
        if [[ $convert_result -eq 124 ]]; then
            log_error "转换超时 (${CONVERT_TIMEOUT}s): $(basename "$source_file")"
            return 1
        fi
    else
        # 无超时命令，直接执行
        case "$ENGINE" in
            mineru)  convert_with_mineru "$source_file" "$target_md"; convert_result=$? ;;
            marker)  convert_with_marker "$source_file" "$target_md"; convert_result=$? ;;
            docling) convert_with_docling "$source_file" "$target_md"; convert_result=$? ;;
            *)       log_error "未知引擎: ${ENGINE}"; return 1 ;;
        esac
    fi

    return $convert_result
}

# 带重试的文档转换（含文件类型路由）
convert_document_with_retry() {
    local source_file="$1"
    local target_md="$2"

    # === 根据文件扩展名选择转换策略 ===
    local ext="${source_file##*.}"
    local ext_lower
    ext_lower="$(echo "$ext" | tr '[:upper:]' '[:lower:]')"

    case "$ext_lower" in
        doc)
            # DOC: MinerU/Marker/Docling 均不支持，使用专用工具转换
            convert_doc_file "$source_file" "$target_md"
            return $?
            ;;
        pptx|ppt)
            # PPTX/PPT: MinerU 不支持，使用 python-pptx 直接提取
            convert_pptx_file "$source_file" "$target_md"
            return $?
            ;;
        pdf|docx)
            # PDF/DOCX: 走正常引擎流程（MinerU/Marker/Docling）
            ;;
        *)
            # 其他格式：尝试走引擎流程
            ;;
    esac

    # === PDF/DOCX 走通用引擎流程（带重试） ===
    local attempt=0
    local max_retries="${CONVERT_MAX_RETRIES}"

    while [[ $attempt -le $max_retries ]]; do
        if [[ $attempt -gt 0 ]]; then
            log_warn "重试第 ${attempt}/${max_retries} 次: $(basename "$source_file")"
            sleep 2  # 重试前短暂等待
        fi

        if convert_document_once "$source_file" "$target_md"; then
            return 0
        fi

        attempt=$(( attempt + 1 ))
    done

    return 1
}

# ========== 代码文件转换 ==========
convert_code_file() {
    local source_file="$1"
    local target_md="$2"
    local relative_path="$3"  # 相对于 SOURCE_DIR 的路径

    local basename_file
    basename_file="$(basename "$source_file")"
    local lang
    lang="$(get_code_language "$source_file")"

    # 尝试读取文件内容，处理编码问题
    local content=""
    local encoding_used="utf-8"

    # 首先尝试 UTF-8
    if content="$(cat "$source_file" 2>/dev/null)" && [[ -n "$content" ]]; then
        encoding_used="utf-8"
    else
        # UTF-8 失败，尝试使用 iconv 转换 GBK
        if command -v iconv &>/dev/null; then
            if content="$(iconv -f GBK -t UTF-8 "$source_file" 2>/dev/null)"; then
                encoding_used="gbk"
                log_debug "文件使用 GBK 编码: ${basename_file}"
            elif content="$(iconv -f LATIN1 -t UTF-8 "$source_file" 2>/dev/null)"; then
                encoding_used="latin-1"
                log_debug "文件使用 Latin-1 编码: ${basename_file}"
            fi
        fi

        # 如果 iconv 不可用或所有编码都失败，使用 cat 强制读取
        if [[ -z "$content" ]]; then
            content="$(cat "$source_file" 2>/dev/null || true)"
            if [[ -z "$content" ]]; then
                # 完全无法读取，标记为二进制
                content="[二进制文件，无法解析文本内容]"
                encoding_used="binary"
                log_warn "文件无法以文本方式读取（可能为二进制）: ${basename_file}"
            fi
        fi
    fi

    # 确保目标目录存在
    mkdir -p "$(dirname "$target_md")"

    # 生成 Markdown 内容
    {
        echo "# 文件名: ${basename_file}"
        echo "# 路径: ${relative_path}"
        if [[ "$encoding_used" != "utf-8" ]]; then
            echo "# 原始编码: ${encoding_used}"
        fi
        echo ""
        echo "\`\`\`${lang}"
        echo "$content"
        echo "\`\`\`"
    } > "$target_md"

    return 0
}

# ========== Markdown 文件复制 ==========
copy_markdown_file() {
    local source_file="$1"
    local target_file="$2"

    # 如果源文件和目标文件是同一个文件（OUTPUT_DIR == SOURCE_DIR），无需操作
    if [[ "$(realpath "$source_file" 2>/dev/null || echo "$source_file")" == "$(realpath "$target_file" 2>/dev/null || echo "$target_file")" ]]; then
        log_debug "MD 文件已在目标位置，无需复制: $(basename "$source_file")"
        return 0
    fi

    # 确保目标目录存在
    mkdir -p "$(dirname "$target_file")"

    # 复制文件
    cp -f "$source_file" "$target_file"
    return 0
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
        if [[ "$(uname)" == "Darwin" ]]; then
            source_mtime="$(stat -f '%m' "$source_file" 2>/dev/null)"
        else
            source_mtime="$(stat -c '%Y' "$source_file" 2>/dev/null)"
        fi
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
    if [[ "$(uname)" == "Darwin" ]]; then
        source_mtime="$(stat -f '%m' "$source_file" 2>/dev/null)"
    else
        source_mtime="$(stat -c '%Y' "$source_file" 2>/dev/null)"
    fi

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
echo -e "${CYAN}║          文档转 Markdown 工具 (MinerU/Marker/Docling)      ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

log_info "========== 开始文件格式转换 =========="
log_info "源目录: ${SOURCE_DIR}"
log_info "输出目录: ${OUTPUT_DIR}"
log_info "转换引擎: ${ENGINE}"
[[ "$ENGINE" == "mineru" ]] && log_info "MinerU 后端: ${BACKEND}"
log_info "运行设备: ${DEVICE}"
log_info "模式: $([ "$FULL_MODE" == "true" ] && echo "全量转换" || echo "增量转换")"
log_info "超时: ${CONVERT_TIMEOUT}s | 重试: ${CONVERT_MAX_RETRIES}次 | 大小上限: ${MAX_FILE_SIZE_MB}MB"
[[ "$DRY_RUN" == "true" ]] && log_info "DRY-RUN 模式：仅预览，不实际转换"

echo ""
echo -e "  引擎:   ${GREEN}${ENGINE}${NC}"
[[ "$ENGINE" == "mineru" ]] && echo -e "  后端:   ${GREEN}${BACKEND}${NC}"
echo -e "  设备:   ${GREEN}${DEVICE}${NC}"
echo -e "  模式:   ${GREEN}$([ "$FULL_MODE" == "true" ] && echo "全量" || echo "增量")${NC}"
echo -e "  超时:   ${GREEN}${CONVERT_TIMEOUT}s${NC}  重试: ${GREEN}${CONVERT_MAX_RETRIES}${NC}次"
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

# ========== 收集需要处理的文件 ==========
declare -a files_to_convert=()
declare -a targets=()
declare -a file_types=()  # 记录每个文件的类型: document / code / markdown

# --- 收集文档文件 ---
while IFS= read -r -d '' file; do
    # 跳过临时文件/隐藏文件/空文件
    if should_skip_file "$file"; then
        continue
    fi

    local_relative="${file#${SOURCE_DIR}/}"
    target_relative="${local_relative%.*}.md"
    target_file="${OUTPUT_DIR}/${target_relative}"

    if needs_conversion "$file" "$target_file"; then
        files_to_convert+=("$file")
        targets+=("$target_file")
        file_types+=("document")
    fi
done < <(find "${SOURCE_DIR}" -type f \( \
    -iname "*.pdf" -o \
    -iname "*.docx" -o \
    -iname "*.doc" -o \
    -iname "*.pptx" -o \
    -iname "*.ppt" \
\) -print0 2>/dev/null)

# --- 收集代码文件 ---
while IFS= read -r -d '' file; do
    # 跳过隐藏目录下的文件（如 .git, .venv 等）
    if [[ "$file" == *"/.*" ]]; then
        continue
    fi

    # 跳过临时文件/隐藏文件/空文件
    if should_skip_file "$file"; then
        continue
    fi

    local_relative="${file#${SOURCE_DIR}/}"
    # 代码文件的目标：原文件名 + .md 后缀（如 main.py -> main.py.md）
    target_file="${OUTPUT_DIR}/${local_relative}.md"

    if needs_conversion "$file" "$target_file"; then
        files_to_convert+=("$file")
        targets+=("$target_file")
        file_types+=("code")
    fi
done < <(find "${SOURCE_DIR}" -type f \( \
    -name "*.py" -o -name "*.pyx" -o -name "*.pyi" -o \
    -name "*.m" -o \
    -name "*.c" -o -name "*.h" -o -name "*.cpp" -o -name "*.hpp" -o -name "*.cc" -o -name "*.cxx" -o \
    -name "*.java" -o \
    -name "*.js" -o -name "*.jsx" -o -name "*.ts" -o -name "*.tsx" -o \
    -name "*.go" -o \
    -name "*.rs" -o \
    -name "*.rb" -o \
    -name "*.sh" -o -name "*.bash" -o -name "*.zsh" -o \
    -name "*.pl" -o -name "*.pm" -o \
    -name "*.r" -o -name "*.R" -o \
    -name "*.scala" -o \
    -name "*.kt" -o -name "*.kts" -o \
    -name "*.swift" -o \
    -name "*.lua" -o \
    -name "*.sql" -o \
    -name "*.html" -o -name "*.htm" -o -name "*.css" -o -name "*.scss" -o -name "*.less" -o \
    -name "*.json" -o -name "*.yaml" -o -name "*.yml" -o -name "*.xml" -o -name "*.toml" -o \
    -name "*.ini" -o -name "*.cfg" -o -name "*.conf" -o \
    -name "*.tex" -o \
    -name "*.php" -o \
    -name "*.cs" -o \
    -name "*.dart" -o \
    -name "*.hs" -o \
    -name "*.ex" -o -name "*.exs" -o \
    -name "*.proto" -o \
    -name "*.v" -o -name "*.vhd" -o -name "*.vhdl" -o \
    -name "Makefile" -o -name "makefile" -o -name "GNUmakefile" -o \
    -name "Dockerfile" \
\) -print0 2>/dev/null)

# --- 收集 Markdown 文件 ---
while IFS= read -r -d '' file; do
    # 跳过隐藏目录下的文件
    if [[ "$file" == *"/.*" ]]; then
        continue
    fi

    # 跳过临时文件/隐藏文件/空文件
    if should_skip_file "$file"; then
        continue
    fi
    # 跳过已经是转换产物的 .md 文件（避免循环处理）
    # 判断依据：如果同目录下存在同名的源文件（如 xx.pdf 对应 xx.md），则跳过
    basename_noext="${file%.md}"
    is_output=false
    for ext in pdf docx doc pptx ppt; do
        if [[ -f "${basename_noext}.${ext}" ]]; then
            is_output=true
            break
        fi
    done
    # 也跳过 .py.md 等代码转换产物
    if [[ "$file" =~ \.(py|c|cpp|java|go|rs|rb|sh|js|ts)\.md$ ]]; then
        is_output=true
    fi
    if [[ "$is_output" == "true" ]]; then
        continue
    fi

    local_relative="${file#${SOURCE_DIR}/}"
    target_file="${OUTPUT_DIR}/${local_relative}"

    if needs_conversion "$file" "$target_file"; then
        files_to_convert+=("$file")
        targets+=("$target_file")
        file_types+=("markdown")
    fi
done < <(find "${SOURCE_DIR}" -type f -iname "*.md" -print0 2>/dev/null)

# ========== 统计信息 ==========
total_found=${#files_to_convert[@]}

if [[ ${total_found} -eq 0 ]]; then
    log_info "没有需要转换的文件（所有文件已是最新）"
    log_info "========== 转换结束 =========="
    exit 0
fi

# 统计各类型文件数
doc_count=0
code_count=0
md_count=0
for ft in "${file_types[@]}"; do
    case "$ft" in
        document) doc_count=$(( doc_count + 1 )) ;;
        code) code_count=$(( code_count + 1 )) ;;
        markdown) md_count=$(( md_count + 1 )) ;;
    esac
done

log_info "发现 ${total_found} 个文件需要处理（文档: ${doc_count}, 代码: ${code_count}, MD: ${md_count}）"

# ========== DRY-RUN 模式 ==========
if [[ "$DRY_RUN" == "true" ]]; then
    echo ""
    echo -e "${BLUE}将要处理的文件（共 ${total_found} 个）：${NC}"
    echo ""
    echo -e "  ${CYAN}--- 文档文件 (${doc_count}) ---${NC}"
    for i in "${!files_to_convert[@]}"; do
        [[ "${file_types[$i]}" != "document" ]] && continue
        local_relative="${files_to_convert[$i]#${SOURCE_DIR}/}"
        echo -e "  ${GREEN}→${NC} ${local_relative}"
    done
    echo ""
    echo -e "  ${CYAN}--- 代码文件 (${code_count}) ---${NC}"
    for i in "${!files_to_convert[@]}"; do
        [[ "${file_types[$i]}" != "code" ]] && continue
        local_relative="${files_to_convert[$i]#${SOURCE_DIR}/}"
        echo -e "  ${GREEN}→${NC} ${local_relative}"
    done
    echo ""
    echo -e "  ${CYAN}--- Markdown 文件 (${md_count}) ---${NC}"
    for i in "${!files_to_convert[@]}"; do
        [[ "${file_types[$i]}" != "markdown" ]] && continue
        local_relative="${files_to_convert[$i]#${SOURCE_DIR}/}"
        echo -e "  ${GREEN}→${NC} ${local_relative}"
    done
    echo ""
    echo -e "${CYAN}引擎: ${ENGINE} | 设备: ${DEVICE}${NC}"
    echo ""
    echo "如需执行转换，请去掉 --dry-run 参数"
    exit 0
fi

# ========== 执行转换 ==========
doc_success=0
doc_fail=0
code_success=0
code_fail=0
md_success=0
md_fail=0
declare -a failed_files=()
declare -a failed_reasons=()

# 记录总开始时间
total_start_time="$(date +%s)"

log_info "开始转换，使用引擎: ${ENGINE}"
echo ""

for i in "${!files_to_convert[@]}"; do
    source_file="${files_to_convert[$i]}"
    target_file="${targets[$i]}"
    file_type="${file_types[$i]}"
    local_relative="${source_file#${SOURCE_DIR}/}"

    # 确保目标目录存在
    target_dir="$(dirname "$target_file")"
    mkdir -p "$target_dir"

    # 获取文件大小
    file_size_mb="$(get_file_size_mb "$source_file")"

    # 打印处理信息
    echo -e "  [$(( i + 1 ))/${total_found}] ${BLUE}处理:${NC} ${local_relative}"
    log_info "正在处理: ${local_relative} (类型: ${file_type}, 大小: ${file_size_mb} MB)"

    # 记录单文件开始时间
    file_start_time="$(date +%s)"

    # === 文件大小检查（仅对文档和代码文件） ===
    if [[ "$file_type" != "markdown" ]] && ! check_file_size "$source_file"; then
        log_warn "文件过大 (>${MAX_FILE_SIZE_MB}MB)，跳过: ${local_relative} (${file_size_mb} MB)"
        echo -e "           ${YELLOW}⚠ 跳过: 文件过大 (${file_size_mb} MB > ${MAX_FILE_SIZE_MB} MB)${NC}"
        case "$file_type" in
            document) doc_fail=$(( doc_fail + 1 )) ;;
            code) code_fail=$(( code_fail + 1 )) ;;
        esac
        failed_files+=("$local_relative")
        failed_reasons+=("文件过大 (${file_size_mb} MB)")
        file_end_time="$(date +%s)"
        log_info "完成: ${local_relative} (耗时: $(( file_end_time - file_start_time ))s)"
        continue
    fi

    # === 根据文件类型分别处理 ===
    convert_success=false
    fail_reason=""
    target_tmp="${target_file}.tmp.$$.$i"
    rm -f "$target_tmp"

    case "$file_type" in
        document)
            # 文档文件：使用引擎转换（带重试和兜底）
            if ( convert_document_with_retry "$source_file" "$target_tmp" ); then
                if validate_output_file "$target_tmp"; then
                    convert_success=true
                else
                    fail_reason="转换产生空文件或过小文件"
                    rm -f "$target_tmp"
                fi
            else
                fail_reason="转换引擎失败（已重试 ${CONVERT_MAX_RETRIES} 次）"
            fi
            ;;
        code)
            # 代码文件：包裹在 Markdown 代码块中
            if ( convert_code_file "$source_file" "$target_tmp" "$local_relative" ); then
                if validate_output_file "$target_tmp"; then
                    convert_success=true
                else
                    fail_reason="代码文件转换产生空输出"
                    rm -f "$target_tmp"
                fi
            else
                fail_reason="代码文件读取/转换失败"
            fi
            ;;
        markdown)
            # Markdown 文件：直接复制
            if ( copy_markdown_file "$source_file" "$target_tmp" ); then
                if validate_output_file "$target_tmp"; then
                    convert_success=true
                else
                    fail_reason="Markdown 文件复制产生无效输出"
                fi
            else
                fail_reason="Markdown 文件复制失败"
            fi
            ;;
    esac

    # 记录单文件结束时间
    file_end_time="$(date +%s)"
    file_elapsed=$(( file_end_time - file_start_time ))

    # === 更新统计计数 ===
    if [[ "$convert_success" == "true" ]]; then
        # 临时文件与目标文件位于同一目录，替换操作具有原子性。
        mv -f "$target_tmp" "$target_file"
        case "$file_type" in
            document) doc_success=$(( doc_success + 1 )) ;;
            code) code_success=$(( code_success + 1 )) ;;
            markdown) md_success=$(( md_success + 1 )) ;;
        esac
        # 更新状态记录
        update_state "$source_file"
        echo -e "           ${GREEN}✓ 成功${NC} → $(basename "$target_file") (${file_elapsed}s)"
        log_info "完成: ${local_relative} (耗时: ${file_elapsed}s)"
    else
        rm -f "$target_tmp"
        case "$file_type" in
            document) doc_fail=$(( doc_fail + 1 )) ;;
            code) code_fail=$(( code_fail + 1 )) ;;
            markdown) md_fail=$(( md_fail + 1 )) ;;
        esac
        failed_files+=("$local_relative")
        failed_reasons+=("${fail_reason}")
        echo -e "           ${RED}✗ ${fail_reason}${NC}"
        log_error "失败: ${local_relative} - ${fail_reason} (耗时: ${file_elapsed}s)"
    fi
done

# 总耗时
total_end_time="$(date +%s)"
total_elapsed=$(( total_end_time - total_start_time ))

# ========== 输出汇总 ==========
total_success=$(( doc_success + code_success + md_success ))
total_fail=$(( doc_fail + code_fail + md_fail ))

echo ""
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}                     转换完成汇总                         ${NC}"
echo -e "${CYAN}══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  引擎:     ${ENGINE}"
echo -e "  设备:     ${DEVICE}"
echo ""
echo -e "  === 转换统计 ==="
echo -e "  文档转换: ${GREEN}${doc_success} 成功${NC} / ${RED}${doc_fail} 失败${NC}"
echo -e "  代码文件: ${GREEN}${code_success} 成功${NC} / ${RED}${code_fail} 失败${NC}"
echo -e "  MD 复制:  ${GREEN}${md_success} 成功${NC} / ${RED}${md_fail} 失败${NC}"
echo -e "  总计: ${total_found} 个文件, 耗时: ${total_elapsed}s"
echo ""

if [[ ${total_fail} -gt 0 ]]; then
    echo -e "${RED}失败文件列表：${NC}"
    for idx in "${!failed_files[@]}"; do
        echo -e "  ${RED}✗${NC} ${failed_files[$idx]}"
        echo -e "    ${YELLOW}原因: ${failed_reasons[$idx]}${NC}"
    done
    echo ""
fi

log_info "转换汇总: 文档=${doc_success}成功/${doc_fail}失败, 代码=${code_success}成功/${code_fail}失败, MD=${md_success}成功/${md_fail}失败, 总计=${total_found}, 耗时=${total_elapsed}s"
log_info "========== 转换结束 =========="

# ========== 后续操作提示 ==========
if [[ ${total_success} -gt 0 ]]; then
    echo -e "${GREEN}转换完成！可执行以下命令进行知识入库：${NC}"
    echo "  bash auto_ingest.sh        # 增量入库（推荐）"
    echo "  bash auto_ingest.sh --full # 全量入库"
    echo ""
fi

# 任一文件失败都返回非零；成功项及其状态保留，重跑只处理失败项。
if [[ ${total_fail} -gt 0 ]]; then
    exit 1
fi

exit 0
