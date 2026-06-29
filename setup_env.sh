#!/bin/bash
# ==============================================================================
# RAG 知识库系统 - 环境准备脚本
# 封装项目初始化所需的全部环境配置步骤
# 用法:
#   bash setup_env.sh                  # 执行全部环境准备步骤
#   bash setup_env.sh --skip-deps      # 跳过 pip 依赖安装
#   bash setup_env.sh --skip-models    # 跳过模型检查/下载
#   bash setup_env.sh --skip-converter # 跳过文档转换工具安装
#   bash setup_env.sh --help           # 显示帮助信息
# ==============================================================================

set -euo pipefail

# ========== 基础配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/.env"
ENV_EXAMPLE_FILE="${SCRIPT_DIR}/.env.example"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/setup_env.log"
LOCK_FILE="/tmp/setup_env.lock"

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
    echo "[INFO] [${timestamp}] $*" >> "${LOG_FILE}" 2>/dev/null || true
}

log_warn() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${YELLOW}[WARN]${NC} [${timestamp}] $*"
    echo "[WARN] [${timestamp}] $*" >> "${LOG_FILE}" 2>/dev/null || true
}

log_error() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${RED}[ERROR]${NC} [${timestamp}] $*" >&2
    echo "[ERROR] [${timestamp}] $*" >> "${LOG_FILE}" 2>/dev/null || true
}

log_debug() {
    local timestamp
    timestamp="$(date '+%Y-%m-%d %H:%M:%S')"
    echo -e "${BLUE}[DEBUG]${NC} [${timestamp}] $*"
    echo "[DEBUG] [${timestamp}] $*" >> "${LOG_FILE}" 2>/dev/null || true
}

# ========== 结果追踪 ==========
# 状态: OK / FAIL / SKIP
declare -a REPORT_ITEMS=()
declare -a REPORT_STATUS=()
declare -a REPORT_DETAILS=()
declare -a FIX_SUGGESTIONS=()
HAS_FAILURE=0

report_add() {
    local item="$1"
    local status="$2"
    local detail="${3:-}"
    REPORT_ITEMS+=("$item")
    REPORT_STATUS+=("$status")
    REPORT_DETAILS+=("$detail")
    if [[ "$status" == "FAIL" ]]; then
        HAS_FAILURE=1
    fi
}

fix_add() {
    FIX_SUGGESTIONS+=("$1")
}

# ========== 帮助信息 ==========
show_help() {
    cat << 'EOF'
RAG 知识库系统 - 环境准备脚本

用法:
  bash setup_env.sh [选项]

选项:
  --skip-deps        跳过 pip 依赖安装（适用于依赖已装好的环境）
  --skip-models      跳过模型检查/下载
  --skip-converter   跳过文档转换工具安装
  --help             显示帮助信息

步骤说明:
  Step 1: 环境检测      - 检查 Python 3.8+、pip、CUDA、操作系统
  Step 2: Python 依赖   - 安装项目核心 pip 依赖
  Step 3: 文档转换工具  - 按优先级安装 MinerU > Marker > Docling
  Step 4: 模型检查      - 检查 LLM、Embedding、Reranker 模型是否存在
  Step 5: 配置文件      - 检查/初始化 .env 配置文件
  Step 6: 目录创建      - 创建 logs/、data/、data/chat_histories/
  Step 7: 用户创建引导  - 提示手动运行 create_user.py

错误处理:
  - 致命错误（如 Python 不存在）立即中止
  - 非致命错误（如某工具安装失败）记录并继续
  - 脚本结束时输出完整环境检查报告

返回码:
  0 - 全部成功
  1 - 存在失败项

示例:
  bash setup_env.sh                           # 完整环境准备
  bash setup_env.sh --skip-deps --skip-models # 仅检查配置和创建目录
EOF
}

# ========== 参数解析 ==========
SKIP_DEPS=false
SKIP_MODELS=false
SKIP_CONVERTER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-deps)
            SKIP_DEPS=true
            shift
            ;;
        --skip-models)
            SKIP_MODELS=true
            shift
            ;;
        --skip-converter)
            SKIP_CONVERTER=true
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

# ========== 初始化日志目录 ==========
mkdir -p "${LOG_DIR}" 2>/dev/null || true

# ========== 文件锁防止并发执行 ==========
exec 200>"${LOCK_FILE}"
if ! flock -n 200 2>/dev/null; then
    log_error "另一个 setup_env 实例正在运行，退出"
    exit 1
fi

# ========== 主流程开始 ==========
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         RAG 知识库系统 - 环境准备脚本                    ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

log_info "========== 开始环境准备 =========="
log_info "工作目录: ${SCRIPT_DIR}"

# ==============================================================================
# Step 1: 环境检测
# ==============================================================================
echo -e "${BLUE}━━━ Step 1: 环境检测 ━━━${NC}"
echo ""

# 1.1 检查操作系统类型
OS_TYPE="unknown"
case "$(uname -s)" in
    Linux*)   OS_TYPE="Linux" ;;
    Darwin*)  OS_TYPE="macOS" ;;
    CYGWIN*|MINGW*|MSYS*) OS_TYPE="Windows" ;;
esac
log_info "操作系统: ${OS_TYPE} ($(uname -s) $(uname -m))"
report_add "操作系统检测" "OK" "${OS_TYPE} ($(uname -m))"

# 1.2 检查 Python 3.8+
PYTHON_CMD=""
PYTHON_VERSION=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
fi

if [[ -n "$PYTHON_CMD" ]]; then
    PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | grep -oP '\d+\.\d+\.\d+' || $PYTHON_CMD --version 2>&1 | sed 's/Python //')
    # 检查版本是否 >= 3.8
    PYTHON_MAJOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.major)" 2>/dev/null || echo "0")
    PYTHON_MINOR=$($PYTHON_CMD -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "0")

    if [[ "$PYTHON_MAJOR" -ge 3 && "$PYTHON_MINOR" -ge 8 ]]; then
        log_info "Python: ${PYTHON_VERSION} (${PYTHON_CMD}) ✓"
        report_add "Python 3.8+" "OK" "${PYTHON_VERSION}"
    else
        log_error "Python 版本过低: ${PYTHON_VERSION}，需要 3.8+"
        report_add "Python 3.8+" "FAIL" "当前版本 ${PYTHON_VERSION}，需要 3.8+"
        fix_add "请升级 Python 到 3.8 或更高版本"
        echo -e "${RED}[致命错误] Python 版本不满足要求，中止执行${NC}"
        # 致命错误，跳转到报告
        HAS_FAILURE=1
        # 打印报告后退出
        echo ""
        echo -e "${CYAN}━━━ 环境检查报告 ━━━${NC}"
        echo ""
        for i in "${!REPORT_ITEMS[@]}"; do
            case "${REPORT_STATUS[$i]}" in
                OK)   echo -e "  ${GREEN}✓${NC} ${REPORT_ITEMS[$i]} — ${REPORT_DETAILS[$i]}" ;;
                FAIL) echo -e "  ${RED}✗${NC} ${REPORT_ITEMS[$i]} — ${REPORT_DETAILS[$i]}" ;;
                SKIP) echo -e "  ${YELLOW}⚠${NC} ${REPORT_ITEMS[$i]} — ${REPORT_DETAILS[$i]}" ;;
            esac
        done
        echo ""
        echo -e "${RED}修复建议：${NC}"
        for s in "${FIX_SUGGESTIONS[@]}"; do
            echo -e "  • $s"
        done
        exit 1
    fi
else
    log_error "未找到 Python，请先安装 Python 3.8+"
    report_add "Python 3.8+" "FAIL" "未找到 python3 或 python 命令"
    fix_add "请安装 Python 3.8+: https://www.python.org/downloads/"
    echo -e "${RED}[致命错误] Python 未安装，中止执行${NC}"
    exit 1
fi

# 1.3 检查 pip
PIP_CMD=""
if command -v pip3 &>/dev/null; then
    PIP_CMD="pip3"
elif command -v pip &>/dev/null; then
    PIP_CMD="pip"
elif $PYTHON_CMD -m pip --version &>/dev/null; then
    PIP_CMD="$PYTHON_CMD -m pip"
fi

if [[ -n "$PIP_CMD" ]]; then
    PIP_VERSION=$($PIP_CMD --version 2>&1 | head -1)
    log_info "pip: ${PIP_VERSION} ✓"
    report_add "pip 可用" "OK" "$(echo "$PIP_VERSION" | awk '{print $2}')"
else
    log_error "未找到 pip，请先安装 pip"
    report_add "pip 可用" "FAIL" "未找到 pip/pip3 命令"
    fix_add "安装 pip: $PYTHON_CMD -m ensurepip --upgrade"
    echo -e "${RED}[致命错误] pip 不可用，中止执行${NC}"
    exit 1
fi

# 1.4 检查 CUDA 可用性
CUDA_AVAILABLE=false
if command -v nvidia-smi &>/dev/null; then
    CUDA_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [[ -n "$CUDA_INFO" ]]; then
        CUDA_AVAILABLE=true
        log_info "CUDA: 可用 (${CUDA_INFO})"
        report_add "CUDA/GPU" "OK" "${CUDA_INFO}"
    else
        log_warn "nvidia-smi 存在但无法获取 GPU 信息"
        report_add "CUDA/GPU" "SKIP" "nvidia-smi 存在但无 GPU 信息"
    fi
else
    log_warn "nvidia-smi 不可用，将以 CPU 模式运行（性能受限）"
    report_add "CUDA/GPU" "SKIP" "未检测到 nvidia-smi，CPU 模式"
fi

echo ""

# ==============================================================================
# Step 2: Python 依赖安装
# ==============================================================================
echo -e "${BLUE}━━━ Step 2: Python 依赖安装 ━━━${NC}"
echo ""

if [[ "$SKIP_DEPS" == "true" ]]; then
    log_info "跳过 pip 依赖安装（--skip-deps）"
    report_add "Python 依赖安装" "SKIP" "用户指定跳过"
else
    CORE_DEPS=(
        "flask"
        "python-dotenv"
        "langchain"
        "langchain-openai"
        "langchain-huggingface"
        "langchain-qdrant"
        "qdrant-client"
        "sentence-transformers"
        "tiktoken"
        "vllm"
    )

    log_info "开始安装核心 Python 依赖（共 ${#CORE_DEPS[@]} 个包）..."
    DEPS_STR="${CORE_DEPS[*]}"

    # 使用子 shell 避免 set -e 中断整体流程
    if ( $PIP_CMD install $DEPS_STR 2>&1 | tee -a "${LOG_FILE}" | tail -5 ); then
        log_info "Python 依赖安装完成 ✓"
        report_add "Python 依赖安装" "OK" "已安装 ${#CORE_DEPS[@]} 个核心包"
    else
        log_error "Python 依赖安装失败（部分包可能未安装成功）"
        report_add "Python 依赖安装" "FAIL" "pip install 返回错误"
        fix_add "手动执行: $PIP_CMD install ${DEPS_STR}"
    fi
fi

echo ""

# ==============================================================================
# Step 3: 文档转换工具安装
# ==============================================================================
echo -e "${BLUE}━━━ Step 3: 文档转换工具安装 ━━━${NC}"
echo ""

if [[ "$SKIP_CONVERTER" == "true" ]]; then
    log_info "跳过文档转换工具安装（--skip-converter）"
    report_add "文档转换工具" "SKIP" "用户指定跳过"
else
    CONVERTER_INSTALLED=false
    CONVERTER_NAME=""

    # 先检查是否已有可用的转换工具
    if command -v mineru &>/dev/null; then
        log_info "MinerU 已安装，跳过转换工具安装"
        CONVERTER_INSTALLED=true
        CONVERTER_NAME="MinerU（已存在）"
    elif command -v marker_single &>/dev/null || command -v marker &>/dev/null; then
        log_info "Marker 已安装，跳过转换工具安装"
        CONVERTER_INSTALLED=true
        CONVERTER_NAME="Marker（已存在）"
    elif command -v docling &>/dev/null; then
        log_info "Docling 已安装，跳过转换工具安装"
        CONVERTER_INSTALLED=true
        CONVERTER_NAME="Docling（已存在）"
    fi

    if [[ "$CONVERTER_INSTALLED" == "false" ]]; then
        # 按优先级尝试安装：MinerU > Marker > Docling
        log_info "尝试安装 MinerU（优先级最高）..."
        if ( $PIP_CMD install uv >> "${LOG_FILE}" 2>&1 && uv pip install -U "mineru[all]" >> "${LOG_FILE}" 2>&1 ); then
            log_info "MinerU 安装成功 ✓"
            CONVERTER_INSTALLED=true
            CONVERTER_NAME="MinerU"
        else
            log_warn "MinerU 安装失败，尝试 Marker..."

            if ( $PIP_CMD install "marker-pdf[full]" >> "${LOG_FILE}" 2>&1 ); then
                log_info "Marker 安装成功 ✓"
                CONVERTER_INSTALLED=true
                CONVERTER_NAME="Marker"
            else
                log_warn "Marker 安装失败，尝试 Docling..."

                if ( $PIP_CMD install docling >> "${LOG_FILE}" 2>&1 ); then
                    log_info "Docling 安装成功 ✓"
                    CONVERTER_INSTALLED=true
                    CONVERTER_NAME="Docling"
                else
                    log_error "所有文档转换工具安装失败"
                fi
            fi
        fi
    fi

    if [[ "$CONVERTER_INSTALLED" == "true" ]]; then
        report_add "文档转换工具" "OK" "${CONVERTER_NAME}"
    else
        report_add "文档转换工具" "FAIL" "MinerU/Marker/Docling 均安装失败"
        fix_add "手动安装文档转换工具（任选一）："
        fix_add "  MinerU: pip install uv && uv pip install -U \"mineru[all]\""
        fix_add "  Marker: pip install marker-pdf[full]"
        fix_add "  Docling: pip install docling"
    fi
fi

echo ""

# ==============================================================================
# Step 4: 模型检查
# ==============================================================================
echo -e "${BLUE}━━━ Step 4: 模型检查 ━━━${NC}"
echo ""

if [[ "$SKIP_MODELS" == "true" ]]; then
    log_info "跳过模型检查（--skip-models）"
    report_add "LLM 模型" "SKIP" "用户指定跳过"
    report_add "Embedding 模型" "SKIP" "用户指定跳过"
    report_add "Reranker 模型" "SKIP" "用户指定跳过"
else
    # 4.1 检查 LLM 模型
    LLM_MODEL_PATH="${SCRIPT_DIR}/models/Qwen3-8B-Instruct"
    if [[ -d "$LLM_MODEL_PATH" ]]; then
        log_info "LLM 模型存在: ${LLM_MODEL_PATH} ✓"
        report_add "LLM 模型 (Qwen3-8B)" "OK" "${LLM_MODEL_PATH}"
    else
        log_warn "LLM 模型不存在: ${LLM_MODEL_PATH}"
        report_add "LLM 模型 (Qwen3-8B)" "FAIL" "目录不存在"
        fix_add "下载 LLM 模型: bash download_model.sh"
    fi

    # 4.2 检查 Embedding 模型
    EMBEDDING_MODEL_PATH="${SCRIPT_DIR}/models/bge-m3"
    if [[ -d "$EMBEDDING_MODEL_PATH" ]]; then
        log_info "Embedding 模型存在: ${EMBEDDING_MODEL_PATH} ✓"
        report_add "Embedding 模型 (bge-m3)" "OK" "${EMBEDDING_MODEL_PATH}"
    else
        log_warn "Embedding 模型不存在: ${EMBEDDING_MODEL_PATH}"
        report_add "Embedding 模型 (bge-m3)" "FAIL" "目录不存在"
        fix_add "下载 Embedding 模型到 ./models/bge-m3 目录"
    fi

    # 4.3 检查 Reranker 模型
    RERANKER_MODEL_PATH="${SCRIPT_DIR}/models/bge-reranker-v2-m3"
    if [[ -d "$RERANKER_MODEL_PATH" ]]; then
        log_info "Reranker 模型存在: ${RERANKER_MODEL_PATH} ✓"
        report_add "Reranker 模型 (bge-reranker-v2-m3)" "OK" "${RERANKER_MODEL_PATH}"
    else
        log_warn "Reranker 模型不存在: ${RERANKER_MODEL_PATH}"
        report_add "Reranker 模型 (bge-reranker-v2-m3)" "FAIL" "目录不存在"
        fix_add "下载 Reranker 模型到 ./models/bge-reranker-v2-m3 目录"
    fi
fi

echo ""

# ==============================================================================
# Step 5: 配置文件检查
# ==============================================================================
echo -e "${BLUE}━━━ Step 5: 配置文件检查 ━━━${NC}"
echo ""

if [[ -f "$ENV_FILE" ]]; then
    log_info ".env 文件存在: ${ENV_FILE} ✓"

    # 检查 FLASK_SECRET_KEY 是否为默认值
    SECRET_KEY=$(grep -E "^FLASK_SECRET_KEY=" "$ENV_FILE" 2>/dev/null | cut -d'=' -f2- || echo "")
    if [[ "$SECRET_KEY" == "lab403-rag-secret-key-change-me-in-production" ]]; then
        log_warn "FLASK_SECRET_KEY 仍为默认值，建议修改为随机强密钥"
        report_add ".env 配置文件" "OK" "存在，但 SECRET_KEY 需修改"
        fix_add "修改 .env 中的 FLASK_SECRET_KEY 为随机强密钥"
        fix_add "  生成方法: python3 -c \"import secrets; print(secrets.token_hex(32))\""
    else
        report_add ".env 配置文件" "OK" "存在且 SECRET_KEY 已自定义"
    fi
else
    log_warn ".env 文件不存在"
    # 尝试从 .env.example 复制
    if [[ -f "$ENV_EXAMPLE_FILE" ]]; then
        log_info "从 .env.example 复制为 .env"
        cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
        report_add ".env 配置文件" "OK" "已从 .env.example 创建"
        fix_add "请编辑 .env 文件，确认配置项（尤其是 GPU 分配和 Qdrant 地址）"
    else
        log_error ".env 和 .env.example 均不存在，需要手动创建配置文件"
        report_add ".env 配置文件" "FAIL" ".env 不存在且无 .env.example"
        fix_add "请参考 README.md 手动创建 .env 配置文件"
    fi
fi

echo ""

# ==============================================================================
# Step 6: 目录创建
# ==============================================================================
echo -e "${BLUE}━━━ Step 6: 目录创建 ━━━${NC}"
echo ""

DIRS_TO_CREATE=(
    "${SCRIPT_DIR}/logs"
    "${SCRIPT_DIR}/data"
    "${SCRIPT_DIR}/data/chat_histories"
)

DIRS_CREATED=0
for dir in "${DIRS_TO_CREATE[@]}"; do
    if [[ ! -d "$dir" ]]; then
        mkdir -p "$dir"
        log_info "已创建目录: ${dir}"
        DIRS_CREATED=$((DIRS_CREATED + 1))
    else
        log_debug "目录已存在: ${dir}"
    fi
done

if [[ $DIRS_CREATED -gt 0 ]]; then
    report_add "必要目录创建" "OK" "新建 ${DIRS_CREATED} 个目录"
else
    report_add "必要目录创建" "OK" "所有目录已存在"
fi

echo ""

# ==============================================================================
# Step 7: 用户创建引导
# ==============================================================================
echo -e "${BLUE}━━━ Step 7: 用户创建引导 ━━━${NC}"
echo ""

if [[ -f "${SCRIPT_DIR}/create_user.py" ]]; then
    log_info "用户创建脚本存在: create_user.py"
    echo -e "  ${YELLOW}提示${NC}: 如需创建登录账号，请手动运行："
    echo -e "    ${GREEN}cd ${SCRIPT_DIR} && $PYTHON_CMD create_user.py${NC}"
    echo ""
    report_add "用户创建引导" "OK" "请手动运行 create_user.py"
else
    log_warn "未找到 create_user.py"
    report_add "用户创建引导" "SKIP" "create_user.py 不存在"
fi

echo ""

# ==============================================================================
# 环境检查报告
# ==============================================================================
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                   环境检查报告                           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# 打印报告表格
printf "  %-4s %-34s %s\n" "状态" "检查项" "详情"
printf "  %-4s %-34s %s\n" "────" "──────────────────────────────────" "──────────────────────"

for i in "${!REPORT_ITEMS[@]}"; do
    local_status="${REPORT_STATUS[$i]}"
    local_item="${REPORT_ITEMS[$i]}"
    local_detail="${REPORT_DETAILS[$i]}"

    case "$local_status" in
        OK)
            printf "  ${GREEN} ✓ ${NC} %-34s %s\n" "$local_item" "$local_detail"
            ;;
        FAIL)
            printf "  ${RED} ✗ ${NC} %-34s %s\n" "$local_item" "$local_detail"
            ;;
        SKIP)
            printf "  ${YELLOW} ⚠ ${NC} %-34s %s\n" "$local_item" "$local_detail"
            ;;
    esac
done

echo ""

# 输出修复建议（如有）
if [[ ${#FIX_SUGGESTIONS[@]} -gt 0 ]]; then
    echo -e "${YELLOW}修复建议：${NC}"
    echo ""
    for suggestion in "${FIX_SUGGESTIONS[@]}"; do
        echo -e "  • ${suggestion}"
    done
    echo ""
fi

# 汇总结论
if [[ $HAS_FAILURE -eq 0 ]]; then
    echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✓ 环境准备完成！所有检查项均已通过${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  后续步骤："
    echo -e "    1. 编辑 .env 确认配置（如尚未配置）"
    echo -e "    2. 运行 ${GREEN}python create_user.py${NC} 创建账号"
    echo -e "    3. 运行 ${GREEN}bash convert_to_md.sh --full${NC} 转换文档"
    echo -e "    4. 运行 ${GREEN}bash auto_ingest.sh --full${NC} 首次入库"
    echo -e "    5. 运行 ${GREEN}bash start_rag.sh start${NC} 启动服务"
    echo ""
else
    echo -e "${RED}══════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ✗ 环境准备存在失败项，请根据上方建议修复后重试${NC}"
    echo -e "${RED}══════════════════════════════════════════════════════════${NC}"
    echo ""
fi

log_info "========== 环境准备结束 (失败项: ${HAS_FAILURE}) =========="

# 返回码：0=全部成功，1=存在失败项
exit $HAS_FAILURE
