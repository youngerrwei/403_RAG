#!/bin/bash
# ==============================================================================
# RAG 知识库系统 - 可重复执行的多环境安装脚本
# 背景：vLLM 和 MinerU 无法共存于同一 conda 环境（PyTorch/flashinfer/paddlepaddle 冲突）
# 方案：创建独立的 conda 环境
#
# 环境规划：
#   rag-vllm   — vLLM 推理服务 (Python 3.10 + PyTorch 2.5.1+cu124 + vLLM 0.8.5.post1)
#   rag-mineru — MinerU 文档转换 (Python 3.10 + PyTorch 2.5.1+cu124 + MinerU)
#   rag        — RAG 主应用 (不存在时创建，并安装核心依赖)
#
# 用法:
#   bash setup_env.sh              # 安装所有环境
#   bash setup_env.sh --vllm       # 仅安装 vLLM 环境
#   bash setup_env.sh --mineru     # 仅安装 MinerU 环境
#   bash setup_env.sh --rag        # 仅安装 RAG 主环境依赖
#   bash setup_env.sh --force      # 强制重建已存在的环境
#   bash setup_env.sh --help       # 显示帮助
# ==============================================================================

# 注意：不使用 set -e，因为 conda 激活/run 命令失败不应中断整体流程
set -uo pipefail

# ========== 基础配置 ==========
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/setup_env.log"
ENV_FILE="${SCRIPT_DIR}/.env"
ENV_EXAMPLE="${SCRIPT_DIR}/.env.example"
source "${SCRIPT_DIR}/scripts/runtime_common.sh"

# conda 环境名称
load_env_keys "$ENV_FILE" RAG_CONDA_ENV VLLM_CONDA_ENV MINERU_CONDA_ENV || true
ENV_RAG="${RAG_CONDA_ENV:-rag}"
ENV_VLLM="${VLLM_CONDA_ENV:-rag-vllm}"
ENV_MINERU="${MINERU_CONDA_ENV:-rag-mineru}"

# PyTorch 版本（统一使用 cu124）
PYTORCH_PACKAGES="torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1"
PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu124"

# ========== 颜色定义 ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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

# ========== 结果追踪 ==========
declare -a REPORT_ITEMS=()
declare -a REPORT_STATUS=()
declare -a REPORT_DETAILS=()
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

# ========== 帮助信息 ==========
show_help() {
    cat << 'EOF'
RAG 知识库系统 - 多环境安装脚本

用法:
  bash setup_env.sh [选项]

选项:
  --vllm          仅安装 vLLM 环境 (rag-vllm)
  --mineru        仅安装 MinerU 环境 (rag-mineru)
  --rag           仅创建/更新 RAG 主环境
  --skip-vllm     跳过 vLLM 环境安装
  --skip-mineru   跳过 MinerU 环境安装
  --force         强制重建已存在的 conda 环境（先删除再创建）
  --help          显示帮助信息

环境说明:
  rag-vllm   — vLLM 推理服务
               Python 3.10 + PyTorch 2.5.1+cu124 + vLLM 0.8.5.post1
  rag-mineru — MinerU 文档转换
               Python 3.10 + PyTorch 2.5.1+cu124 + MinerU (via uv)
  rag        — RAG 主应用（不存在时自动创建）
               Flask + LangChain + Qdrant + sentence-transformers 等

示例:
  bash setup_env.sh                    # 安装所有环境
  bash setup_env.sh --vllm             # 仅安装 vLLM 环境
  bash setup_env.sh --mineru --force   # 强制重建 MinerU 环境
  bash setup_env.sh --rag              # 仅安装 RAG 核心依赖
  bash setup_env.sh --skip-vllm        # 跳过 vLLM，安装其余环境
EOF
}

# ========== 参数解析 ==========
INSTALL_VLLM=true
INSTALL_MINERU=true
INSTALL_RAG=true
FORCE=false
ONLY_MODE=false  # 是否指定了 --vllm/--mineru/--rag（仅安装模式）

while [[ $# -gt 0 ]]; do
    case "$1" in
        --vllm)
            ONLY_MODE=true
            INSTALL_VLLM=true
            INSTALL_MINERU=false
            INSTALL_RAG=false
            shift
            ;;
        --mineru)
            ONLY_MODE=true
            INSTALL_VLLM=false
            INSTALL_MINERU=true
            INSTALL_RAG=false
            shift
            ;;
        --rag)
            ONLY_MODE=true
            INSTALL_VLLM=false
            INSTALL_MINERU=false
            INSTALL_RAG=true
            shift
            ;;
        --skip-vllm)
            INSTALL_VLLM=false
            shift
            ;;
        --skip-mineru)
            INSTALL_MINERU=false
            shift
            ;;
        --force)
            FORCE=true
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

# ========== 初始化 ==========
mkdir -p "${LOG_DIR}" 2>/dev/null || true

ensure_private_env() {
    if [[ ! -f "$ENV_FILE" ]]; then
        [[ -f "$ENV_EXAMPLE" ]] || { log_error "缺少配置模板: $ENV_EXAMPLE"; return 1; }
        cp "$ENV_EXAMPLE" "$ENV_FILE"
        log_info "已从 .env.example 创建本机 .env"
    fi

    local current_secret secret tmp_file openssl_bin
    current_secret="$(awk -F= '$1 == "FLASK_SECRET_KEY" {sub(/^[^=]*=/, ""); print; exit}' "$ENV_FILE")"
    if is_weak_flask_secret "$current_secret"; then
        openssl_bin="$(find_executable openssl /usr/bin/openssl /usr/local/bin/openssl || true)"
        if [[ -n "$openssl_bin" ]]; then
            secret="$($openssl_bin rand -hex 32)"
        elif [[ -r /dev/urandom ]] && command -v od >/dev/null 2>&1; then
            secret="$(od -An -N32 -tx1 /dev/urandom | tr -d ' \n')"
        else
            log_error "无法生成安全的 FLASK_SECRET_KEY（缺少 openssl 和 /dev/urandom）"
            return 1
        fi
        tmp_file="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
        awk -v secret="$secret" '
            BEGIN { replaced = 0 }
            /^FLASK_SECRET_KEY=/ { print "FLASK_SECRET_KEY=" secret; replaced = 1; next }
            { print }
            END { if (!replaced) print "FLASK_SECRET_KEY=" secret }
        ' "$ENV_FILE" > "$tmp_file"
        mv -f "$tmp_file" "$ENV_FILE"
        chmod 600 "$ENV_FILE" 2>/dev/null || true
        log_info "已生成本机随机 FLASK_SECRET_KEY（现有登录 Session 将失效）"
    fi
}

ensure_private_env || exit 1

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       RAG 知识库系统 - 多环境安装脚本                     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

log_info "========== 开始环境安装 =========="
log_info "工作目录: ${SCRIPT_DIR}"
log_info "安装计划: vLLM=${INSTALL_VLLM}, MinerU=${INSTALL_MINERU}, RAG=${INSTALL_RAG}, Force=${FORCE}"

# ========== 检查 conda 是否可用 ==========
echo -e "${BLUE}━━━ 前置检查: conda 可用性 ━━━${NC}"
echo ""

CONDA_CMD="$(find_conda || true)"

if [[ -z "$CONDA_CMD" ]]; then
    log_error "未找到 conda 命令！请先安装 Miniconda 或 Anaconda"
    echo ""
    echo -e "${YELLOW}安装方法：${NC}"
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    exit 1
fi

CONDA_VERSION=$($CONDA_CMD --version 2>&1 | head -1)
log_info "conda 可用: ${CONDA_VERSION} (${CONDA_CMD})"
echo ""

# ========== 检查 conda 环境是否存在 ==========
check_env_exists() {
    local env_name="$1"
    [[ -n "$(conda_env_prefix "$CONDA_CMD" "$env_name" || true)" ]]
}

# ========== 安装 vLLM 环境 ==========
install_vllm_env() {
    echo -e "${BLUE}━━━ 安装 vLLM 环境 (${ENV_VLLM}) ━━━${NC}"
    echo ""

    # 检查环境是否已存在
    if check_env_exists "$ENV_VLLM"; then
        if [[ "$FORCE" == "true" ]]; then
            log_warn "环境 ${ENV_VLLM} 已存在，--force 模式下将删除并重建"
            $CONDA_CMD env remove -n "$ENV_VLLM" -y 2>&1 | tee -a "${LOG_FILE}" | tail -3
        else
            log_info "环境 ${ENV_VLLM} 已存在，执行依赖验证"
            if "$CONDA_CMD" run -n "$ENV_VLLM" python -c "import vllm, torch" >>"${LOG_FILE}" 2>&1; then
                report_add "vLLM 环境 (${ENV_VLLM})" "SKIP" "环境已存在且验证通过"
            else
                log_error "现有 ${ENV_VLLM} 环境依赖验证失败，请使用 --force 重建"
                report_add "vLLM 环境 (${ENV_VLLM})" "FAIL" "现有环境 import 验证失败"
                return 1
            fi
            echo ""
            return 0
        fi
    fi

    # 创建 conda 环境
    log_info "创建 conda 环境: ${ENV_VLLM} (Python 3.10)..."
    if ! $CONDA_CMD create -n "$ENV_VLLM" python=3.10 -y 2>&1 | tee -a "${LOG_FILE}" | tail -5; then
        log_error "创建 conda 环境 ${ENV_VLLM} 失败"
        report_add "vLLM 环境 (${ENV_VLLM})" "FAIL" "conda create 失败"
        echo ""
        return 1
    fi

    # 安装 PyTorch（必须先装，确保 CUDA 12.4 版本）
    log_info "安装 PyTorch 2.5.1+cu124..."
    if ! $CONDA_CMD run -n "$ENV_VLLM" pip install $PYTORCH_PACKAGES \
        --index-url "$PYTORCH_INDEX_URL" 2>&1 | tee -a "${LOG_FILE}" | tail -5; then
        log_error "PyTorch 安装失败"
        report_add "vLLM 环境 (${ENV_VLLM})" "FAIL" "PyTorch 安装失败"
        echo ""
        return 1
    fi

    # 安装 vLLM
    log_info "安装 vLLM 0.8.5.post1..."
    if ! $CONDA_CMD run -n "$ENV_VLLM" pip install vllm==0.8.5.post1 2>&1 | tee -a "${LOG_FILE}" | tail -5; then
        log_error "vLLM 安装失败"
        report_add "vLLM 环境 (${ENV_VLLM})" "FAIL" "vLLM 安装失败"
        echo ""
        return 1
    fi

    # 验证安装
    log_info "验证 vLLM 安装..."
    if $CONDA_CMD run -n "$ENV_VLLM" python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>&1; then
        log_info "vLLM 环境安装完成 ✓"
        report_add "vLLM 环境 (${ENV_VLLM})" "OK" "vLLM 0.8.5.post1 + PyTorch 2.5.1+cu124"
    else
        log_error "vLLM import 验证失败"
        report_add "vLLM 环境 (${ENV_VLLM})" "FAIL" "import vllm 失败"
        echo ""
        return 1
    fi

    echo ""
    return 0
}

# ========== 安装 MinerU 环境 ==========
install_mineru_env() {
    echo -e "${BLUE}━━━ 安装 MinerU 环境 (${ENV_MINERU}) ━━━${NC}"
    echo ""

    # 检查环境是否已存在
    if check_env_exists "$ENV_MINERU"; then
        if [[ "$FORCE" == "true" ]]; then
            log_warn "环境 ${ENV_MINERU} 已存在，--force 模式下将删除并重建"
            $CONDA_CMD env remove -n "$ENV_MINERU" -y 2>&1 | tee -a "${LOG_FILE}" | tail -3
        else
            log_info "环境 ${ENV_MINERU} 已存在，执行 CLI 验证"
            if "$CONDA_CMD" run -n "$ENV_MINERU" bash -c \
                "command -v mineru >/dev/null || command -v magic-pdf >/dev/null" >>"${LOG_FILE}" 2>&1; then
                report_add "MinerU 环境 (${ENV_MINERU})" "SKIP" "环境已存在且验证通过"
            else
                log_error "现有 ${ENV_MINERU} 环境验证失败，请使用 --force 重建"
                report_add "MinerU 环境 (${ENV_MINERU})" "FAIL" "现有环境 CLI 验证失败"
                return 1
            fi
            echo ""
            return 0
        fi
    fi

    # 创建 conda 环境
    log_info "创建 conda 环境: ${ENV_MINERU} (Python 3.10)..."
    if ! $CONDA_CMD create -n "$ENV_MINERU" python=3.10 -y 2>&1 | tee -a "${LOG_FILE}" | tail -5; then
        log_error "创建 conda 环境 ${ENV_MINERU} 失败"
        report_add "MinerU 环境 (${ENV_MINERU})" "FAIL" "conda create 失败"
        echo ""
        return 1
    fi

    # 安装 PyTorch GPU 版
    log_info "安装 PyTorch 2.5.1+cu124..."
    if ! $CONDA_CMD run -n "$ENV_MINERU" pip install $PYTORCH_PACKAGES \
        --index-url "$PYTORCH_INDEX_URL" 2>&1 | tee -a "${LOG_FILE}" | tail -5; then
        log_error "PyTorch 安装失败"
        report_add "MinerU 环境 (${ENV_MINERU})" "FAIL" "PyTorch 安装失败"
        echo ""
        return 1
    fi

    # 用 uv 安装 MinerU
    log_info "安装 uv 包管理器..."
    if ! $CONDA_CMD run -n "$ENV_MINERU" pip install uv 2>&1 | tee -a "${LOG_FILE}" | tail -3; then
        log_error "uv 安装失败"
        report_add "MinerU 环境 (${ENV_MINERU})" "FAIL" "uv 安装失败"
        echo ""
        return 1
    fi

    log_info "使用 uv 安装 MinerU..."
    if ! $CONDA_CMD run -n "$ENV_MINERU" uv pip install -U "mineru[all]" 2>&1 | tee -a "${LOG_FILE}" | tail -5; then
        log_error "MinerU 安装失败"
        report_add "MinerU 环境 (${ENV_MINERU})" "FAIL" "MinerU 安装失败"
        echo ""
        return 1
    fi

    # 验证安装（新版 MinerU import 名为 mineru，旧版为 magic_pdf）
    log_info "验证 MinerU 安装..."
    local mineru_verify_script="
try:
    import mineru
    print(f'MinerU {mineru.__version__} imported successfully')
except ImportError:
    import magic_pdf
    print(f'MinerU (magic-pdf) {magic_pdf.__version__} imported successfully')
"
    if $CONDA_CMD run -n "$ENV_MINERU" python -c "$mineru_verify_script" 2>&1; then
        log_info "MinerU 环境安装完成 ✓"
        report_add "MinerU 环境 (${ENV_MINERU})" "OK" "MinerU + PyTorch 2.5.1+cu124"
    else
        # import 均失败时，尝试 CLI 验证
        if $CONDA_CMD run -n "$ENV_MINERU" magic-pdf --version &>/dev/null; then
            log_info "MinerU 环境安装完成 ✓（magic-pdf CLI 可用）"
            report_add "MinerU 环境 (${ENV_MINERU})" "OK" "MinerU CLI + PyTorch 2.5.1+cu124"
        else
            log_error "MinerU import 与 CLI 验证均失败"
            report_add "MinerU 环境 (${ENV_MINERU})" "FAIL" "安装后验证失败"
            return 1
        fi
    fi

    echo ""
    return 0
}

# ========== 按唯一清单安装 RAG 主环境依赖 ==========
install_rag_deps() {
    echo -e "${BLUE}━━━ 安装 RAG 主环境依赖 ━━━${NC}"
    echo ""

    local requirements_file="${SCRIPT_DIR}/requirements-rag.txt"
    if [[ ! -f "$requirements_file" ]]; then
        log_error "缺少 RAG 依赖清单: $requirements_file"
        report_add "RAG 主环境 (${ENV_RAG})" "FAIL" "缺少 requirements-rag.txt"
        return 1
    fi
    if check_env_exists "$ENV_RAG"; then
        if [[ "$FORCE" == "true" ]]; then
            log_warn "环境 ${ENV_RAG} 已存在，--force 模式下将删除并重建"
            if ! "$CONDA_CMD" env remove -n "$ENV_RAG" -y >>"${LOG_FILE}" 2>&1; then
                report_add "RAG 主环境 (${ENV_RAG})" "FAIL" "删除旧环境失败"
                return 1
            fi
        else
            log_info "环境 ${ENV_RAG} 已存在，将在该环境中验证并更新依赖"
        fi
    fi
    if ! check_env_exists "$ENV_RAG"; then
        log_info "创建 conda 环境: ${ENV_RAG} (Python 3.10)..."
        if ! "$CONDA_CMD" create -n "$ENV_RAG" python=3.10 -y >>"${LOG_FILE}" 2>&1; then
            report_add "RAG 主环境 (${ENV_RAG})" "FAIL" "conda create 失败"
            return 1
        fi
    fi

    log_info "目标环境: ${ENV_RAG}"

    # 安装依赖
    if "$CONDA_CMD" run -n "$ENV_RAG" python -m pip install -r "$requirements_file" \
        2>&1 | tee -a "${LOG_FILE}" | tail -5; then
        log_info "RAG 核心依赖安装完成 ✓"
    else
        log_error "RAG 核心依赖安装失败"
        report_add "RAG 主环境 (${ENV_RAG})" "FAIL" "pip install 返回错误"
        return 1
    fi

    log_info "验证关键依赖..."
    if ! "$CONDA_CMD" run -n "$ENV_RAG" python -c \
        "import flask, dotenv, langchain, langchain_core, langchain_openai, langchain_huggingface, langchain_qdrant, langchain_text_splitters, qdrant_client, sentence_transformers, tiktoken, modelscope, huggingface_hub, requests" \
        >>"${LOG_FILE}" 2>&1; then
        log_error "RAG 关键依赖 import 验证失败"
        report_add "RAG 主环境 (${ENV_RAG})" "FAIL" "关键 import 失败"
        return 1
    fi
    report_add "RAG 主环境 (${ENV_RAG})" "OK" "核心依赖安装并验证通过"
    echo ""
    return 0
}

# ========== 主流程 ==========

# 执行安装
if [[ "$INSTALL_VLLM" == "true" ]]; then
    install_vllm_env || true
fi

if [[ "$INSTALL_MINERU" == "true" ]]; then
    install_mineru_env || true
fi

if [[ "$INSTALL_RAG" == "true" ]]; then
    install_rag_deps || true
fi

# ========== 环境状态报告 ==========
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                   环境安装报告                           ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""

# 打印报告表格
printf "  %-4s %-30s %s\n" "状态" "检查项" "详情"
printf "  %-4s %-30s %s\n" "────" "──────────────────────────────" "──────────────────────────"

for i in "${!REPORT_ITEMS[@]}"; do
    local_status="${REPORT_STATUS[$i]}"
    local_item="${REPORT_ITEMS[$i]}"
    local_detail="${REPORT_DETAILS[$i]}"

    case "$local_status" in
        OK)
            printf "  ${GREEN} ✓ ${NC} %-30s %s\n" "$local_item" "$local_detail"
            ;;
        FAIL)
            printf "  ${RED} ✗ ${NC} %-30s %s\n" "$local_item" "$local_detail"
            ;;
        SKIP)
            printf "  ${YELLOW} ⚠ ${NC} %-30s %s\n" "$local_item" "$local_detail"
            ;;
    esac
done

echo ""

# 显示 conda 环境状态总览
echo -e "${BLUE}━━━ Conda 环境状态 ━━━${NC}"
echo ""
if check_env_exists "$ENV_VLLM"; then
    echo -e "  ${GREEN}●${NC} ${ENV_VLLM}    — 已创建"
else
    echo -e "  ${RED}○${NC} ${ENV_VLLM}    — 未创建"
fi
if check_env_exists "$ENV_MINERU"; then
    echo -e "  ${GREEN}●${NC} ${ENV_MINERU}  — 已创建"
else
    echo -e "  ${RED}○${NC} ${ENV_MINERU}  — 未创建"
fi
if check_env_exists "$ENV_RAG"; then
    echo -e "  ${GREEN}●${NC} ${ENV_RAG}         — 已创建"
else
    echo -e "  ${RED}○${NC} ${ENV_RAG}         — 未创建"
fi
echo ""

# 汇总结论
if [[ $HAS_FAILURE -eq 0 ]]; then
    echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  ✓ 环境安装完成！${NC}"
    echo -e "${GREEN}══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  后续步骤（首次部署）："
    echo -e "    1. 编辑 .env 并准备三个本地模型"
    echo -e "    2. 创建用户:       ${GREEN}conda run -n ${ENV_RAG} python create_user.py${NC}"
    echo -e "    3. 转换文档:       ${GREEN}bash convert_to_md.sh --full${NC}"
    echo -e "    4. 启动 vLLM:      ${GREEN}bash start_vllm.sh --background${NC}"
    echo -e "    5. 知识入库:       ${GREEN}bash auto_ingest.sh --full${NC}"
    echo -e "    6. 启动 RAG 系统:  ${GREEN}bash start_rag.sh start${NC}"
    echo ""
else
    echo -e "${RED}══════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}  ✗ 部分环境安装失败，请根据上方信息排查${NC}"
    echo -e "${RED}══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "  排查建议："
    echo -e "    • 检查日志: ${LOG_FILE}"
    echo -e "    • 使用 --force 强制重建失败的环境"
    echo -e "    • 确保网络连接正常（需要下载 PyTorch/vLLM/MinerU）"
    echo ""
fi

log_info "========== 环境安装结束 (失败项: ${HAS_FAILURE}) =========="

exit $HAS_FAILURE
