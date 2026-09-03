#!/bin/bash
# =============================================================================
# 模型下载脚本 - 使用明确的 RAG 环境下载 Qwen3-8B-Instruct 到本地
# =============================================================================
# 用法:
#   bash download_model.sh [选项] [目标路径]
#
# 选项:
#   --source modelscope   从 ModelScope（魔搭）下载（默认，国内推荐，无需认证）
#   --source huggingface  从 HuggingFace 下载（需要 HF_TOKEN）
#
# 环境变量:
#   HF_MIRROR  - HuggingFace 镜像地址（仅 huggingface 源生效）
#   HF_TOKEN   - HuggingFace 访问令牌（仅 huggingface 源需要）
#
# 示例:
#   bash download_model.sh                              # ModelScope 下载（推荐）
#   bash download_model.sh --source huggingface         # 从 HuggingFace 下载
#   bash download_model.sh ./models/my-model            # 下载到自定义路径
#   HF_TOKEN=hf_xxx bash download_model.sh --source huggingface  # HF + Token
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/runtime_common.sh"
ENV_FILE="${RAG_ENV_FILE:-$SCRIPT_DIR/.env}"
load_env_keys "$ENV_FILE" RAG_CONDA_ENV VLLM_MODEL_NAME || true
RAG_CONDA_ENV="${RAG_CONDA_ENV:-rag}"
CONDA_CMD="$(find_conda || true)"
DOWNLOAD_PYTHON=""
[[ -n "$CONDA_CMD" ]] && DOWNLOAD_PYTHON="$(conda_env_python "$CONDA_CMD" "$RAG_CONDA_ENV" || true)"

# ----------------------------- 配置 ------------------------------------
MODEL_ID="Qwen/Qwen3-8B-Instruct"
MODELSCOPE_MODEL_ID="Qwen/Qwen3-8B"
TARGET_DIR=""
SOURCE="modelscope"  # 默认使用 ModelScope（国内无需翻墙、无需 Token）

# ----------------------------- 辅助函数 --------------------------------
log_info() {
    echo "[信息] $1"
}

log_error() {
    echo "[错误] $1" >&2
}

log_success() {
    echo "[成功] $1"
}

# ----------------------------- 参数解析 --------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            if [[ -n "$2" && "$2" != --* ]]; then
                SOURCE="$2"
                shift
            else
                log_error "--source 参数需要指定来源：modelscope 或 huggingface"
                exit 1
            fi
            ;;
        --help|-h)
            echo "用法: bash download_model.sh [--source modelscope|huggingface] [目标路径]"
            echo ""
            echo "选项:"
            echo "  --source modelscope    从 ModelScope 下载（默认，国内推荐）"
            echo "  --source huggingface   从 HuggingFace 下载（需要 Token）"
            echo ""
            echo "环境变量:"
            echo "  HF_TOKEN    HuggingFace 认证令牌（仅 huggingface 源需要）"
            echo "  HF_MIRROR   HuggingFace 镜像地址（仅 huggingface 源生效）"
            exit 0
            ;;
        *)
            # 非选项参数作为目标路径
            TARGET_DIR="$1"
            ;;
    esac
    shift
done

# 设置默认目标路径
TARGET_DIR="${TARGET_DIR:-${VLLM_MODEL_NAME:-./models/Qwen3-8B-Instruct}}"
TARGET_DIR="$(resolve_project_path "$TARGET_DIR")"

# 验证 source 参数
if [[ "$SOURCE" != "modelscope" && "$SOURCE" != "huggingface" ]]; then
    log_error "无效的下载源: $SOURCE（可选: modelscope, huggingface）"
    exit 1
fi

# ----------------------------- 镜像源配置 --------------------------------
if [ "$SOURCE" = "huggingface" ]; then
    HF_MIRROR="${HF_MIRROR-https://hf-mirror.com}"
    if [ -n "$HF_MIRROR" ]; then
        export HF_ENDPOINT="$HF_MIRROR"
        log_info "HuggingFace 镜像源: $HF_MIRROR"
    fi
fi

# ----------------------------- 验证模型完整性 -----------------------------
verify_model() {
    local model_dir="$1"
    local missing_files=()

    # Qwen3-8B-Instruct 关键文件列表
    local required_files=(
        "config.json"
        "tokenizer.json"
        "tokenizer_config.json"
    )

    for f in "${required_files[@]}"; do
        if [ ! -f "$model_dir/$f" ]; then
            missing_files+=("$f")
        fi
    done

    # 检查是否存在模型权重文件（safetensors 或 bin 格式）
    local has_weights=false
    if ls "$model_dir"/*.safetensors 1>/dev/null 2>&1; then
        has_weights=true
    elif ls "$model_dir"/*.bin 1>/dev/null 2>&1; then
        has_weights=true
    fi

    if [ "$has_weights" = "false" ]; then
        missing_files+=("模型权重文件 (*.safetensors 或 *.bin)")
    fi

    if [ ${#missing_files[@]} -gt 0 ]; then
        log_error "模型验证失败，以下文件缺失:"
        for f in "${missing_files[@]}"; do
            echo "  - $f"
        done
        return 1
    fi

    return 0
}

# ----------------------------- 主流程 ----------------------------------
echo "============================================="
echo "  模型下载工具"
echo "============================================="
echo "  模型:     $MODEL_ID"
echo "  目标路径: $TARGET_DIR"
echo "  下载源:   $SOURCE"
echo "============================================="
echo ""

# 检查目标目录是否已存在且包含完整模型
if [ -d "$TARGET_DIR" ]; then
    if verify_model "$TARGET_DIR" 2>/dev/null; then
        log_success "模型已存在且完整: $TARGET_DIR"
        echo "如需重新下载，请先删除目标目录: rm -rf $TARGET_DIR"
        exit 0
    else
        log_info "目标目录已存在但模型不完整，将继续下载（断点续传）..."
    fi
fi

# 创建父目录
mkdir -p "$(dirname "$TARGET_DIR")"

# =============================================================================
# ModelScope 下载逻辑
# =============================================================================
download_from_modelscope() {
    log_info "从 ModelScope（魔搭）下载，国内直连无需认证"
    echo ""

    # 检测下载工具
    local ms_method=""

    if command -v modelscope &>/dev/null; then
        ms_method="cli"
        log_info "检测到 modelscope CLI"
    elif [[ -n "$DOWNLOAD_PYTHON" ]] && "$DOWNLOAD_PYTHON" -c "import modelscope" 2>/dev/null; then
        ms_method="python"
        log_info "检测到 ${RAG_CONDA_ENV} 环境中的 modelscope 库"
    else
        log_error "未找到 modelscope 工具！请先安装："
        echo ""
        echo "  pip install modelscope"
        echo ""
        echo "或切换到 HuggingFace 源（需要翻墙或配置 Token）："
        echo "  bash download_model.sh --source huggingface"
        echo ""
        return 1
    fi

    log_info "开始下载模型（支持断点续传）..."
    echo ""

    case "$ms_method" in
        cli)
            if modelscope download --model "$MODELSCOPE_MODEL_ID" --local_dir "$TARGET_DIR"; then
                return 0
            fi
            ;;
        python)
            if "$DOWNLOAD_PYTHON" -c "
from modelscope import snapshot_download
snapshot_download(
    model_id='${MODELSCOPE_MODEL_ID}',
    local_dir='${TARGET_DIR}',
)
print('下载完成')
"; then
                return 0
            fi
            ;;
    esac

    return 1
}

# =============================================================================
# HuggingFace 下载逻辑
# =============================================================================
download_from_huggingface() {
    # Token 检查
    HF_TOKEN="${HF_TOKEN:-}"
    if [ -z "$HF_TOKEN" ]; then
        log_info "[提示] 未设置 HF_TOKEN 环境变量"
        echo "  如果下载出现 401 错误，说明需要认证，请："
        echo "    1. 在 https://huggingface.co/${MODEL_ID} 接受模型许可协议"
        echo "    2. 在 https://huggingface.co/settings/tokens 创建 Access Token"
        echo "    3. 设置环境变量后重新运行："
        echo "       HF_TOKEN=hf_xxx bash download_model.sh --source huggingface"
        echo ""
        echo "  或改用 ModelScope 源（国内推荐，无需认证）："
        echo "       bash download_model.sh --source modelscope"
        echo ""
    fi

    # 检测下载工具
    local hf_method=""

    if command -v hf &>/dev/null; then
        hf_method="hf_cli"
        log_info "检测到 hf CLI"
    elif command -v huggingface-cli &>/dev/null; then
        hf_method="huggingface_cli"
        log_info "检测到 huggingface-cli"
    elif [[ -n "$DOWNLOAD_PYTHON" ]] && "$DOWNLOAD_PYTHON" -c "import huggingface_hub" 2>/dev/null; then
        hf_method="python"
        log_info "使用 ${RAG_CONDA_ENV} 环境中的 huggingface_hub 库下载"
    else
        log_error "未找到 HuggingFace 下载工具！请安装："
        echo ""
        echo "  pip install 'huggingface_hub[cli]'"
        echo ""
        echo "或改用 ModelScope 源（国内推荐）："
        echo "  bash download_model.sh --source modelscope"
        echo ""
        return 1
    fi

    log_info "开始下载模型（支持断点续传）..."
    echo ""

    case "$hf_method" in
        hf_cli|huggingface_cli)
            local cli_cmd="hf"
            [ "$hf_method" = "huggingface_cli" ] && cli_cmd="huggingface-cli"

            local cli_args="download $MODEL_ID --local-dir $TARGET_DIR --resume-download"
            if [ -n "$HF_TOKEN" ]; then
                cli_args="$cli_args --token $HF_TOKEN"
            fi

            if $cli_cmd $cli_args; then
                return 0
            fi
            ;;
        python)
            if "$DOWNLOAD_PYTHON" -c "
from huggingface_hub import snapshot_download
import os

token = os.environ.get('HF_TOKEN', None)

snapshot_download(
    repo_id='${MODEL_ID}',
    local_dir='${TARGET_DIR}',
    resume_download=True,
    token=token,
)
print('下载完成')
"; then
                return 0
            fi
            ;;
    esac

    return 1
}

# ----------------------------- 执行下载 ---------------------------------
DOWNLOAD_SUCCESS=false

case "$SOURCE" in
    modelscope)
        if download_from_modelscope; then
            DOWNLOAD_SUCCESS=true
        fi
        ;;
    huggingface)
        if download_from_huggingface; then
            DOWNLOAD_SUCCESS=true
        fi
        ;;
esac

echo ""

# ----------------------------- 验证下载结果 -----------------------------
if [ "$DOWNLOAD_SUCCESS" = "false" ]; then
    log_error "下载失败！"
    echo ""
    if [ "$SOURCE" = "huggingface" ]; then
        echo "  可能原因："
        echo "    1. 401 错误 → 需要 HF_TOKEN 认证"
        echo "    2. 网络不通 → 国内推荐使用 ModelScope 源"
        echo ""
        echo "  推荐解决方案（国内用户）："
        echo "    bash download_model.sh --source modelscope"
        echo ""
        echo "  或设置 Token 后重试："
        echo "    HF_TOKEN=hf_xxx HF_MIRROR=https://hf-mirror.com bash download_model.sh --source huggingface"
    else
        echo "  可能原因："
        echo "    1. modelscope 库未安装 → pip install modelscope"
        echo "    2. 网络连接失败 → 检查网络后重试"
        echo ""
        echo "  备选方案（需要翻墙或 Token）："
        echo "    HF_TOKEN=hf_xxx bash download_model.sh --source huggingface"
    fi
    echo ""
    exit 1
fi

log_info "下载完成，正在验证模型文件完整性..."

if verify_model "$TARGET_DIR"; then
    echo ""
    echo "============================================="
    log_success "模型下载并验证成功！"
    echo "  模型路径: $TARGET_DIR"
    echo "  模型大小: $(du -sh "$TARGET_DIR" 2>/dev/null | cut -f1)"
    echo "============================================="
    echo ""
    echo "现在可以启动 vLLM 服务："
    echo "  bash start_vllm.sh"
else
    echo ""
    log_error "模型文件验证失败，部分关键文件缺失"
    echo "建议重新运行下载脚本（将自动断点续传）："
    echo "  bash download_model.sh"
    exit 1
fi
