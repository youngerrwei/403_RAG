#!/bin/bash
# =============================================================================
# 模型下载脚本 - 下载 Qwen3-8B-Instruct 到本地
# =============================================================================
# 用法:
#   bash download_model.sh [目标路径]
#
# 环境变量:
#   HF_MIRROR  - HuggingFace 镜像地址（默认使用 hf-mirror.com 国内镜像）
#              设置为空字符串可禁用镜像：HF_MIRROR="" bash download_model.sh
#   HF_TOKEN   - HuggingFace 访问令牌（可选，用于私有模型）
#
# 示例:
#   bash download_model.sh                          # 使用国内镜像下载到默认路径
#   bash download_model.sh ./models/my-model        # 下载到自定义路径
#   HF_MIRROR="" bash download_model.sh             # 不使用镜像，直连 HuggingFace
# =============================================================================

set -e

# ----------------------------- 配置 ------------------------------------
MODEL_ID="Qwen/Qwen3-8B-Instruct"
TARGET_DIR="${1:-./models/Qwen3-8B-Instruct}"

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

# ----------------------------- 镜像源配置 --------------------------------
# 如需直连 HuggingFace 官方源，设置 HF_MIRROR="" 即可
HF_MIRROR="${HF_MIRROR-https://hf-mirror.com}"

if [ -n "$HF_MIRROR" ]; then
    export HF_ENDPOINT="$HF_MIRROR"
    log_info "使用镜像源: $HF_MIRROR"
else
    log_info "使用 HuggingFace 官方源（未配置镜像）"
fi

# 验证模型文件完整性（检查关键文件是否存在）
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

# ----------------------------- 选择下载工具 -----------------------------
DOWNLOAD_METHOD=""

# 优先使用 huggingface-cli
if command -v huggingface-cli &>/dev/null; then
    DOWNLOAD_METHOD="cli"
    log_info "检测到 huggingface-cli，将使用 CLI 工具下载"
else
    # 检查 Python huggingface_hub 库
    if python3 -c "import huggingface_hub" 2>/dev/null; then
        DOWNLOAD_METHOD="python"
        log_info "未找到 huggingface-cli，将使用 Python huggingface_hub 库下载"
    elif python -c "import huggingface_hub" 2>/dev/null; then
        DOWNLOAD_METHOD="python_fallback"
        log_info "未找到 huggingface-cli，将使用 Python huggingface_hub 库下载"
    else
        log_error "未找到可用的下载工具！请安装以下任一工具："
        echo ""
        echo "  方式 1（推荐）: pip install huggingface_hub[cli]"
        echo "  方式 2:         pip install huggingface_hub"
        echo ""
        exit 1
    fi
fi

# ----------------------------- 执行下载 ---------------------------------
log_info "开始下载模型（支持断点续传）..."
echo ""

DOWNLOAD_SUCCESS=false

case "$DOWNLOAD_METHOD" in
    cli)
        # 构建 huggingface-cli 下载命令
        CLI_ARGS="download $MODEL_ID --local-dir $TARGET_DIR --resume-download"

        # 添加 token（如果设置了）
        if [ -n "$HF_TOKEN" ]; then
            CLI_ARGS="$CLI_ARGS --token $HF_TOKEN"
        fi

        if huggingface-cli $CLI_ARGS; then
            DOWNLOAD_SUCCESS=true
        fi
        ;;

    python|python_fallback)
        # 确定 Python 解释器
        PYTHON_CMD="python3"
        if [ "$DOWNLOAD_METHOD" = "python_fallback" ]; then
            PYTHON_CMD="python"
        fi

        # 使用 Python huggingface_hub 下载
        $PYTHON_CMD -c "
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
"
        if [ $? -eq 0 ]; then
            DOWNLOAD_SUCCESS=true
        fi
        ;;
esac

echo ""

# ----------------------------- 验证下载结果 -----------------------------
if [ "$DOWNLOAD_SUCCESS" = "false" ]; then
    log_error "下载失败！请检查网络连接或尝试使用镜像源："
    echo "  HF_MIRROR=https://hf-mirror.com bash download_model.sh"
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
