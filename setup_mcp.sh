#!/bin/bash
# 安装独立 MCP 环境并生成本机内部接口 Token；不改变原 RAG 环境。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/runtime_common.sh"

ENV_FILE="${RAG_ENV_FILE:-$SCRIPT_DIR/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
    echo "[错误] 缺少本机 .env，请先运行 bash setup_env.sh" >&2
    exit 1
fi

load_env_keys "$ENV_FILE" MCP_CONDA_ENV MCP_INTERNAL_TOKEN || true
MCP_CONDA_ENV="${MCP_CONDA_ENV:-rag-mcp}"
MCP_INTERNAL_TOKEN="${MCP_INTERNAL_TOKEN:-}"

generate_token() {
    local openssl_bin
    openssl_bin="$(find_executable openssl /usr/bin/openssl /usr/local/bin/openssl || true)"
    if [[ -n "$openssl_bin" ]]; then
        "$openssl_bin" rand -hex 32
    elif [[ -r /dev/urandom ]] && command -v od >/dev/null 2>&1; then
        od -An -N32 -tx1 /dev/urandom | tr -d ' \n'
    else
        echo "[错误] 无法生成 MCP_INTERNAL_TOKEN（缺少 openssl 和 /dev/urandom）" >&2
        return 1
    fi
}

if [[ ${#MCP_INTERNAL_TOKEN} -lt 32 ]]; then
    new_token="$(generate_token)"
    temp_file="$(mktemp "${ENV_FILE}.tmp.XXXXXX")"
    awk -v token="$new_token" '
        BEGIN { replaced = 0 }
        /^MCP_INTERNAL_TOKEN=/ { print "MCP_INTERNAL_TOKEN=" token; replaced = 1; next }
        { print }
        END { if (!replaced) print "MCP_INTERNAL_TOKEN=" token }
    ' "$ENV_FILE" > "$temp_file"
    mv -f "$temp_file" "$ENV_FILE"
    chmod 600 "$ENV_FILE" 2>/dev/null || true
    echo "[信息] 已生成本机 MCP_INTERNAL_TOKEN（Token 未输出）"
fi

CONDA_CMD="$(find_conda || true)"
if [[ -z "$CONDA_CMD" ]]; then
    echo "[错误] 未找到 conda" >&2
    exit 1
fi

if ! "$CONDA_CMD" env list 2>/dev/null | grep -qE "^${MCP_CONDA_ENV}[[:space:]]"; then
    echo "[信息] 创建独立 MCP 环境: $MCP_CONDA_ENV"
    "$CONDA_CMD" create -n "$MCP_CONDA_ENV" python=3.10 -y
else
    echo "[信息] 复用现有 MCP 环境: $MCP_CONDA_ENV"
fi

echo "[信息] 安装 MCP Python SDK v2"
"$CONDA_CMD" run -n "$MCP_CONDA_ENV" pip install "mcp[cli]>=2,<3"
"$CONDA_CMD" run -n "$MCP_CONDA_ENV" python -c "from mcp.server import MCPServer"

echo "[信息] MCP 环境准备完成。启动命令由客户端配置为: bash $SCRIPT_DIR/start_mcp.sh"
