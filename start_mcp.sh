#!/bin/bash
# 由 MCP Host 按需启动；stdout 必须完全保留给 stdio 协议。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/runtime_common.sh"

ENV_FILE="${RAG_ENV_FILE:-$SCRIPT_DIR/.env}"
load_env_keys "$ENV_FILE" MCP_CONDA_ENV MCP_INTERNAL_TOKEN || true
MCP_CONDA_ENV="${MCP_CONDA_ENV:-rag-mcp}"
MCP_INTERNAL_TOKEN="${MCP_INTERNAL_TOKEN:-}"

if [[ ${#MCP_INTERNAL_TOKEN} -lt 32 ]]; then
    echo "[错误] MCP_INTERNAL_TOKEN 未配置或强度不足，请先运行 bash setup_mcp.sh" >&2
    exit 1
fi

CONDA_CMD="$(find_conda || true)"
if [[ -z "$CONDA_CMD" ]]; then
    echo "[错误] 未找到 conda，请先运行 bash setup_mcp.sh" >&2
    exit 1
fi

MCP_PYTHON="$(conda_env_python "$CONDA_CMD" "$MCP_CONDA_ENV" || true)"
if [[ -z "$MCP_PYTHON" ]]; then
    echo "[错误] MCP 环境不存在或无 Python: $MCP_CONDA_ENV；请先运行 bash setup_mcp.sh" >&2
    exit 1
fi

cd "$SCRIPT_DIR"
exec "$MCP_PYTHON" "$SCRIPT_DIR/mcp_server.py"
