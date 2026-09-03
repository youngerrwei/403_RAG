#!/bin/bash
# 运行脚本共享工具。调用方必须先定义 SCRIPT_DIR；所有服务脚本共用本文件。

trim_value() {
    local value="${1-}"
    value="${value%$'\r'}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    if [[ ${#value} -ge 2 ]]; then
        if [[ "$value" == \"*\" && "$value" == *\" ]]; then
            value="${value:1:${#value}-2}"
        elif [[ "$value" == \'*\' && "$value" == *\' ]]; then
            value="${value:1:${#value}-2}"
        fi
    fi
    printf '%s' "$value"
}

# 只加载调用方明确列出的配置键，防止 .env 覆盖脚本内部变量。
load_env_keys() {
    local env_file="$1"
    shift
    [[ -f "$env_file" ]] || return 1
    local allowed=" $* " key value
    while IFS='=' read -r key value || [[ -n "${key:-}" ]]; do
        key="$(trim_value "${key:-}")"
        [[ -z "$key" || "$key" == \#* ]] && continue
        [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
        [[ "$allowed" == *" $key "* ]] || continue
        value="$(trim_value "${value:-}")"
        printf -v "$key" '%s' "$value"
        export "$key"
    done < "$env_file"
}

resolve_project_path() {
    local path="$1"
    if [[ "$path" = /* ]]; then
        printf '%s' "$path"
    else
        printf '%s/%s' "$SCRIPT_DIR" "${path#./}"
    fi
}

find_executable() {
    local name="$1"
    shift
    local found
    found="$(command -v "$name" 2>/dev/null || true)"
    if [[ -n "$found" && -x "$found" ]]; then
        printf '%s' "$found"
        return 0
    fi
    for found in "$@"; do
        if [[ -n "$found" && -x "$found" ]]; then
            printf '%s' "$found"
            return 0
        fi
    done
    return 1
}

find_conda() {
    local candidate
    if [[ -n "${CONDA_EXE:-}" && -x "${CONDA_EXE}" ]]; then
        printf '%s' "$CONDA_EXE"
        return 0
    fi
    candidate="$(command -v conda 2>/dev/null || true)"
    if [[ -n "$candidate" && -x "$candidate" ]]; then
        printf '%s' "$candidate"
        return 0
    fi
    for candidate in \
        /public/apps/anaconda3/bin/conda \
        "$HOME/miniconda3/bin/conda" \
        "$HOME/anaconda3/bin/conda" \
        /opt/conda/bin/conda; do
        if [[ -x "$candidate" ]]; then
            printf '%s' "$candidate"
            return 0
        fi
    done
    return 1
}

# 使用目标 conda 环境的 Python 标准库发起 HTTP 请求，避免依赖非登录 shell 中的 curl。
http_get_body() {
    local python_bin="$1" url="$2" timeout="$3" token="${4:-}"
    RAG_HTTP_TOKEN="$token" "$python_bin" - "$url" "$timeout" <<'PY'
import os
import sys
import urllib.request

url, timeout = sys.argv[1], float(sys.argv[2])
token = os.environ.get("RAG_HTTP_TOKEN", "")
headers = {"Authorization": f"Bearer {token}"} if token else {}
request = urllib.request.Request(url, headers=headers)
with urllib.request.urlopen(request, timeout=timeout) as response:
    if response.status < 200 or response.status >= 300:
        raise RuntimeError(f"HTTP {response.status}")
    sys.stdout.buffer.write(response.read())
PY
}

# Qdrant 只有在服务可达且所有必要集合存在时才可用于完整检索。
qdrant_collections_ready() {
    local python_bin="$1" url="$2" timeout="$3"
    shift 3
    "$python_bin" - "$url" "$timeout" "$@" <<'PY'
import json
import sys
import urllib.request

url, timeout = sys.argv[1], float(sys.argv[2])
required = set(sys.argv[3:])
with urllib.request.urlopen(url, timeout=timeout) as response:
    if response.status < 200 or response.status >= 300:
        raise RuntimeError(f"HTTP {response.status}")
    payload = json.load(response)
existing = {
    item.get("name")
    for item in payload.get("result", {}).get("collections", [])
    if isinstance(item, dict) and item.get("name")
}
raise SystemExit(0 if required.issubset(existing) else 1)
PY
}

http_request_status() {
    local python_bin="$1" method="$2" url="$3" timeout="$4" token="${5:-}"
    RAG_HTTP_TOKEN="$token" "$python_bin" - "$method" "$url" "$timeout" <<'PY'
import os
import sys
import urllib.error
import urllib.request

method, url, timeout = sys.argv[1], sys.argv[2], float(sys.argv[3])
token = os.environ.get("RAG_HTTP_TOKEN", "")
headers = {"Authorization": f"Bearer {token}"} if token else {}
request = urllib.request.Request(url, headers=headers, method=method)
try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
        print(response.status)
except urllib.error.HTTPError as exc:
    print(exc.code)
PY
}

conda_env_prefix() {
    local conda_cmd="$1" env_name="$2"
    "$conda_cmd" env list 2>/dev/null | awk -v env_name="$env_name" \
        '$1 == env_name { print $NF; exit }'
}

conda_env_python() {
    local conda_cmd="$1" env_name="$2" prefix
    prefix="$(conda_env_prefix "$conda_cmd" "$env_name")"
    [[ -n "$prefix" && -x "$prefix/bin/python" ]] || return 1
    printf '%s/bin/python' "$prefix"
}

port_pid() {
    local port="$1" pid=""
    if command -v lsof >/dev/null 2>&1; then
        pid="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
    elif command -v ss >/dev/null 2>&1; then
        pid="$(ss -tlnp 2>/dev/null | awk -v p=":${port}" '$4 ~ p"$" { if (match($0, /pid=[0-9]+/)) { print substr($0, RSTART+4, RLENGTH-4); exit } }')"
    fi
    printf '%s' "$pid"
}

pid_cmdline() {
    local pid="$1"
    [[ "$pid" =~ ^[0-9]+$ && -r "/proc/$pid/cmdline" ]] || return 1
    tr '\0' ' ' < "/proc/$pid/cmdline"
}

pid_is_vllm() {
    local pid="$1" cmd
    cmd="$(pid_cmdline "$pid" 2>/dev/null || true)"
    [[ "$cmd" == *"vllm.entrypoints.openai.api_server"* ]]
}

pid_is_webapp() {
    local pid="$1" cmd
    cmd="$(pid_cmdline "$pid" 2>/dev/null || true)"
    [[ "$cmd" == *"web_app.py"* ]]
}

stop_pid_gracefully() {
    local pid="$1" label="$2" wait_seconds="${3:-10}" elapsed=0
    kill "$pid" 2>/dev/null || return 1
    while kill -0 "$pid" 2>/dev/null && (( elapsed < wait_seconds )); do
        sleep 1
        elapsed=$((elapsed + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        echo "[警告] $label 未在 ${wait_seconds} 秒内退出，发送 SIGKILL"
        kill -9 "$pid" 2>/dev/null || return 1
    fi
}

normalize_model_name() {
    local model="$1"
    model="${model%/}"
    printf '%s' "${model#./}"
}

model_names_match() {
    local actual expected
    actual="$(normalize_model_name "$1")"
    expected="$(normalize_model_name "$2")"
    [[ -n "$actual" && -n "$expected" ]] || return 1
    [[ "$actual" == "$expected" || "${actual##*/}" == "${expected##*/}" ]]
}

extract_json_string() {
    local key="$1"
    sed -n "s/.*\"${key}\"[[:space:]]*:[[:space:]]*\"\([^\"]*\)\".*/\1/p" | head -n 1
}

extract_vllm_model_id() {
    local python_bin="$1"
    "$python_bin" -c '
import json
import sys

payload = json.load(sys.stdin)
models = payload.get("data") or []
if models and isinstance(models[0], dict):
    print(models[0].get("id", ""))
'
}

is_weak_flask_secret() {
    local secret="${1:-}"
    [[ ${#secret} -lt 32 ]] && return 0
    case "$secret" in
        lab403-rag-secret-key-change-me-in-production|change-me|changeme|secret|EMPTY)
            return 0 ;;
    esac
    return 1
}
