import os
import json
import traceback
import hashlib
import hmac
import queue
import threading
import time
import logging
import secrets
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from threading import Semaphore

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    Response,
    redirect,
    url_for,
    session,
    stream_with_context,
    g,
)

from rag_agent import ask_stream as ask_rag_stream, load_history, clear_user_history, get_runtime, save_history, cleanup_expired_history

from logger import get_logger

_logger = get_logger("web")

# 简单的登录速率限制
_login_attempts = defaultdict(list)  # IP -> [timestamps]
_login_attempts_lock = threading.Lock()
_LOGIN_RATE_LIMIT = 5  # 每分钟最多5次
_LOGIN_RATE_WINDOW = 60  # 60秒窗口

# 登录失败锁定机制
_login_failures = defaultdict(list)  # IP -> [failure timestamps]
_LOGIN_LOCKOUT_THRESHOLD = 5  # 5次失败后锁定
_LOGIN_LOCKOUT_DURATION = 900  # 锁定15分钟


def _check_login_rate_limit(ip: str) -> bool:
    """检查IP是否超过登录速率限制，返回True表示允许"""
    with _login_attempts_lock:
        now = time.time()
        # 清理过期记录
        _login_attempts[ip] = [t for t in _login_attempts[ip] if now - t < _LOGIN_RATE_WINDOW]
        if len(_login_attempts[ip]) >= _LOGIN_RATE_LIMIT:
            return False
        _login_attempts[ip].append(now)
        return True


def _check_login_lockout(ip: str) -> tuple:
    """检查 IP 是否被锁定"""
    with _login_attempts_lock:
        now = time.time()
        # 清理过期记录
        _login_failures[ip] = [t for t in _login_failures[ip] if now - t < _LOGIN_LOCKOUT_DURATION]
        if len(_login_failures[ip]) >= _LOGIN_LOCKOUT_THRESHOLD:
            remaining = int(_LOGIN_LOCKOUT_DURATION - (now - _login_failures[ip][0]))
            return False, f"登录尝试过多，请{remaining // 60}分钟后重试"
        return True, ""


def _record_login_failure(ip: str):
    """记录登录失败"""
    with _login_attempts_lock:
        _login_failures[ip].append(time.time())


def _clear_login_failures(ip: str):
    """登录成功后清除该 IP 的失败记录"""
    with _login_attempts_lock:
        _login_failures.pop(ip, None)


# 并发请求控制
_MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "20"))
_request_semaphore = Semaphore(_MAX_CONCURRENT_REQUESTS)

app = Flask(__name__)

# Flask 密钥加固
secret_key = os.getenv("FLASK_SECRET_KEY", "")
if not secret_key or len(secret_key) < 32:
    logging.warning("[SECURITY] FLASK_SECRET_KEY 未设置或强度不足(< 32字符)！建议运行: python -c \"import secrets; print(secrets.token_hex(32))\"")
    if not secret_key:
        secret_key = "unsafe-dev-key-change-in-production"
app.secret_key = secret_key

# Session Cookie 安全加固
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"
app.config["SESSION_COOKIE_SECURE"] = os.getenv("HTTPS_ENABLED", "false").lower() == "true"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)
app.config["SESSION_REFRESH_EACH_REQUEST"] = True
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1MB

USERS_FILE = os.getenv("USERS_FILE", "config/users.json")


@app.before_request
def _generate_csp_nonce():
    """为每个请求生成 CSP nonce"""
    g.csp_nonce = secrets.token_hex(16)


@app.before_request
def _limit_concurrent():
    """限制并发请求数（仅对问答接口）"""
    if request.endpoint == "ask_stream_api":
        if not _request_semaphore.acquire(blocking=False):
            return jsonify({"error": "服务繁忙，请稍后重试"}), 503
        # 标记已获取锁
        request._acquired_semaphore = True


@app.teardown_request
def _release_concurrent(exc=None):
    """释放并发锁"""
    if getattr(request, '_acquired_semaphore', False):
        _request_semaphore.release()


@app.after_request
def _set_security_headers(response):
    # Content Security Policy
    # script-src: nonce 保护内联脚本，允许 CDN 外部脚本
    # style-src: 允许 unsafe-inline（内联样式安全风险极低，且 MathJax 等库必须用动态样式）
    # connect-src: 允许 self 和 CDN（source map 等）
    nonce = getattr(g, 'csp_nonce', '')
    response.headers["Content-Security-Policy"] = (
        f"default-src 'self'; "
        f"script-src 'self' 'nonce-{nonce}' https://cdn.jsdelivr.net https://unpkg.com; "
        f"style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
        f"img-src 'self' data:; "
        f"font-src 'self' https://cdn.jsdelivr.net; "
        f"connect-src 'self' https://cdn.jsdelivr.net; "
        f"frame-ancestors 'self'"
    )
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "SAMEORIGIN"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


def load_users():
    try:
        if not os.path.exists(USERS_FILE):
            return []

        with open(USERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return []

        valid_users = []
        for item in data:
            if not isinstance(item, dict):
                continue
            username = str(item.get("username", "")).strip()
            password_hash = str(item.get("password_hash", "")).strip()
            if username and password_hash:
                valid_users.append(item)
        return valid_users
    except Exception:
        traceback.print_exc()
        return []


def find_user(username: str):
    username = (username or "").strip()
    if not username:
        return None
    for user in load_users():
        if user.get("username") == username:
            return user
    return None


def verify_password(username: str, password: str) -> bool:
    try:
        user = find_user(username)
        if not user:
            return False

        if user.get("hash_method", "pbkdf2_sha256") != "pbkdf2_sha256":
            return False

        salt_hex = user.get("salt", "")
        stored_hash_hex = user.get("password_hash", "")
        iterations = int(user.get("iterations", 200000))

        if not salt_hex or not stored_hash_hex:
            return False

        salt = bytes.fromhex(salt_hex)
        computed_hash = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations
        ).hex()

        return hmac.compare_digest(computed_hash, stored_hash_hex)

    except Exception:
        traceback.print_exc()
        return False


def is_logged_in() -> bool:
    return bool(session.get("logged_in"))


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if is_logged_in():
            return view_func(*args, **kwargs)
        # API 端点精确匹配
        api_endpoints = {"/ask_stream", "/clear", "/whoami", "/save_history"}
        if request.path in api_endpoints:
            return jsonify({"success": False, "error": "未登录或登录已过期，请先登录。"}), 401
        return redirect(url_for("login"))
    return wrapper


@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查端点，无需认证，检测各组件就绪状态"""
    components = {
        "embedding": False,
        "reranker": False,
        "qdrant": False,
        "llm": False,
    }

    try:
        runtime = get_runtime()
    except Exception as e:
        _logger.warning(f"健康检查：获取运行时失败: {e}")
        return jsonify({"status": "error", "components": components}), 503

    # 检查 Embedding 模型是否已加载（通过 vectorstore 内部的 embeddings 对象判断）
    try:
        vs = runtime.get("vectorstore")
        if vs and getattr(vs, "_embedding", None) is not None:
            components["embedding"] = True
        elif vs and getattr(vs, "embeddings", None) is not None:
            components["embedding"] = True
    except Exception as e:
        _logger.warning(f"健康检查：Embedding 检测异常: {e}")

    # 检查 Reranker 模型是否已加载
    try:
        reranker = runtime.get("reranker")
        if reranker is not None:
            components["reranker"] = True
    except Exception as e:
        _logger.warning(f"健康检查：Reranker 检测异常: {e}")

    # 检查 Qdrant 连接是否正常
    try:
        client = runtime.get("client")
        if client is not None:
            client.get_collections()
            components["qdrant"] = True
    except Exception as e:
        _logger.warning(f"健康检查：Qdrant 连接异常: {e}")

    # 检查 LLM (vLLM) 是否可用
    try:
        llm = runtime.get("llm")
        if llm is not None:
            components["llm"] = True
    except Exception as e:
        _logger.warning(f"健康检查：LLM 检测异常: {e}")

    # 判断整体状态
    healthy_count = sum(components.values())
    if healthy_count == len(components):
        status = "ok"
    elif healthy_count == 0:
        status = "error"
    else:
        status = "degraded"

    http_code = 503 if status == "error" else 200
    return jsonify({"status": status, "components": components}), http_code


@app.route("/login", methods=["GET", "POST"])
def login():
    try:
        if request.method == "GET":
            return render_template("login.html", nonce=g.csp_nonce)

        # 速率限制检查
        ip = request.remote_addr
        if not _check_login_rate_limit(ip):
            return jsonify({"success": False, "error": "登录尝试过于频繁，请稍后再试。"}), 429

        # 锁定检查
        allowed, lockout_msg = _check_login_lockout(ip)
        if not allowed:
            _logger.warning(f"IP {ip} 因多次登录失败被锁定")
            return jsonify({"success": False, "error": lockout_msg}), 429

        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"success": False, "error": "用户名和密码不能为空。"}), 400

        # 在 verify_password 调用前
        if len(password) > 128:
            _record_login_failure(ip)
            return jsonify({"success": False, "error": "用户名或密码错误。"}), 401  # 防超长密码攻击

        if not verify_password(username, password):
            _record_login_failure(ip)
            _logger.info(f"登录失败: user={username}, ip={ip}")
            return jsonify({"success": False, "error": "用户名或密码错误。"}), 401

        # 登录成功：清除失败记录和旧session，防止session fixation
        _clear_login_failures(ip)
        session.clear()
        session.permanent = True
        session["logged_in"] = True
        session["username"] = username

        return jsonify({
            "success": True,
            "message": "登录成功。",
            "username": username
        })
    except Exception as e:
        traceback.print_exc()
        # 不向客户端暴露详细错误
        return jsonify({"success": False, "error": "服务器内部错误，请稍后重试。"}), 500


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/save_history", methods=["POST"])
@login_required
def save_history_api():
    try:
        username = session.get("username", "")
        if not username:
            return jsonify({"success": False, "error": "无效的用户状态"}), 401

        data = request.get_json(silent=True) or {}
        new_history = data.get("history", [])

        # 读取文件中已有的历史记录（包含 rag_agent 自动追加的记录）
        existing_history = load_history(username)

        # 合并去重：以 (role, content) 为 key，跳过已存在的记录
        def _dedup_key(record):
            """生成去重键：基于 role 和 content（忽略 timestamp 等额外字段）"""
            if not isinstance(record, dict):
                return None
            role = record.get("role") or ""
            content = record.get("content") or ""
            if not content:
                return None
            return f"{role}::{content}"

        seen = set()
        merged = []

        # 先放入文件中已有的记录（保留 rag_agent 自动追加的记录）
        for record in existing_history:
            key = _dedup_key(record)
            if key and key not in seen:
                seen.add(key)
                merged.append(record)

        # 再添加前端发送的新记录（跳过已存在的）
        for record in new_history:
            key = _dedup_key(record)
            if key and key not in seen:
                seen.add(key)
                merged.append(record)

        # 将合并去重后的完整历史写回文件
        save_history(username, merged)

        return jsonify({
            "success": True,
            "message": "历史已保存"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "服务器内部错误，请稍后重试。"
        }), 500


@app.route("/", methods=["GET"])
@login_required
def index():
    username = session.get("username", "")
    user_history = load_history(username) if username else []

    return render_template(
        "index.html",
        history=user_history,
        username=username,
        nonce=g.csp_nonce
    )


@app.route("/ask_stream", methods=["POST"])
@login_required
def ask_stream_api():
    data = request.get_json(silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    username = session.get("username", "anonymous")
    use_agent = data.get("use_agent", False)

    @stream_with_context
    def generate():
        start_time = time.time()
        max_duration = 300  # 5分钟总超时
        heartbeat_interval = 15  # 15秒心跳

        # 哨兵对象，标记 stream_iter 已结束
        _SENTINEL = object()

        producer_thread = None
        heartbeat_thread = None
        stop_heartbeat = threading.Event()

        try:
            yield f"data: {json.dumps({'type': 'start', 'question': question}, ensure_ascii=False)}\n\n"

            if use_agent:
                from agent_entry import ask_agent_stream
                stream_iter = ask_agent_stream(question)
            else:
                stream_iter = ask_rag_stream(question, username=username)

            # 使用 Queue 解耦 stream_iter 和心跳发送，使心跳独立于流数据
            msg_queue = queue.Queue()

            def _producer():
                """生产者线程：从 stream_iter 读取数据放入 Queue"""
                try:
                    for item in stream_iter:
                        msg_queue.put(item)
                except Exception as e:
                    # 将异常传递给主生成器处理
                    msg_queue.put(("__stream_error__", e))
                finally:
                    msg_queue.put(_SENTINEL)

            def _heartbeat_sender():
                """心跳线程：定期向 Queue 放入心跳事件"""
                # stop_heartbeat.wait() 返回 True 表示事件已设置（应停止）
                # 返回 False 表示超时（应发送心跳）
                while not stop_heartbeat.wait(heartbeat_interval):
                    msg_queue.put({"type": "heartbeat"})

            # 启动生产者线程和心跳线程（均为守护线程，防止进程卡死）
            producer_thread = threading.Thread(target=_producer, daemon=True, name="sse-producer")
            heartbeat_thread = threading.Thread(target=_heartbeat_sender, daemon=True, name="sse-heartbeat")
            producer_thread.start()
            heartbeat_thread.start()

            stream_ended_normally = False
            while True:
                # 检查总超时
                if time.time() - start_time > max_duration:
                    _logger.warning(f"SSE 流响应超时 (user={username}, duration={time.time() - start_time:.1f}s)")
                    yield f"data: {json.dumps({'type': 'error', 'message': '响应超时'}, ensure_ascii=False)}\n\n"
                    # 超时也需要发送 [DONE] 结束标记，确保前端正确关闭连接
                    yield "data: [DONE]\n\n"
                    break

                # 从 Queue 获取数据（带超时以便定期检查 max_duration）
                try:
                    item = msg_queue.get(timeout=1.0)
                except queue.Empty:
                    continue

                # 流结束标记
                if item is _SENTINEL:
                    stream_ended_normally = True
                    break

                # 生产者线程中的异常，传递给外层 except 处理
                if isinstance(item, tuple) and len(item) == 2 and item[0] == "__stream_error__":
                    raise item[1]

                # 正常数据项，直接 yield
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

            # 流正常结束，发送 [DONE] 标记（符合 SSE 协议规范）
            if stream_ended_normally:
                yield "data: [DONE]\n\n"

        except GeneratorExit:
            # 客户端断开连接，正常情况
            _logger.info(f"客户端断开 SSE 连接 (user={username})")
        except Exception as e:
            # 服务端异常，不向客户端暴露详细错误
            _logger.error(f"SSE 生成异常 (user={username}): {repr(e)}")
            traceback.print_exc()
            try:
                yield f"data: {json.dumps({'type': 'error', 'message': '服务器内部错误，请稍后重试。'}, ensure_ascii=False)}\n\n"
                # 异常情况也需要发送 [DONE] 结束标记，确保前端正确关闭连接
                yield "data: [DONE]\n\n"
            except GeneratorExit:
                pass
        finally:
            # 通知心跳线程停止
            stop_heartbeat.set()
            # 等待线程退出（设置超时避免卡死）
            if heartbeat_thread and heartbeat_thread.is_alive():
                heartbeat_thread.join(timeout=5)
            if producer_thread and producer_thread.is_alive():
                _logger.warning(f"生产者线程未在超时内退出 (user={username})")
                producer_thread.join(timeout=5)

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response


@app.route("/clear", methods=["POST"])
@login_required
def clear():
    try:
        username = session.get("username", "")
        clear_user_history(username)
        return jsonify({
            "success": True,
            "message": "历史对话已清空。"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": "服务器内部错误，请稍后重试。"
        }), 500


@app.route("/whoami", methods=["GET"])
@login_required
def whoami():
    return jsonify({
        "success": True,
        "username": session.get("username", "")
    })


def _daily_cleanup_worker():
    """后台守护线程：每天凌晨 2:00 自动清理过期历史文件"""
    while True:
        try:
            # 计算到下一个凌晨 2:00 的等待秒数
            now = datetime.now()
            next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
            if now >= next_run:
                next_run += timedelta(days=1)
            wait_seconds = (next_run - now).total_seconds()

            time.sleep(wait_seconds)

            # 执行清理
            cleanup_expired_history()
            _logger.info("定时清理：已完成过期历史文件清理")
        except Exception as e:
            _logger.error(f"定时清理异常: {e}")
            # 出错后等待 1 小时再重试
            time.sleep(3600)


if __name__ == "__main__":
    # 启动时清理过期历史
    cleanup_expired_history()

    # 启动后台清理守护线程
    cleanup_thread = threading.Thread(target=_daily_cleanup_worker, daemon=True, name="history-cleanup")
    cleanup_thread.start()
    _logger.info("后台清理线程已启动，每天 02:00 自动清理过期历史")

    # 运行时预热：提前加载模型和连接，尽早暴露配置问题
    _logger.info("正在预热 RAG 运行时组件...")
    try:
        runtime = get_runtime()
        # 输出关键组件就绪状态
        components = []
        if runtime.get("vectorstore"):
            components.append("向量库连接")
        if runtime.get("llm"):
            components.append("LLM连接")
        if runtime.get("reranker"):
            components.append("Reranker模型")
        _logger.info(f"RAG 运行时预热成功，就绪组件: {', '.join(components)}")
    except Exception as e:
        _logger.warning(f"RAG 运行时预热失败: {e}")
        _logger.warning("服务将继续启动，但首次请求可能会延迟或失败")
        _logger.warning("请检查: 1) 模型文件是否存在 2) GPU是否可用 3) Qdrant是否可连接")

    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() in {"true", "1", "yes"}
    _logger.info(f"启动本地网页服务：http://0.0.0.0:5000 (debug={debug_mode})")
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=False, threaded=True)
