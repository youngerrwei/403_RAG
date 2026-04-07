# import os
# import json
# import traceback
# from functools import wraps
#
# import hashlib
# import hmac
# from flask import (
#     Flask,
#     render_template,
#     request,
#     jsonify,
#     Response,
#     redirect,
#     url_for,
#     session,
#     stream_with_context,
# )
#
# from rag_core import ask_rag_stream, clear_history, init_runtime, chat_history
#
# app = Flask(__name__)
#
# # ================= 基础配置 =================
#
# app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace_this_with_a_strong_secret_key")
#
# app.config["SESSION_COOKIE_HTTPONLY"] = True
# app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
# app.config["PERMANENT_SESSION_LIFETIME"] = 60 * 60 * 8  # 8小时
#
# USERS_FILE = os.getenv("USERS_FILE", "config/users.json")
#
#
# # ================= 工具函数 =================
#
# def load_users():
#     """
#     从本地 JSON 文件加载用户列表。
#     文件格式示例：
#     [
#       {
#         "username": "admin",
#         "password_hash": "$2b$12$xxxxxx"
#       }
#     ]
#     """
#     try:
#         if not os.path.exists(USERS_FILE):
#             print(f"[WARN] 用户配置文件不存在: {USERS_FILE}")
#             return []
#
#         with open(USERS_FILE, "r", encoding="utf-8") as f:
#             data = json.load(f)
#
#         if not isinstance(data, list):
#             print(f"[WARN] 用户配置文件格式错误，顶层必须为 list: {USERS_FILE}")
#             return []
#
#         valid_users = []
#         for item in data:
#             if not isinstance(item, dict):
#                 continue
#
#             username = str(item.get("username", "")).strip()
#             password_hash = str(item.get("password_hash", "")).strip()
#
#             if username and password_hash:
#                 valid_users.append({
#                     "username": username,
#                     "password_hash": password_hash
#                 })
#
#         return valid_users
#
#     except Exception as e:
#         print(f"[ERROR] 加载用户配置失败: {e}")
#         traceback.print_exc()
#         return []
#
#
# def find_user(username: str):
#     """
#     按用户名查找用户。
#     """
#     try:
#         username = (username or "").strip()
#         if not username:
#             return None
#
#         users = load_users()
#         for user in users:
#             if user["username"] == username:
#                 return user
#
#         return None
#
#     except Exception as e:
#         print(f"[ERROR] 查找用户失败: {e}")
#         traceback.print_exc()
#         return None
#
#
# def verify_password(username: str, password: str) -> bool:
#     """
#     使用 PBKDF2-HMAC-SHA256 校验用户名和密码。
#     """
#     try:
#         if not username or not password:
#             return False
#
#         user = find_user(username)
#         if not user:
#             return False
#
#         hash_method = user.get("hash_method", "pbkdf2_sha256")
#         if hash_method != "pbkdf2_sha256":
#             print(f"[WARN] 不支持的哈希方法: {hash_method}")
#             return False
#
#         salt_hex = user.get("salt", "")
#         stored_hash_hex = user.get("password_hash", "")
#         iterations = int(user.get("iterations", 200000))
#
#         if not salt_hex or not stored_hash_hex:
#             print("[WARN] 用户密码记录不完整。")
#             return False
#
#         salt = bytes.fromhex(salt_hex)
#         password_bytes = password.encode("utf-8")
#
#         computed_hash = hashlib.pbkdf2_hmac(
#             "sha256",
#             password_bytes,
#             salt,
#             iterations
#         ).hex()
#
#         return hmac.compare_digest(computed_hash, stored_hash_hex)
#
#     except Exception as e:
#         print(f"[ERROR] 密码校验失败: {e}")
#         traceback.print_exc()
#         return False
#
#
# def is_logged_in() -> bool:
#     """
#     判断当前用户是否已登录。
#     """
#     return bool(session.get("logged_in"))
#
#
# def login_required(view_func):
#     """
#     登录保护装饰器。
#     未登录时：
#     - 页面请求：跳转登录页
#     - 接口请求：返回 JSON 错误
#     """
#     @wraps(view_func)
#     def wrapper(*args, **kwargs):
#         if is_logged_in():
#             return view_func(*args, **kwargs)
#
#         if request.path.startswith("/ask") or request.path.startswith("/clear") or request.path.startswith("/whoami"):
#             return jsonify({
#                 "success": False,
#                 "error": "未登录或登录已过期，请先登录。"
#             }), 401
#
#         return redirect(url_for("login"))
#
#     return wrapper
#
#
# # ================= 登录 / 退出 =================
#
# @app.route("/login", methods=["GET", "POST"])
# def login():
#     """
#     登录页面与登录提交。
#     """
#     try:
#         if request.method == "GET":
#             return render_template("login.html")
#
#         data = request.get_json(silent=True)
#
#         if data:
#             username = (data.get("username") or "").strip()
#             password = data.get("password") or ""
#         else:
#             username = (request.form.get("username") or "").strip()
#             password = request.form.get("password") or ""
#
#         if not username or not password:
#             return jsonify({
#                 "success": False,
#                 "error": "用户名和密码不能为空。"
#             }), 400
#
#         if not verify_password(username, password):
#             return jsonify({
#                 "success": False,
#                 "error": "用户名或密码错误。"
#             }), 401
#
#         session.permanent = True
#         session["logged_in"] = True
#         session["username"] = username
#
#         return jsonify({
#             "success": True,
#             "message": "登录成功。",
#             "username": username
#         })
#
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({
#             "success": False,
#             "error": f"登录失败: {e}"
#         }), 500
#
#
# @app.route("/logout", methods=["GET", "POST"])
# def logout():
#     """
#     退出登录。
#     """
#     try:
#         session.clear()
#         return redirect(url_for("login"))
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({
#             "success": False,
#             "error": f"退出失败: {e}"
#         }), 500
#
#
# # ================= 页面 =================
#
# @app.route("/", methods=["GET"])
# @login_required
# def index():
#     """
#     首页。
#     """
#     return render_template(
#         "index.html",
#         history=chat_history,
#         username=session.get("username", "")
#     )
#
#
# # ================= 流式问答接口 =================
#
# @app.route("/ask_stream", methods=["POST"])
# @login_required
# def ask_stream():
#     """
#     流式接口。
#     """
#     try:
#         data = request.get_json(silent=True) or {}
#         question = (data.get("question") or "").strip()
#
#         if not question:
#             return jsonify({"success": False, "error": "问题不能为空。"}), 400
#
#         @stream_with_context
#         def generate():
#             try:
#                 for item in ask_rag_stream(question):
#                     yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
#
#             except GeneratorExit:
#                 print("[INFO] 客户端已断开 SSE 连接。")
#
#             except Exception as e:
#                 traceback.print_exc()
#                 error_msg = {"type": "error", "content": str(e)}
#                 yield f"data: {json.dumps(error_msg, ensure_ascii=False)}\n\n"
#
#         response = Response(generate(), mimetype="text/event-stream; charset=utf-8")
#         response.headers["Cache-Control"] = "no-cache"
#         response.headers["Pragma"] = "no-cache"
#         response.headers["X-Accel-Buffering"] = "no"
#         response.headers["Connection"] = "keep-alive"
#         return response
#
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({"success": False, "error": f"请求处理失败: {e}"}), 500
#
#
# @app.route("/clear", methods=["POST"])
# @login_required
# def clear():
#     """
#     清空历史对话。
#     """
#     try:
#         clear_history()
#         return jsonify({
#             "success": True,
#             "message": "历史对话已清空。"
#         })
#     except Exception as e:
#         traceback.print_exc()
#         return jsonify({
#             "success": False,
#             "error": f"清空失败: {e}"
#         }), 500
#
#
# @app.route("/whoami", methods=["GET"])
# @login_required
# def whoami():
#     """
#     查看当前登录用户。
#     """
#     return jsonify({
#         "success": True,
#         "username": session.get("username", "")
#     })
#
#
# # ================= 启动 =================
#
# if __name__ == "__main__":
#     print("正在预热，请稍候...")
#     init_runtime()
#     print("启动本地网页服务：http://0.0.0.0:5000")
#     app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)


import os
import json
import traceback
import hashlib
import hmac
from datetime import timedelta
from functools import wraps

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
)

from rag_agent import ask_stream as ask_rag_stream, clear_history, get_runtime, chat_history

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY", "replace_this_with_a_strong_secret_key")
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(hours=8)

USERS_FILE = os.getenv("USERS_FILE", "config/users.json")


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

        if request.path.startswith("/ask") or request.path.startswith("/clear") or request.path.startswith("/whoami"):
            return jsonify({
                "success": False,
                "error": "未登录或登录已过期，请先登录。"
            }), 401

        return redirect(url_for("login"))
    return wrapper


@app.route("/login", methods=["GET", "POST"])
def login():
    try:
        if request.method == "GET":
            return render_template("login.html")

        data = request.get_json(silent=True) or {}
        username = (data.get("username") or "").strip()
        password = data.get("password") or ""

        if not username or not password:
            return jsonify({"success": False, "error": "用户名和密码不能为空。"}), 400

        if not verify_password(username, password):
            return jsonify({"success": False, "error": "用户名或密码错误。"}), 401

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
        return jsonify({"success": False, "error": f"登录失败: {e}"}), 500


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/", methods=["GET"])
@login_required
def index():
    return render_template(
        "index.html",
        history=chat_history,
        username=session.get("username", "")
    )


@app.route("/ask_stream", methods=["POST"])
@login_required
def ask_stream_api():
    try:
        data = request.get_json(silent=True) or {}
        question = (data.get("question") or "").strip()

        if not question:
            return jsonify({"success": False, "error": "问题不能为空。"}), 400

        username = session.get("username", "")

        @stream_with_context
        def generate():
            try:
                yield f"data: {json.dumps({'type': 'start', 'question': question, 'username': username}, ensure_ascii=False)}\n\n"

                for item in ask_rag_stream(question):
                    yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

                yield f"data: {json.dumps({'type': 'end'}, ensure_ascii=False)}\n\n"

            except GeneratorExit:
                print("[INFO] 客户端已断开 SSE 连接。")

            except Exception as e:
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

        response = Response(generate(), mimetype="text/event-stream")
        response.headers["Cache-Control"] = "no-cache, no-transform"
        response.headers["Pragma"] = "no-cache"
        response.headers["X-Accel-Buffering"] = "no"
        return response

    except Exception as e:
        traceback.print_exc()
        return jsonify({"success": False, "error": f"请求处理失败: {e}"}), 500


@app.route("/clear", methods=["POST"])
@login_required
def clear():
    try:
        clear_history()
        return jsonify({
            "success": True,
            "message": "历史对话已清空。"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"清空失败: {e}"
        }), 500


@app.route("/whoami", methods=["GET"])
@login_required
def whoami():
    return jsonify({
        "success": True,
        "username": session.get("username", "")
    })


if __name__ == "__main__":
    print("正在预热，请稍候...")
    get_runtime()
    print("启动本地网页服务：http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)