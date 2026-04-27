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

from rag_agent import ask_stream as ask_rag_stream, load_history, clear_user_history, get_runtime, save_history

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


@app.route("/save_history", methods=["POST"])
@login_required
def save_history_api():
    try:
        username = session.get("username", "")
        if not username:
            return jsonify({"success": False, "error": "无效的用户状态"}), 401

        data = request.get_json(silent=True) or {}
        new_history = data.get("history", [])

        # 覆写保存给对应用户
        save_history(username, new_history)

        return jsonify({
            "success": True,
            "message": "历史已保存"
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": f"保存失败: {e}"
        }), 500


@app.route("/", methods=["GET"])
@login_required
def index():
    username = session.get("username", "")
    user_history = load_history(username) if username else []

    return render_template(
        "index.html",
        history=user_history,
        username=username
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

                for item in ask_rag_stream(question, username=username):
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
        clear_user_history(username)
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