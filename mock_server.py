"""
Mock Server - 用于在不依赖 vLLM 和 Qdrant 的情况下预览前端页面效果。
运行: python mock_server.py
访问: http://127.0.0.1:5001
"""

import json
import time
import secrets

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

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Strict"


# ======================== 模拟历史对话数据 ========================

MOCK_HISTORY = [
    {"role": "user", "content": "什么是水下可见光通信？"},
    {"role": "assistant", "content": "水下可见光通信（UVLC）是一种利用可见光波段（380-780nm）在水下进行无线数据传输的技术。它相比传统水声通信具有更高的带宽和更低的时延，适用于短距离高速水下通信场景。"},
    {"role": "user", "content": "OFDM在水下光通信中有什么优势？"},
    {"role": "assistant", "content": "OFDM（正交频分复用）在水下光通信中的主要优势包括：\n1. 抗多径干扰能力强\n2. 频谱利用率高\n3. 可以灵活分配子载波功率\n4. 通过循环前缀消除符号间干扰"},
    {"role": "user", "content": "实验室目前有哪些相关设备？"},
    {"role": "assistant", "content": "根据知识库信息，实验室目前配备了LED发射模块、APD光电探测器、示波器、信号发生器等水下光通信实验设备。具体型号和数量请参考设备台账文件。"},
]


# ======================== 登录相关 ========================

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "GET":
        return render_template("login.html")

    # POST: 模拟登录，直接成功
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "mock_user").strip()

    session["logged_in"] = True
    session["username"] = username

    return jsonify({
        "success": True,
        "message": "登录成功。",
        "username": username,
    })


@app.route("/logout", methods=["GET", "POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


# ======================== 主页 ========================

@app.route("/", methods=["GET"])
def index():
    # 不强制登录验证，直接设置 session 并返回页面
    if not session.get("logged_in"):
        session["logged_in"] = True
        session["username"] = "mock_user"

    return render_template(
        "index.html",
        history=MOCK_HISTORY,
        username=session.get("username", "mock_user"),
    )


# ======================== 历史相关 mock ========================

@app.route("/save_history", methods=["POST"])
def save_history_api():
    return jsonify({"success": True, "message": "历史已保存"})


@app.route("/api/history/load", methods=["GET", "POST"])
def api_history_load():
    return jsonify({"success": True, "history": MOCK_HISTORY})


@app.route("/api/history/save", methods=["POST"])
def api_history_save():
    return jsonify({"success": True, "message": "历史已保存"})


@app.route("/clear", methods=["POST"])
def clear():
    return jsonify({"success": True, "message": "历史对话已清空。"})


@app.route("/whoami", methods=["GET"])
def whoami():
    return jsonify({
        "success": True,
        "username": session.get("username", "mock_user"),
    })


# ======================== SSE 流式问答 ========================

@app.route("/ask_stream", methods=["POST"])
def ask_stream_api():
    data = request.get_json(force=True, silent=True) or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    use_agent = data.get("use_agent", False)

    @stream_with_context
    def generate():
        # start 事件
        yield sse_event({"type": "start", "question": question})

        if use_agent:
            yield from generate_agent_mock()
        else:
            yield from generate_rag_mock()

        # end 事件
        yield sse_event({"type": "end"})

    response = Response(generate(), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    response.headers["Connection"] = "keep-alive"
    return response


def sse_event(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ======================== RAG 模式模拟 ========================

def generate_rag_mock():
    # 1. 路由元数据
    time.sleep(0.3)
    yield sse_event({
        "type": "metadata",
        "stage": "route",
        "route": "rag_search",
        "route_target": "知识库检索",
        "route_reason": "用户提问涉及专业知识",
    })

    # 2. 检索元数据
    time.sleep(0.3)
    source_map = {
        "1": {"doc_title": "水下可见光通信关键技术", "rel_path": "data/水下可见光通信关键技术.md", "section_title": "系统架构"},
        "2": {"doc_title": "水下可见光通信关键技术", "rel_path": "data/水下可见光通信关键技术.md", "section_title": "调制技术"},
        "3": {"doc_title": "基于AI的水下光信号检测研究", "rel_path": "data/20309071_韦子扬_论文.pdf", "section_title": "深度学习方案"},
    }

    yield sse_event({
        "type": "metadata",
        "stage": "retrieval",
        "rewritten_question": "水下可见光通信的关键技术有哪些？",
        "keywords": ["水下通信", "可见光", "OFDM", "调制技术"],
        "queries": ["水下可见光通信关键技术", "水下光通信调制方式", "VLC水下信道"],
        "retrievals": [
            {
                "index": 0,
                "doc_title": "水下可见光通信关键技术",
                "file_name": "水下可见光通信关键技术.md",
                "rel_path": "data/水下可见光通信关键技术.md",
                "rerank_score": 0.923,
                "preview": "水下可见光通信（UVLC）是一种利用可见光波段进行水下信息传输的技术。相比传统声学通信，UVLC具有高带宽、低延迟的优势...",
                "content": "水下可见光通信（UVLC）是一种利用可见光波段（390nm~700nm）进行水下信息传输的技术。相比传统声学通信，UVLC具有高带宽、低延迟的优势，适用于短距离高速水下数据传输场景。其基本原理是利用LED或激光器作为发射源，通过调制光信号携带信息，经水下信道传输后由光电探测器接收并解调。",
            },
            {
                "index": 1,
                "doc_title": "水下可见光通信关键技术",
                "file_name": "水下可见光通信关键技术.md",
                "rel_path": "data/水下可见光通信关键技术.md",
                "rerank_score": 0.856,
                "preview": "OFDM（正交频分复用）技术在水下可见光通信中的应用，可以有效抵抗多径效应和频率选择性衰落...",
                "content": "OFDM（正交频分复用）技术在水下可见光通信中的应用日益广泛。OFDM通过将高速数据流分配到多个正交子载波上并行传输，可以有效抵抗多径效应和频率选择性衰落，显著提高频谱利用率。在水下环境中，由于散射和吸收导致的信道时延扩展，OFDM的抗干扰能力尤为重要。",
            },
            {
                "index": 2,
                "doc_title": "基于人工智能网络的水下无线光信号检测技术研究",
                "file_name": "20309071_韦子扬_论文.pdf",
                "rel_path": "data/20309071_韦子扬_论文.pdf",
                "rerank_score": 0.781,
                "preview": "本文提出了一种基于深度学习的水下光信号检测方案，利用卷积神经网络对接收信号进行特征提取和分类...",
                "content": "本文提出了一种基于深度学习的水下光信号检测方案，利用卷积神经网络（CNN）对接收信号进行特征提取和分类，在低信噪比条件下相较传统最大似然估计方法具有更低的误码率。实验结果表明，所提方案在浑浊水质条件下仍能保持较好的检测性能。",
            },
        ],
        "source_map": source_map,
    })

    # 3. 流式文本 chunks
    chunks = [
        "水下可见光通信（UVLC）的关键技术主要包括以下几个方面：\n\n",
        "**1. 调制技术**\n\n",
        "水下可见光通信常用的调制方式包括 OOK（开关键控）、PPM（脉冲位置调制）和 OFDM（正交频分复用）。",
        "其中 OFDM 技术因其抗多径干扰能力强，在水下信道中表现优异 [来源2]。\n\n",
        "**2. 信道建模**\n\n",
        "水下光信道受到吸收和散射的双重影响，Beer-Lambert 定律常用于描述光强的衰减特性。",
        "不同水质（清水、沿海水、港口水）对信道特性有显著影响 [来源1]。\n\n",
        "**3. 信号检测与处理**\n\n",
        "传统方法采用最大似然估计等方式进行信号检测，",
        "而近年来基于深度学习的方法（如 CNN、LSTM）在复杂水下环境中展现了更强的鲁棒性 [来源3]。\n\n",
        "**4. 系统架构设计**\n\n",
        "典型的 UVLC 系统由 LED 发射端、水下信道和光电探测器接收端组成，",
        "系统设计需要综合考虑发射功率、传输距离和误码率等指标 [来源1]。",
    ]

    full_answer = ""
    for chunk in chunks:
        time.sleep(0.07)
        full_answer += chunk
        yield sse_event({"type": "chunk", "content": chunk})

    # 4. 最终结果
    time.sleep(0.3)
    yield sse_event({
        "type": "final",
        "content": full_answer,
        "file_result": None,
        "route": "rag_search",
        "route_reason": "用户提问涉及专业知识",
        "rewritten_question": "水下可见光通信的关键技术有哪些？",
        "keywords": ["水下通信", "可见光", "OFDM"],
        "queries": ["水下可见光通信关键技术"],
        "source_map": source_map,
    })


# ======================== Agent 模式模拟 ========================

def generate_agent_mock():
    # Step 1: Agent 思考
    time.sleep(0.3)
    yield sse_event({
        "type": "step_llm",
        "step": 1,
        "content": "Thought: 用户想了解水下可见光通信的关键技术，我需要先搜索知识库。\nAction: rag_qa\nAction Input: 水下可见光通信关键技术",
    })

    time.sleep(0.5)

    # Step 1: 工具调用结果
    yield sse_event({
        "type": "step_tool",
        "step": 1,
        "tool": "rag_qa",
        "arg": "水下可见光通信关键技术",
        "observation": "水下可见光通信（UVLC）的关键技术包括：1）调制技术：OOK、PPM、OFDM等；2）信道建模：Beer-Lambert定律；3）信号检测：传统ML估计和深度学习方法...",
    })

    time.sleep(0.5)

    # Step 2: Agent 继续思考
    final_content = (
        "水下可见光通信（UVLC）的关键技术主要包括以下几个方面：\n\n"
        "1. **调制技术**：包括OOK、PPM和OFDM等，其中OFDM因抗多径能力强而广泛使用。\n\n"
        "2. **信道建模**：基于Beer-Lambert定律描述光强衰减。\n\n"
        "3. **信号检测**：从传统最大似然估计发展到基于深度学习的智能检测。\n\n"
        "4. **系统架构**：LED发射 + 水下信道 + 光电探测器接收的典型架构。"
    )

    yield sse_event({
        "type": "step_llm",
        "step": 2,
        "content": (
            "Thought: 我已经获取了关于水下可见光通信关键技术的信息，现在可以整理回答了。\n"
            f"Answer: {final_content}"
        ),
    })

    time.sleep(0.3)

    # 最终答案
    yield sse_event({
        "type": "final",
        "content": final_content,
    })


# ======================== 启动 ========================

if __name__ == "__main__":
    print("Mock server running on http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
