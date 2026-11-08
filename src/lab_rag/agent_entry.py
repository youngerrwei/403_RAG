# agent_entry.py：Agent 模式的单一 ReAct 编排器（不持有独立模型实例）。
#
# 结构约定（3.4.3）：
# - `ask_agent_stream` 是唯一的编排循环，产出统一 SSE 事件流（协议见 AGENTS.md 第 7 节）。
# - `ask_agent` 只是消费同一事件流的非流式薄适配器，不再另写一套循环。
# - 工具元数据、校验、分派统一由 `agent_tools.TOOL_SPECS` 提供，禁止在此硬编码工具名分支。
# - RAG 模式（use_agent=false）完全不经过本模块，由 web_app 直接调用 rag_agent.ask_stream。

import json
import time
import traceback
import re
from typing import Dict, Any, List, Optional, Generator

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .logger import get_logger
from .paths import PROJECT_ROOT
from .rag_agent import (
    append_user_chat_history,
    build_catalog_answer_section,
    get_runtime,
    load_config,
    route_query,
    rule_based_route,
    sanitize_generated_answer,
)
from .agent_tools import (CONTENT_TOOL_NAME, dispatch_tool, render_tool_prompt_section,
                          tool_names, tool_role)


load_dotenv(PROJECT_ROOT / ".env")
_logger = get_logger("agent")


def debug_log(*args):
    _logger.debug(" ".join(str(arg) for arg in args))


def build_qwen_llm() -> ChatOpenAI:
    """返回主 RAG 运行时中的同一个 LLM 实例（不新建模型）。"""
    return get_runtime()["llm"]


_PROMPT_HEAD = """你是实验室内部知识库助手，采用 ReAct 思维链 + 工具调用方式工作。

## 知识库的实际结构（据此判断信息是否充分）
语料按「文档 > 路径（小组/设备/Manuals）> 章节 > 段落摘要」组织，检索结果会带
【文档：文件名 > 路径：小组 > 设备 > Manuals > 章节：标题层级 > 段落摘要：…】这类前缀。
- `设备操作指南/<小组>/<设备名>/Manuals` 下是中英文手册、软件安装包与较深的 SDK 子目录；
  文件名/标题常含设备型号（如 MSO-X 3054A）与英文手册名。
- `组会资料/<日期>期末组会/小组期末总结/` 是按日期与小组归档的总结与分享，
  会议资料混合个人收获、论文写作经验与研究内容，标题未必含小组名。
- `参考书/` 是主题书籍；另有团队文档要求、论文格式规范等主题文档。
目录枚举只列目标目录的直接子级，不会自动展开更深层的 `Manuals` / SDK 子目录；
要定位这类子目录，需用「设备名 + 子目录名」组合关键词再次枚举。
构造检索词时保留原文中的小组名、设备型号、日期戳（如 20230106），不要翻译或改写。

"""

_PROMPT_TAIL = """

## 输出格式（ReAct）
- 回答前先 Thought，再决定是否 Action。
- 每轮**只能输出一个 Action 行**，严格单独成行：Action: 工具名[参数值]
  例如：Action: list_group_files["VLC小组"]
- 工具返回以 "Observation:" 开头，你需要基于它继续 Thought/Action，或给出最终 Answer。
- 信息足够时输出：Answer: 你的最终回答内容
- Observation 过长时会被压缩（中段标为省略），更早轮次可能被折叠为 `[已折叠]` 占位；
  此时如需细节请换更具体的关键词重新检索，不要假设信息仍完整可用。

## 多轮检索决策规则（重要）
每次拿到 Observation 后，先判断信息是否**覆盖用户问题的每一个子问**：
- 只有列出/定位类问题：目录结果已给出完整清单即算充分。
- 同时要求清单与内容说明（如「有哪些设备及使用规范」）：目录结果只回答了清单部分，
  **必须再调用内容工具（read_file 或 rag_qa）读正文**，且第二轮问题要保留用户原始意图，不能换成无关的固定话术。
- 已通过 list_group_files 定位到具体文档/目录、而用户要的是该文档的详细用法/步骤/参数时，
  **优先用 read_file 精读该文档**（内容更完整、更聚焦），而非仅用 rag_qa 全局检索；
  rag_qa 更适合跨多文档综合检索或用户未指明具体文档的场景。
- 涉及多个小组、设备或文档的对比/分别说明：目录或正文只覆盖其中一个实体时，
  **必须分别检索其余实体**，再综合，不能凭单一文档作答。
- 正文检索结果与问题不相关、只命中同主题的其他文档、或目录未命中：信息仍不充分。

## 关联性 / 链式任务分解规则（重要）
当任务既要先定位/检索一批条目（论文、文档、设备、参数），又要求对其中若干做进一步
处理（提取、摘要、对比、归纳、梳理）时，按「先定位 → 分批处理 → 最后综合」的顺序：
1) 先用一次检索（rag_qa 或 list_group_files）拿到这批条目的候选与来源；
2) 针对具体条目用更聚焦的 query 逐个二次检索（如「X 论文的摘要/要点」），
   不要试图用一次宽泛检索就给出所有条目的详细总结；
3) 综合时明确引用来源。
条目不多（≤3）时可尝试一次检索覆盖；条目较多或需各自来源时，务必分批检索，
避免单次 Observation 过长或被压缩而丢失细节。

信息不充分时，继续调用工具，但必须**换一个检索角度**，禁止用完全相同的参数重复调用同一工具：
- 目录未命中 → 改用更窄的「设备名 + 子目录名」或更宽的「仅小组名」关键词再枚举一次；
- 目录命中但缺正文 → 用命中的具体设备/文件名或用户原始问题调用 rag_qa；
- 正文答非所问 → 用原始问题、型号、日期戳、章节名等更具辨识度的词重新构造 query；
- 跨实体问题 → 逐个实体分别检索。
工具调用有步数预算与总耗时预算；若已换过角度仍无依据，或预算即将用尽，就基于已获得的
依据给出最完整的回答，并明确说明哪一部分未在知识库中检索到依据，不要反复重试同一角度。

## 重要约束
- 回答必须优先依据工具返回，不得凭空编造。
- 「格式是什么/怎么写/要求/规范/步骤/方法」等是在查询文档内容，必须直接调用内容工具（rag_qa 或 read_file）；
  若已通过 list_group_files 定位到具体文档，用 read_file 精读该文档比 rag_qa 更直接。
- 目录工具未找到结果不代表正文没有答案；应使用用户原始问题再调用 rag_qa，
  禁止改为输出通用模板、通用规范或知识库外常识。
- 未获得知识库依据时，不得输出「通用规范」「通用模板」或虚构「主要参考」。
- 总结时请明确使用类似表述：「根据知识库内容可概括为」；尽量综合多个相关文件，
  不要仅依据单一片段作答。
- 若检索结果提供了来源，可在回答内引用；目录清单会由系统统一附在回答中，
  不要自行重复罗列或虚构参考路径。"""

REACT_SYSTEM_PROMPT = _PROMPT_HEAD + render_tool_prompt_section() + _PROMPT_TAIL


def _agent_config() -> Dict[str, Any]:
    """读取 Agent 编排层配置；配置不可用时回退到安全默认值。"""
    try:
        return load_config()
    except Exception as e:
        debug_log(f"[AGENT] 配置读取失败，使用默认值: {e!r}")
        return {}


_REPEAT_ACTION_HINT = (
    "你刚刚用完全相同的参数调用过同一工具，重复调用不会带来新信息。"
    "请换一个检索角度：改用更窄的「设备名+子目录名」或更宽的「仅小组名」关键词，"
    "用命中的具体文件/设备名或用户原始问题调用 rag_qa，或逐个检索尚未覆盖的实体；"
    "若信息已足够，直接输出 Answer 并说明哪一部分未检索到依据。"
)

# 工具执行失败（超时/异常/未知工具）：注入恢复提示，不计入有效依据，继续换角度重试
_RECOVERY_HINT = (
    "上一步工具执行未成功（可能超时或出错），该结果不可作为依据。"
    "请换一个检索角度或更换工具：目录枚举未命中时改用语正文检索 rag_qa，"
    "正文检索异常时改用语目录枚举或换更具体的关键词；不要基于失败结果作答。"
    "若多次尝试仍失败且信息不足，直接说明未在知识库中检索到依据。"
)

# 正文检索到但覆盖度低/未命中（none/very_low 或空结果）：提示换角度重试，不标记有效依据
_NO_EVIDENCE_HINT = (
    "本次正文检索未获得充分依据（覆盖度低或未命中相关文档），该结果不可作为作答依据。"
    "请换更具体的关键词、设备型号、日期戳或实体重试，或改用 list_group_files 枚举目录；"
    "不要基于无依据结果作答。若已多次换角度仍无依据，直接说明未检索到。"
)


def _compact_observation(text: str, cfg: dict) -> str:
    """有界裁剪「注入 LLM 上下文」的工具结果，保留首尾并标注省略量。

    只压缩喂给模型的 messages；SSE 的 step_tool.observation 仍传完整原文。
    判据是字符数，与具体内容无关。
    """
    text = str(text or "")
    limit = max(0, int((cfg or {}).get("AGENT_OBSERVATION_MAX_CHARS", 2000)))
    if limit <= 0 or len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - head - tail
    return (f"{text[:head]}\n…（共 {len(text)} 字符，中间省略 {omitted} 字符；"
            f"完整结果已在工具事件中展示）\n{text[-tail:] if tail else ''}")


def _compact_agent_history(messages: List[Dict[str, str]],
                           observation_indices: List[int], cfg: dict) -> None:
    """只保留最近 N 轮完整工具结果，更早的 Observation 折叠为占位文本。"""
    keep = max(0, int((cfg or {}).get("AGENT_KEEP_RECENT_OBSERVATIONS", 2)))
    stale = observation_indices[:-keep] if keep > 0 else list(observation_indices)
    for idx in stale:
        if 0 <= idx < len(messages):
            content = str(messages[idx].get("content", ""))
            if not content.startswith("[已折叠]"):
                messages[idx]["content"] = (
                    "[已折叠] 早期工具结果已用于推理；如需细节请用更具体的关键词重新检索。")


def _agent_fallback_answer(question: str, tool_log: list,
                           last_observation: str = "", cfg: dict = None) -> str:
    """预算耗尽时诚实收敛：列出检索角度 + 已获得的部分依据，不编造、不回显长原文。"""
    lines = ["根据知识库检索结果，暂未能形成完整回答。", "", "已尝试的检索角度："]
    for tool, arg in (tool_log or [])[:8]:
        lines.append(f"- {tool}[{str(arg)[:60]}]")
    if not tool_log:
        lines.append("- （尚未成功执行工具）")
    snippet_limit = max(0, int((cfg or {}).get("AGENT_FALLBACK_SNIPPET_CHARS", 300)))
    snippet = str(last_observation or "").strip()
    if snippet_limit and snippet:
        if len(snippet) > snippet_limit:
            snippet = snippet[:snippet_limit] + "…"
        lines += ["", "已获得的部分依据：", snippet]
    lines += ["", f"未在知识库中检索到充分依据的部分：{str(question)[:120]}",
              "建议缩小范围，或改用更具体的设备名、日期、文档名再次提问。"]
    return "\n".join(lines)


def parse_action(text: str) -> Dict[str, Any]:
    """从模型输出解析 Action 行 `Action: tool_name["arg"]`，未解析到返回 {}。

    按输出格式契约每轮只有一个 Action；取最后一个 Action 行以兼容模型附带的解释文本。
    """
    action_line = ""
    for line in reversed(text.strip().splitlines()):
        if line.strip().startswith("Action:"):
            action_line = line.strip()
            break
    if not action_line:
        return {}
    m = re.match(r"Action:\s*([a-zA-Z0-9_]+)\s*\[(.*)\]\s*$", action_line)
    if not m:
        return {}
    tool_name = m.group(1)
    raw_arg = m.group(2).strip()
    if (raw_arg.startswith('"') and raw_arg.endswith('"')) or (
        raw_arg.startswith("'") and raw_arg.endswith("'")
    ):
        raw_arg = raw_arg[1:-1]
    return {"tool_name": tool_name, "arg": raw_arg}


def parse_answer(text: str) -> str:
    """解析最终 Answer；没有显式 Answer: 时返回全文。"""
    match = re.search(r"(?:^|\n)\s*Answer:\s*([\s\S]*)$", text.strip())
    return match.group(1).strip() if match else text.strip()


def _fold_turn_agent(turn: Dict[str, str]) -> Dict[str, str]:
    """Agent 侧历史折叠：超字符预算的最旧轮次压成紧凑占位，保留意图线索。"""
    role = str(turn.get("role", ""))
    content = str(turn.get("content", "")).strip()
    if not content:
        return {"role": role, "content": ""}
    if role == "user":
        if len(content) <= 50:
            return {"role": role, "content": f"用户：{content}"}
        return {"role": role,
                "content": f"[早期历史压缩] 用户曾问：{content[:50]}（…共 {len(content)} 字）"}
    if len(content) <= 100:
        return {"role": role, "content": f"助手：{content}"}
    label = "助手曾答" if role == "assistant" else role
    return {"role": role,
            "content": f"[早期历史压缩] {label}：{content[:100]}（…共 {len(content)} 字）"}


def _trim_agent_context(messages: List[Dict[str, str]], budget: int,
                        protected_indices: set) -> None:
    """messages 总字符超 budget 时，从最旧的非系统/非受保护消息开始折叠。

    对抗「RAG 结果爆炸」：多轮工具结果累积超出模型窗口时，把早期上下文压成
    占位，而非任由超长上下文导致截断或报错；受保护索引（最近若干 observation）不折叠。
    """
    if budget <= 0:
        return
    total = sum(len(m.get("content", "")) for m in messages)
    if total <= budget:
        return
    for i, m in enumerate(messages):
        if i == 0 or i in protected_indices:
            continue
        if total <= budget:
            break
        c = m.get("content", "")
        if c.startswith("[已折叠]") or c.startswith("[早期历史压缩]"):
            continue
        new_c = ("[已折叠] 早期上下文已压缩（原 %d 字符），如需细节请换更具体的"
                 "关键词重新检索。" % len(c))
        total -= (len(c) - len(new_c))
        m["content"] = new_c


def _initial_messages(question: str, chat_history: Optional[List[Dict[str, str]]] = None,
                      cfg: dict = None):
    """构建 Agent 初始消息：注入对话历史（受轮次与字符预算约束）。

    历史超 AGENT_HISTORY_MAX_CHARS 预算时，最旧的轮次折叠为紧凑占位而非丢弃，
    支持更长对话上下文而不撑爆模型窗口；AGENT_HISTORY_MAX_TURNS 为总轮次护栏。
    """
    cfg = cfg or {}
    messages: List[Dict[str, str]] = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
    max_turns = max(1, int(cfg.get("AGENT_HISTORY_MAX_TURNS", 20)))
    max_chars = max(0, int(cfg.get("AGENT_HISTORY_MAX_CHARS", 3000)))
    turns = [it for it in (chat_history or [])[-max_turns:]
             if str(it.get("role", "")) in {"user", "assistant"}
             and str(it.get("content", "")).strip()]
    kept, used, folded = [], 0, []
    for it in reversed(turns):
        c = str(it.get("content", ""))
        if max_chars and kept and used + len(c) > max_chars:
            folded.insert(0, _fold_turn_agent(it))
        else:
            kept.insert(0, {"role": str(it.get("role")), "content": c})
            used += len(c)
    for m in folded + kept:
        if m.get("content"):
            messages.append(m)
    messages.append({"role": "user", "content": question})
    return messages


def ask_agent_stream(
    question: str,
    username: str = "legacy-agent",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Generator[dict, None, None]:
    """Agent 模式唯一编排器：产出统一事件流。

    事件：route_direct（确定性路由决策，非模型输出）、step_llm（模型原文）、
    step_tool（工具调用 + 完整 observation + tool_call_id）、tool（目录结果）、
    final（含 agent_metrics）、error。协议见 AGENTS.md 第 7 节。
    """
    # 构建 LLM 实例（供路由与 ReAct 循环共用）；agent 模式强制走大模型路由（route_query force_llm），
    # 由模型决定目录/正文/混合路径；仅在 LLM 路由完全失败时才回退到规则路由。
    llm = build_qwen_llm()
    initial_route = route_query(llm, question, force_llm=True)
    requires_rag_after_list = bool(initial_route and initial_route.get("route") == "hybrid")

    cfg = _agent_config()
    guard_repeat = bool(cfg.get("AGENT_REPEAT_ACTION_GUARD", True))
    total_timeout = max(1.0, float(cfg.get("AGENT_TOTAL_TIMEOUT_S", 240)))
    # 多步连锁场景优化：循环内 rag_qa 子问题复用内容路由，跳过冗余规则/LLM 重路由。
    # 模型已显式选择 rag_qa（内容工具），语义等价于内容路由；该 hint 对 list_group_files 无副作用。
    reuse_route_hint = bool(cfg.get("AGENT_REUSE_ROUTE_HINT", True))
    tool_route_hint = {"route": "rag_search"} if reuse_route_hint else None
    seen_actions = set()
    latest_catalog_result = None
    used_list_tool = False
    used_rag_after_list = False
    last_observation = ""
    # 依据记账：是否获得过正文/目录有效依据，用于最终作答前的「无依据诚实收敛」守卫
    content_grounded = False
    catalog_grounded = False

    def _grounded_for_answer() -> bool:
        """是否已有作答依据：正文依据，或目录型问题已获得目录枚举结果。"""
        if content_grounded:
            return True
        if (initial_route and initial_route.get("route") == "file_list") and catalog_grounded:
            return True
        return False

    def finalize_answer(value: str) -> str:
        answer = sanitize_generated_answer(value)
        if latest_catalog_result and "## 目录 / 文件检索结果" not in answer:
            answer = f"{build_catalog_answer_section(latest_catalog_result)}\n\n## 文档内容说明\n\n{answer}".strip()
        return answer

    try:
        # ---- 确定性直调：内容型问题不进入 LLM 循环（README.md 的 Agent 边界契约）----
        if initial_route and initial_route.get("route") == "rag_search":
            # 不再伪造 step_llm / Thought：明确发出编排器决策事件
            yield {"type": "route_direct", "step": 1, "tool": CONTENT_TOOL_NAME,
                   "arg": question, "route": "rag_search",
                   "reason": "内容/格式/规范型问题直接检索文档正文"}
            result = dispatch_tool(CONTENT_TOOL_NAME, question, username,
                                   route_hint=initial_route, tool_call_id="tc-1")
            observation_text = result["observation"]
            last_observation = observation_text
            latest_catalog_result = result["catalog_result"]
            yield {"type": "step_tool", "step": 1, "tool": CONTENT_TOOL_NAME, "arg": question,
                   "observation": observation_text, "tool_call_id": "tc-1",
                   "deterministic": True, "elapsed_s": result["elapsed_s"]}
            if latest_catalog_result:
                yield {"type": "tool", "tool_name": "list_catalog_entries",
                       "content": latest_catalog_result}
            final_answer = finalize_answer(observation_text)
            append_user_chat_history(username, "user", question)
            append_user_chat_history(username, "assistant", final_answer)
            # rag_result.get("file_result") 由 agent_tools 的 handler 统一提供
            yield {"type": "final", "content": final_answer,
                   "file_result": latest_catalog_result,
                   "agent_metrics": {"llm_calls": 0, "tool_calls": 1, "idle_steps": 0,
                                     "context_chars_peak": 0, "converged": True,
                                     "content_grounded": True, "catalog_grounded": False,
                                     "elapsed_s": 0.0, "deterministic_route": True}}
            return

        # ---- ReAct 循环 ----
        messages = _initial_messages(question, chat_history, cfg)
        max_steps = max(1, int(cfg.get("AGENT_MAX_STEPS", 6)))
        max_idle = max(0, int(cfg.get("AGENT_MAX_IDLE_STEPS", 2)))
        llm_budget = max_steps + max_idle + 1
        observation_indices: List[int] = []
        tool_log = []
        tool_calls = idle_steps = llm_calls = 0
        context_chars_peak = 0
        timed_out = False
        started = time.perf_counter()

        def append_observation(raw: str, hint: str) -> None:
            compact = _compact_observation(raw, cfg)
            messages.append({"role": "user", "content": f"Observation: {compact}\n{hint}"})
            observation_indices.append(len(messages) - 1)
            _compact_agent_history(messages, observation_indices, cfg)

        def agent_metrics(converged: bool) -> Dict[str, Any]:
            return {
                "llm_calls": llm_calls, "tool_calls": tool_calls,
                "idle_steps": idle_steps, "context_chars_peak": context_chars_peak,
                "converged": converged, "timed_out": timed_out,
                "content_grounded": content_grounded, "catalog_grounded": catalog_grounded,
                "elapsed_s": round(time.perf_counter() - started, 3),
            }

        def emit_tool_step(step: int, name: str, arg: str, result: dict) -> Generator[dict, None, None]:
            yield {"type": "step_tool", "step": step, "tool": name, "arg": arg,
                   "observation": result["observation"],
                   "tool_call_id": result["tool_call_id"],
                   "elapsed_s": result["elapsed_s"], "ok": result["ok"]}
            if result["catalog_result"]:
                yield {"type": "tool", "tool_name": "list_catalog_entries",
                       "content": result["catalog_result"]}

        # 步数预算只计有效工具调用；空转受 max_idle、llm_budget 与总耗时预算保护
        while tool_calls < max_steps and llm_calls < llm_budget:
            if time.perf_counter() - started > total_timeout:
                timed_out = True
                debug_log("[AGENT] 总耗时预算耗尽，提前收敛")
                break

            llm_calls += 1
            step = llm_calls
            context_chars_peak = max(
                context_chars_peak, sum(len(m.get("content", "")) for m in messages))
            # 总上下文预算护栏：多轮 RAG 结果 / 长历史累积超长时折叠最旧消息，防爆炸
            context_budget = max(0, int(cfg.get("AGENT_CONTEXT_MAX_CHARS", 16000)))
            keep_obs = max(0, int(cfg.get("AGENT_KEEP_RECENT_OBSERVATIONS", 2)))
            protected = set(observation_indices[-keep_obs:]) if observation_indices else set()
            _trim_agent_context(messages, context_budget, protected)

            llm_start = time.perf_counter()
            try:
                resp = llm.invoke(messages)
                content = resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                debug_log(f"[AGENT] LLM 调用失败: {e!r}")
                yield {"type": "error", "message": "模型服务暂时不可用，请稍后重试。"}
                return
            debug_log(f"[AGENT] step={step} LLM耗时={time.perf_counter()-llm_start:.3f}s")
            yield {"type": "step_llm", "step": step, "content": content}

            # 1) 最终 Answer（hybrid 清单+内容问题必须先补读正文）
            if "Answer:" in content:
                if requires_rag_after_list and used_list_tool and not used_rag_after_list \
                        and (CONTENT_TOOL_NAME, question) not in seen_actions:
                    messages.append({"role": "assistant", "content": content})
                    result = dispatch_tool(CONTENT_TOOL_NAME, question, username,
                                           route_hint=tool_route_hint,
                                           tool_call_id=f"tc-{tool_calls + 1}")
                    tool_calls += 1
                    tool_log.append((CONTENT_TOOL_NAME, question))
                    seen_actions.add((CONTENT_TOOL_NAME, question))
                    used_rag_after_list = True
                    last_observation = result["observation"]
                    # 与 step-4 一致的记账：有效正文结果记为依据，避免最终作答被误判无依据
                    res_cov = result.get("coverage_level")
                    if result.get("ok") and res_cov not in ("none", "very_low") \
                            and (result["observation"] or "").strip():
                        content_grounded = True
                    if result["catalog_result"]:
                        latest_catalog_result = result["catalog_result"]
                    yield from emit_tool_step(step, CONTENT_TOOL_NAME, question, result)
                    append_observation(result["observation"], "请基于目录结果与正文给出 Answer。")
                    continue

                # 无依据却要作答：拒绝凭空作答，转入诚实收敛
                if tool_calls > 0 and not _grounded_for_answer():
                    final_answer = finalize_answer(
                        _agent_fallback_answer(question, tool_log, last_observation, cfg))
                    append_user_chat_history(username, "user", question)
                    append_user_chat_history(username, "assistant", final_answer)
                    yield {"type": "final", "content": final_answer,
                           "file_result": latest_catalog_result,
                           "agent_metrics": agent_metrics(False)}
                    return

                final_answer = finalize_answer(parse_answer(content))
                append_user_chat_history(username, "user", question)
                append_user_chat_history(username, "assistant", final_answer)
                yield {"type": "final", "content": final_answer,
                       "file_result": latest_catalog_result,
                       "agent_metrics": agent_metrics(True)}
                return

            # 2) 解析 Action；未知工具名交由 dispatch_tool 统一降级
            action = parse_action(content)
            if not action:
                final_answer = finalize_answer(content)
                append_user_chat_history(username, "user", question)
                append_user_chat_history(username, "assistant", final_answer)
                yield {"type": "final", "content": final_answer,
                       "file_result": latest_catalog_result,
                       "agent_metrics": agent_metrics(True)}
                return

            name, arg = action["tool_name"], action["arg"]
            action_key = (name, arg)

            # 3) 重复调用抑制：算空转，不占工具预算
            if guard_repeat and action_key in seen_actions:
                if idle_steps >= max_idle:
                    debug_log("[AGENT] 空转预算耗尽，提前收敛")
                    break
                idle_steps += 1
                messages.append({"role": "assistant", "content": content})
                append_observation(_REPEAT_ACTION_HINT, "请换一个检索角度或给出 Answer。")
                continue
            seen_actions.add(action_key)

            # 4) 统一分派执行
            result = dispatch_tool(name, arg, username,
                                   route_hint=tool_route_hint,
                                   tool_call_id=f"tc-{tool_calls + 1}")
            tool_calls += 1
            tool_log.append((name, arg))
            last_observation = result["observation"]

            # 工具执行失败：注入恢复提示、不计入有效依据，继续换角度重试
            if not result.get("ok"):
                yield from emit_tool_step(step, name, arg, result)
                append_observation(result["observation"], _RECOVERY_HINT)
                continue

            role = tool_role(name)
            if role == "catalog":
                used_list_tool = True
                # 目录枚举返回了非空结果（无论 catalog_result 是否为结构化对象）即可作为作答依据
                if result["catalog_result"] or (result["observation"] or "").strip():
                    catalog_grounded = True
                    if result["catalog_result"]:
                        latest_catalog_result = result["catalog_result"]
            elif role == "content":
                if used_list_tool:
                    used_rag_after_list = True
                # 工具执行失败（ok=False，如 read_file 路径未命中）：视为无依据，提示换角度重试
                if not result.get("ok"):
                    if result["catalog_result"]:
                        latest_catalog_result = result["catalog_result"]
                    yield from emit_tool_step(step, name, arg, result)
                    append_observation(result["observation"], _RECOVERY_HINT)
                    continue
                cov = result.get("coverage_level")
                # 检索无命中（覆盖度 none）或返回空：提示换角度重试，不标记依据；
                # 仅 none 触发（very_low 仍属有命中，交由底层可答性闸门与 LLM 判断，避免过度收敛）
                if cov == "none" or not (result["observation"] or "").strip():
                    if result["catalog_result"]:
                        latest_catalog_result = result["catalog_result"]
                    yield from emit_tool_step(step, name, arg, result)
                    append_observation(result["observation"], _NO_EVIDENCE_HINT)
                    continue
                content_grounded = True
                if result["catalog_result"]:
                    latest_catalog_result = result["catalog_result"]
            else:
                # 未知工具：dispatch 已降级为错误观察，复用其结果
                pass

            yield from emit_tool_step(step, name, arg, result)
            append_observation(
                result["observation"],
                "请继续思考并决定下一步：信息尚不足则继续调用工具（换检索角度）；"
                "信息已足够且复合意图已在目录基础上读过正文，则给出 Answer。")

        # 预算用尽后给一次收敛轮（不执行新工具）
        if (tool_log and llm_calls < llm_budget and not timed_out
                and bool(cfg.get("AGENT_FINAL_CONVERGE_ROUND", True))):
            llm_calls += 1
            step = llm_calls
            context_chars_peak = max(
                context_chars_peak, sum(len(m.get("content", "")) for m in messages))
            try:
                resp = llm.invoke(messages + [{
                    "role": "user",
                    "content": ("工具调用预算已用尽，请不要再发起新的 Action；"
                                "请基于上面已经获得的 Observation 直接输出 Answer，"
                                "并明确说明哪一部分未检索到依据。"),
                }])
                content = resp.content if hasattr(resp, "content") else str(resp)
                yield {"type": "step_llm", "step": step, "content": content}
                if "Answer:" in content:
                    # 无依据：诚实收敛而非回显无依据作答
                    if tool_calls > 0 and not _grounded_for_answer():
                        final_answer = finalize_answer(
                            _agent_fallback_answer(question, tool_log, last_observation, cfg))
                        append_user_chat_history(username, "user", question)
                        append_user_chat_history(username, "assistant", final_answer)
                        yield {"type": "final", "content": final_answer,
                               "file_result": latest_catalog_result,
                               "agent_metrics": agent_metrics(False)}
                        return
                    final_answer = finalize_answer(parse_answer(content))
                    append_user_chat_history(username, "user", question)
                    append_user_chat_history(username, "assistant", final_answer)
                    yield {"type": "final", "content": final_answer,
                           "file_result": latest_catalog_result,
                           "agent_metrics": agent_metrics(True)}
                    return
            except Exception as e:
                debug_log(f"[AGENT] 收敛轮失败，回退诚实说明: {e!r}")

        final_answer = finalize_answer(
            _agent_fallback_answer(question, tool_log, last_observation, cfg))
        append_user_chat_history(username, "user", question)
        append_user_chat_history(username, "assistant", final_answer)
        yield {"type": "final", "content": final_answer,
               "file_result": latest_catalog_result,
               "agent_metrics": agent_metrics(False)}

    except Exception as e:
        _logger.error(f"ReAct 执行异常: {e!r}")
        _logger.debug(traceback.format_exc())
        yield {"type": "error", "message": "服务暂时不可用，请稍后重试。"}


def ask_agent(
    question: str,
    username: str = "legacy-agent",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """非流式薄适配器：消费 ask_agent_stream 的同一事件流，不重复实现循环。"""
    for event in ask_agent_stream(question, username=username, chat_history=chat_history):
        if event.get("type") == "final":
            return event.get("content", "")
        if event.get("type") == "error":
            return event.get("message", "服务暂时不可用，请稍后重试。")
    return "服务暂时不可用，请稍后重试。"
