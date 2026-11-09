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
    estimate_tokens,
    question_has_content_intent,
    question_is_compound_intent,
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
- 已通过 list_group_files 得到目录结果、而用户要的是详细用法/步骤/参数时，
  **直接把结果里的 rel_path（设备目录或已定位目录）交给 read_file 精读**——目录路径本身
  就是合法参数，无需再找具体文件名；read_file 会读取该目录下全部手册，比 rag_qa 全局
  检索更完整、更聚焦。仅当 read_file 未命中、或需要跨多文档综合、或用户未指明具体
  文档时才用 rag_qa。
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

# 小上下文窗口（如单卡 3090 的 VLLM_MAX_MODEL_LEN=4096）下，完整提示词本身就会挤爆输入预算，
# 此时改用精简版：保留工具清单、输出格式与核心约束，去掉知识库结构与链式规则的长篇说明。
_PROMPT_HEAD_COMPACT = """你是实验室内部知识库助手，采用 ReAct 思维链 + 工具调用方式工作。
检索结果带【文档/路径/章节/段落摘要】前缀；检索词保留原文中的小组名、设备型号、日期戳。"""

_PROMPT_TAIL_COMPACT = """

## 输出格式（ReAct）
- 每轮先 Thought，再输出一个 Action 行：Action: 工具名[参数值]；信息足够时输出 Answer: 最终回答。
- 工具结果以 Observation: 开头；Observation 过长会被截断，如需细节请换更聚焦的关键词重新检索。

## 规则
- 回答必须依据工具返回，不得凭空编造；未获得依据时明确说明未检索到。
- 同时要求清单与内容说明：列目录后必须再调用内容工具（read_file 或 rag_qa）读正文。
- 已得到目录 rel_path 且用户要详细用法/步骤/参数时，直接把该目录路径交给 read_file 精读
  （目录路径是合法参数）；read_file 失败或需跨文档综合时才用 rag_qa。
- 多实体/链式任务要分别检索后综合；换检索角度，禁止用完全相同的参数重复调用同一工具。
- 总结时用「根据知识库内容可概括为」；目录清单由系统统一附在回答中，不要虚构参考路径。"""

REACT_SYSTEM_PROMPT = _PROMPT_HEAD + render_tool_prompt_section() + _PROMPT_TAIL
REACT_SYSTEM_PROMPT_COMPACT = (_PROMPT_HEAD_COMPACT + render_tool_prompt_section()
                               + _PROMPT_TAIL_COMPACT)


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


def _agent_input_token_budget(cfg: dict) -> int:
    """计算 LLM 输入侧可用的 token 预算。

    vLLM 按「messages tokens + RESPONSE_MAX_TOKENS(输出预留)」校验窗口，故
    输入预算 = VLLM_MAX_MODEL_LEN - RESPONSE_MAX_TOKENS - 分词估算误差余量。
    显式配置 AGENT_CONTEXT_MAX_TOKENS > 0 时优先使用（便于人工收紧）。
    """
    cfg = cfg or {}
    explicit = int(cfg.get("AGENT_CONTEXT_MAX_TOKENS", 0) or 0)
    if explicit > 0:
        return max(512, explicit)
    # 仅在拿到真实模型窗口配置时启用 token 护栏：生产 cfg 来自 load_config（含
    # VLLM_MAX_MODEL_LEN）；离线 mock 只给精简 cfg，沿用字符护栏以免扭曲对照实验。
    has_window = any(k in cfg for k in ("VLLM_MAX_MODEL_LEN", "RESPONSE_MAX_TOKENS",
                                        "AGENT_CONTEXT_SAFETY_TOKENS"))
    if not has_window:
        return 0
    model_len = int(cfg.get("VLLM_MAX_MODEL_LEN", 16384) or 16384)
    response = int(cfg.get("RESPONSE_MAX_TOKENS", 2048) or 2048)
    safety = int(cfg.get("AGENT_CONTEXT_SAFETY_TOKENS", 256) or 0)
    return max(512, model_len - response - safety)


def _message_tokens(m: Dict[str, str]) -> int:
    """单条消息 token 估算：内容 + 角色/chat template 固定开销。"""
    return 4 + estimate_tokens(str(m.get("content", "")))


def _messages_tokens(messages: List[Dict[str, str]]) -> int:
    return sum(_message_tokens(m) for m in messages)


def _clip_to_tokens(text: str, max_tokens: int) -> str:
    """按 token 上限首尾保留地裁剪文本（字符级护栏的 token 级补充）。"""
    text = str(text or "")
    if max_tokens <= 0:
        return "[已折叠] 内容因模型窗口限制已省略。"
    est = estimate_tokens(text)
    if est <= max_tokens:
        return text
    # 按「估算 token / 实际字符」比例换算可保留字符数：中文 1 字可能对应 >1 token，
    # 直接以 max_tokens 当字符数裁剪会失效；估算偏差由安全余量吸收。
    keep = max(0, min(len(text), int(len(text) * max_tokens / max(1, est))))
    head = int(keep * 0.7)
    tail = keep - head
    omitted = len(text) - head - tail
    return (f"{text[:head]}\n…（共 {len(text)} 字符，为适配模型窗口省略中间 {omitted} 字符；"
            f"完整结果已在工具事件中展示）\n{text[-tail:] if tail else ''}")


def _fit_agent_context_tokens(messages: List[Dict[str, str]], budget: int,
                              protected_indices: set) -> None:
    """按 token 预算折叠/裁剪 messages，保证 messages + 输出预留不超模型窗口。

    顺序：折叠最旧的非系统/非受保护消息 → 仍超则裁剪受保护的最近工具结果 →
    最后兜底裁剪系统提示词（保留工具清单部分）。返回值无意义，直接原地修改。
    """
    if budget <= 0:
        return
    # 最后一条永远保留：它是当前问题或最新 Observation，折叠后模型将无事可做
    protected_indices = set(protected_indices or set())
    if messages:
        protected_indices.add(len(messages) - 1)

    def _over() -> int:
        return _messages_tokens(messages) - budget

    if _over() <= 0:
        return
    # 1) 折叠最旧的非系统、非受保护消息
    for i, m in enumerate(messages):
        if _over() <= 0:
            return
        if i == 0 or i in protected_indices:
            continue
        c = str(m.get("content", ""))
        if c.startswith("[已折叠]") or c.startswith("[早期历史压缩]"):
            continue
        new_c = ("[已折叠] 早期上下文已压缩以适配模型窗口（原 %d 字符），"
                 "如需细节请换更具体的关键词重新检索。" % len(c))
        m["content"] = new_c

    if _over() <= 0:
        return
    # 2) 裁剪受保护的最近工具结果（从最旧的一条开始）；省略标记本身也占 token，
    #    故留足余量并循环重算，保证裁剪后真正降到达标范围。
    for _ in range(3):
        if _over() <= 0:
            return
        for i in sorted(protected_indices):
            if _over() <= 0:
                return
            if not (0 <= i < len(messages)) or i == 0:
                continue
            m = messages[i]
            allowed = max(0, _message_tokens(m) - _over() - 128)
            m["content"] = _clip_to_tokens(m["content"], allowed)

    if _over() <= 0:
        return
    # 3) 最后兜底：系统提示词改用精简版（工具清单仍由注册表生成，保持不漂移）
    if messages and estimate_tokens(str(messages[0].get("content", ""))) > 600:
        messages[0]["content"] = REACT_SYSTEM_PROMPT_COMPACT


def _is_context_overflow(exc: Exception) -> bool:
    """识别 vLLM / OpenAI 的「上下文超长」400 错误，避免误报成服务不可用。"""
    text = f"{type(exc).__name__} {exc}".lower()
    markers = ("maximum context length", "context length", "context_length",
               "too many tokens", "reduce the length", "prompt is too long",
               "exceeds the maximum", "requested", "tokens in the messages")
    return any(k in text for k in markers)


def _initial_messages(question: str, chat_history: Optional[List[Dict[str, str]]] = None,
                      cfg: dict = None):
    """构建 Agent 初始消息：注入对话历史（受轮次、字符与 token 预算约束）。

    系统提示词若已占满输入预算（小窗口场景），改用精简版；历史超
    AGENT_HISTORY_MAX_CHARS 预算时最旧轮次折叠为紧凑占位，最后按 token
    预算兜底折叠，保证 messages + 输出预留不超过 VLLM_MAX_MODEL_LEN。
    """
    cfg = cfg or {}
    token_budget = _agent_input_token_budget(cfg)
    system_prompt = REACT_SYSTEM_PROMPT
    # 完整提示词约占 2.5k tokens；小窗口下它已吃满输入预算，换精简版给工具结果留空间
    if token_budget > 0 and _message_tokens({"content": system_prompt}) > token_budget * 0.85:
        system_prompt = REACT_SYSTEM_PROMPT_COMPACT
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
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
    # 初始消息按 token 预算兜底折叠（保护最后一条当前问题）
    _fit_agent_context_tokens(messages, token_budget, set())
    return messages


def ask_agent_stream(
    question: str,
    username: str = "legacy-agent",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> Generator[dict, None, None]:
    """Agent 模式唯一编排器：产出统一事件流。

    事件：route_direct（编排器 LLM 路由决策，非模型输出；其后仍进入 ReAct 循环）、step_llm（模型原文）、
    step_tool（工具调用 + 完整 observation + tool_call_id）、tool（目录结果）、
    final（含 agent_metrics）、error。协议见 AGENTS.md 第 7 节。
    """
    # 构建 LLM 实例（供路由与 ReAct 循环共用）；agent 模式强制走大模型路由（route_query force_llm），
    # 由模型决定目录/正文/混合路径；仅在 LLM 路由完全失败时才回退到规则路由。
    llm = build_qwen_llm()
    initial_route = route_query(llm, question, force_llm=True)
    # 复合意图守卫不依赖 LLM 路由标签：LLM 常把「有什么设备 怎么用」误判为纯 file_list，
    # 因此只要问题本身同时要求清单与内容说明，列目录后就必须补读正文。
    question_wants_content = question_has_content_intent(question)
    requires_rag_after_list = bool(
        (initial_route and initial_route.get("route") == "hybrid")
        or question_is_compound_intent(question))
    # 内容型路由：进入循环后仍须先由模型调用内容工具取得正文依据，不得凭空作答
    content_intent = bool(initial_route and initial_route.get("route") == "rag_search")

    cfg = _agent_config()
    guard_repeat = bool(cfg.get("AGENT_REPEAT_ACTION_GUARD", True))
    total_timeout = max(1.0, float(cfg.get("AGENT_TOTAL_TIMEOUT_S", 240)))
    # 多步连锁场景优化：循环内 rag_qa 子问题复用内容路由，跳过冗余规则/LLM 重路由。
    # 模型已显式选择 rag_qa（内容工具），语义等价于内容路由；该 hint 对 list_group_files 无副作用。
    reuse_route_hint = bool(cfg.get("AGENT_REUSE_ROUTE_HINT", True))
    # hint 必须自带 target/reason：ask_stream 复用路由时会读取这些键，
    # 只给 route 会触发 KeyError('target') 并让整轮工具调用失败。
    tool_route_hint = ({"route": "rag_search", "target": question,
                        "reason": "Agent 已选择内容工具，等价于内容路由"}
                       if reuse_route_hint else None)
    seen_actions = set()
    latest_catalog_result = None
    used_list_tool = False
    used_rag_after_list = False
    last_observation = ""
    # 依据记账：是否获得过正文/目录有效依据，用于最终作答前的「无依据诚实收敛」守卫
    content_grounded = False
    catalog_grounded = False

    def _grounded_for_answer() -> bool:
        """是否已有作答依据：正文依据，或目录型问题已获得目录枚举结果。

        问题本身要求内容说明时，仅有目录清单不足以作答（目录只回答「有哪些」），
        必须有正文依据，避免模型凭清单编造用法。
        """
        if content_grounded:
            return True
        if question_wants_content:
            return False
        if (initial_route and initial_route.get("route") == "file_list") and catalog_grounded:
            return True
        return False

    def _preferred_content_action():
        """复合意图强制补读时的工具选择：已定位目录则优先 read_file 精读，否则 rag_qa。

        list_group_files 返回的 matched_dir / directories[].rel_path 本身就是
        read_file 可直接使用的路径，无需再靠 rag_qa 全局检索去猜文档。
        """
        cat = latest_catalog_result
        if isinstance(cat, dict):
            rel = cat.get("matched_dir")
            if not rel:
                items = cat.get("directories") or cat.get("files") or []
                if items and isinstance(items[0], dict):
                    rel = items[0].get("rel_path")
            if rel and str(rel).strip():
                return "read_file", str(rel)
        return CONTENT_TOOL_NAME, question

    def finalize_answer(value: str) -> str:
        answer = sanitize_generated_answer(value)
        if latest_catalog_result and "## 目录 / 文件检索结果" not in answer:
            answer = f"{build_catalog_answer_section(latest_catalog_result)}\n\n## 文档内容说明\n\n{answer}".strip()
        return answer

    try:
        # 内容型路由（rag_search）：发出编排器决策事件（非模型输出），随后进入 ReAct 循环。
        # 不再短路直调：由模型在循环中自行决定调用 rag_qa / read_file（允许作答前再补精读、
        # 换角度检索），从而统一所有意图的循环范式；「必有正文依据」由下方 content-first 守卫保证。
        if content_intent:
            yield {"type": "route_direct", "step": 1, "tool": CONTENT_TOOL_NAME,
                   "arg": question, "route": "rag_search",
                   "reason": "内容/格式/规范型问题，进入循环由模型检索文档正文"}

        # ---- ReAct 循环 ----
        messages = _initial_messages(question, chat_history, cfg)
        # 内容型路由：在用户问题前注入路由倾向，让模型首轮即调用内容工具（减少无谓空转；仍可自主选择 read_file）
        if content_intent:
            for _m in reversed(messages):
                if _m.get("role") == "user":
                    _m["content"] = (f"（路由判断为内容检索型，请先调用 {CONTENT_TOOL_NAME} 或 read_file "
                                   f"检索知识库正文后再作答）\n{_m['content']}")
                    break
        max_steps = max(1, int(cfg.get("AGENT_MAX_STEPS", 6)))
        max_idle = max(0, int(cfg.get("AGENT_MAX_IDLE_STEPS", 2)))
        # content-first 守卫连续挡回的次数上限：模型反复凭空 Answer 时及时诚实收敛，
        # 避免把 LLM 预算全耗在无工具空转上。
        guard_hits = 0
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
            # 双层上下文护栏：字符预算粗剪 + token 预算精剪（与 vLLM 窗口对齐）。
            # token 护栏是防 400「maximum context length」的关键，字符预算无法反映中文分词。
            keep_obs = max(0, int(cfg.get("AGENT_KEEP_RECENT_OBSERVATIONS", 2)))
            protected = set(observation_indices[-keep_obs:]) if observation_indices else set()
            _trim_agent_context(messages, max(0, int(cfg.get("AGENT_CONTEXT_MAX_CHARS", 16000))),
                                protected)
            token_budget = _agent_input_token_budget(cfg)
            _fit_agent_context_tokens(messages, token_budget, protected)

            llm_start = time.perf_counter()
            try:
                resp = llm.invoke(messages)
                content = resp.content if hasattr(resp, "content") else str(resp)
            except Exception as e:
                if not _is_context_overflow(e):
                    debug_log(f"[AGENT] LLM 调用失败: {e!r}")
                    yield {"type": "error", "message": "模型服务暂时不可用，请稍后重试。"}
                    return
                # 上下文超长：估算与真实分词有偏差时，激进折叠后再重试一次，避免整轮丢弃
                debug_log(f"[AGENT] 上下文超长，紧急折叠后重试: {e!r}")
                _fit_agent_context_tokens(messages, max(512, int(token_budget * 0.7)), set())
                try:
                    resp = llm.invoke(messages)
                    content = resp.content if hasattr(resp, "content") else str(resp)
                except Exception as e2:
                    debug_log(f"[AGENT] 折叠重试仍失败: {e2!r}")
                    # 不再误报「服务不可用」：基于已获得依据诚实收敛
                    final_answer = finalize_answer(
                        _agent_fallback_answer(question, tool_log, last_observation, cfg))
                    append_user_chat_history(username, "user", question)
                    append_user_chat_history(username, "assistant", final_answer)
                    yield {"type": "final", "content": final_answer,
                           "file_result": latest_catalog_result,
                           "agent_metrics": agent_metrics(False)}
                    return
            debug_log(f"[AGENT] step={step} LLM耗时={time.perf_counter()-llm_start:.3f}s")
            yield {"type": "step_llm", "step": step, "content": content}

            action = parse_action(content)

            # content-first 守卫：内容型路由必须先由模型调用内容工具取得正文依据。
            # 覆盖 Answer 与不带前缀的自由文本，避免模型绕过检索凭空作答；一旦已调用过工具
            # （成功或失败），则交给后续覆盖度判定与「无依据诚实收敛」逻辑处理。
            if (content_intent or question_is_compound_intent(question)) and tool_calls == 0 \
                    and not content_grounded and not action:
                guard_hits += 1
                if guard_hits > max_idle + 1:
                    debug_log("[AGENT] 反复凭空作答，提前诚实收敛")
                    final_answer = finalize_answer(
                        _agent_fallback_answer(question, tool_log, last_observation, cfg))
                    append_user_chat_history(username, "user", question)
                    append_user_chat_history(username, "assistant", final_answer)
                    yield {"type": "final", "content": final_answer,
                           "file_result": latest_catalog_result,
                           "agent_metrics": agent_metrics(False)}
                    return
                messages.append({"role": "assistant", "content": content})
                append_observation(
                    "你尚未检索知识库正文，请先调用 rag_qa（若已通过 list_group_files 定位到具体文档也可调用 "
                    "read_file）获取依据后再作答；不要凭记忆或猜测作答。",
                    "请先调用内容工具检索知识库正文，再给出 Answer。")
                continue

            # 1) 最终 Answer（清单+内容说明的复合问题必须先补读正文）
            # 动态守卫：只要已列目录且问题要求内容说明，不论 LLM 路由判成什么，
            # 都强制补一次正文检索，避免「搜一轮 file 就直接 answer」凭清单编造用法。
            if "Answer:" in content:
                need_read_after_list = (
                    requires_rag_after_list or (used_list_tool and question_wants_content))
                if need_read_after_list and used_list_tool and not used_rag_after_list:
                    messages.append({"role": "assistant", "content": content})
                    # 已定位目录 → 先 read_file 精读；失败/无内容再回退 rag_qa 全局检索。
                    primary_name, primary_arg = _preferred_content_action()
                    attempts = [(primary_name, primary_arg)]
                    if primary_name != CONTENT_TOOL_NAME:
                        attempts.append((CONTENT_TOOL_NAME, question))
                    last_result = None
                    for name, arg in attempts:
                        if (name, arg) in seen_actions:
                            continue
                        result = dispatch_tool(name, arg, username,
                                               route_hint=tool_route_hint,
                                               tool_call_id=f"tc-{tool_calls + 1}")
                        tool_calls += 1
                        tool_log.append((name, arg))
                        seen_actions.add((name, arg))
                        last_result = result
                        last_observation = result["observation"]
                        res_cov = result.get("coverage_level")
                        valid = bool(result.get("ok") and res_cov not in ("none", "very_low")
                                     and (result["observation"] or "").strip())
                        if valid:
                            content_grounded = True
                        if result["catalog_result"]:
                            latest_catalog_result = result["catalog_result"]
                        yield from emit_tool_step(step, name, arg, result)
                        if valid:
                            break
                    used_rag_after_list = True
                    if last_result is not None:
                        append_observation(last_observation,
                                           "请基于目录结果与正文给出 Answer。")
                    continue

                # 无依据却要作答：拒绝凭空作答，转入诚实收敛。
                # content_intent 即使 tool_calls==0（模型反复 Answer 被 content-first 守卫挡回），
                # 也必须有正文依据才允许输出，避免预算耗尽后回显无依据内容。
                if (tool_calls > 0 or content_intent) and not _grounded_for_answer():
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
            if not action:
                # 自由文本也无依据时同样诚实收敛，避免绕过 Answer 分支的守卫
                if (tool_calls > 0 or content_intent) and not _grounded_for_answer():
                    final_answer = finalize_answer(
                        _agent_fallback_answer(question, tool_log, last_observation, cfg))
                    append_user_chat_history(username, "user", question)
                    append_user_chat_history(username, "assistant", final_answer)
                    yield {"type": "final", "content": final_answer,
                           "file_result": latest_catalog_result,
                           "agent_metrics": agent_metrics(False)}
                    return
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
            # 收敛轮追加提示后同样按 token 预算裁剪，避免预算耗尽时反而因超长报错
            converge_messages = list(messages) + [{
                "role": "user",
                "content": ("工具调用预算已用尽，请不要再发起新的 Action；"
                            "请基于上面已经获得的 Observation 直接输出 Answer，"
                            "并明确说明哪一部分未检索到依据。"),
            }]
            keep_obs = max(0, int(cfg.get("AGENT_KEEP_RECENT_OBSERVATIONS", 2)))
            converge_protected = set(observation_indices[-keep_obs:]) if observation_indices else set()
            _fit_agent_context_tokens(converge_messages, _agent_input_token_budget(cfg),
                                      converge_protected | {len(converge_messages) - 1})
            try:
                resp = llm.invoke(converge_messages)
                content = resp.content if hasattr(resp, "content") else str(resp)
                yield {"type": "step_llm", "step": step, "content": content}
                if "Answer:" in content:
                    # 无依据：诚实收敛而非回显无依据作答（content_intent 同样强制有正文依据）
                    if (tool_calls > 0 or content_intent) and not _grounded_for_answer():
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
