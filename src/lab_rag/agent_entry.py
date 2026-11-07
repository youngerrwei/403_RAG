# agent_entry.py：可选 ReAct 编排层，不持有独立模型实例。

import os
import json
import time
import traceback
import re
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .logger import get_logger
from .paths import PROJECT_ROOT
from .rag_agent import (
    append_user_chat_history,
    build_catalog_answer_section,
    build_file_context,
    get_runtime,
    list_catalog_entries,
    load_config,
    rule_based_route,
    sanitize_generated_answer,
)
from .rag_tool import ask_rag
from .tools import rag_qa, list_group_files, agent_tool_context


load_dotenv(PROJECT_ROOT / ".env")
_logger = get_logger("agent")


def debug_log(*args):
    _logger.debug(" ".join(str(arg) for arg in args))


def build_qwen_llm() -> ChatOpenAI:
    """兼容旧入口，返回主 RAG 运行时中的同一个 LLM 实例。"""
    return get_runtime()["llm"]


TOOLS = {
    "rag_qa": rag_qa,
    "list_group_files": list_group_files,
}


REACT_SYSTEM_PROMPT = """你是实验室内部知识库助手，采用 ReAct 思维链 + 工具调用方式工作。

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

## 你可以使用以下工具

1) rag_qa(query: str) -> str
   - 通用问答工具：基于知识库正文进行检索并回答问题。
   - 用于具体技术问题、操作步骤、名词定义、原理说明、格式/要求/规范/步骤/方法、内容总结与对比。
   - 可在拿到文件清单后，针对其中具体文件或设备构造更聚焦的问题再次调用。

2) list_group_files(keyword: str) -> str
   - 文件/目录搜索工具：按关键词列出相关文件信息。
   - 关键词可以是小组名、设备名或型号、文档类型、日期、章节名、`Manuals` 等子目录名。
   - 返回 file_name、doc_title、rel_path、chunks 数量，用于了解有哪些文档及其所在目录层级。
   - 仅用于用户明确要求列出或定位文件、目录、路径、设备清单的场景。

## 输出格式（ReAct）
- 回答前先 Thought，再决定是否 Action。
- 需要调用工具时，严格单独输出一行：Action: 工具名[参数值]
  例如：Action: list_group_files["VLC小组"]
  或：Action: rag_qa["根据知识库说明 HOLOEYE LETO-3 空间光调制器的主要参数和用途"]
- 工具返回以 "Observation:" 开头，你需要基于它继续 Thought/Action，或给出最终 Answer。
- 信息足够时输出：Answer: 你的最终回答内容

## 多轮检索决策规则（重要）
每次拿到 Observation 后，先判断信息是否**覆盖用户问题的每一个子问**：
- 只有列出/定位类问题：目录结果已给出完整清单即算充分。
- 同时要求清单与内容说明（如「有哪些设备及使用规范」）：目录结果只回答了清单部分，
  **必须再调用 rag_qa 读正文**，且第二轮问题要保留用户原始意图，不能换成无关的固定话术。
- 涉及多个小组、设备或文档的对比/分别说明：目录或正文只覆盖其中一个实体时，
  **必须分别检索其余实体**，再综合，不能凭单一文档作答。
- 正文检索结果与问题不相关、只命中同主题的其他文档、或目录未命中：信息仍不充分。

信息不充分时，继续调用工具，但必须**换一个检索角度**，禁止用完全相同的参数重复调用同一工具：
- 目录未命中 → 改用更窄的「设备名 + 子目录名」或更宽的「仅小组名」关键词再枚举一次；
- 目录命中但缺正文 → 用命中的具体设备/文件名或用户原始问题调用 rag_qa；
- 正文答非所问 → 用原始问题、型号、日期戳、章节名等更具辨识度的词重新构造 query；
- 跨实体问题 → 逐个实体分别检索。
工具调用有步数预算；若已换过角度仍无依据，或预算即将用尽，就基于已获得的依据给出
最完整的回答，并明确说明哪一部分未在知识库中检索到依据，不要反复重试同一角度。

## 重要约束
- 回答必须优先依据工具返回，不得凭空编造。
- 「格式是什么/怎么写/要求/规范/步骤/方法」等是在查询文档内容，必须直接调用 rag_qa；
  不能先调用 list_group_files 猜测模板文件。
- 目录工具未找到结果不代表正文没有答案；应使用用户原始问题再调用 rag_qa，
  禁止改为输出通用模板、通用规范或知识库外常识。
- 未获得知识库依据时，不得输出「通用规范」「通用模板」或虚构「主要参考」。
- 总结时请明确使用类似表述：「根据知识库内容可概括为」；尽量综合多个相关文件，
  不要仅依据单一片段作答。
- 回答末尾请给出主要参考的文档标题或路径（若 Observation 或 rag_qa 结果提供了）。"""


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


def parse_action(text: str) -> Dict[str, Any]:
    """
    从模型输出中解析 Action 行，形如：
    Action: tool_name["arg string"]
    返回: {"tool_name": ..., "arg": ...}
    若未解析到则返回 {}。
    """
    # 找到最后一行包含 "Action:" 的行
    lines = text.strip().splitlines()
    action_line = ""
    for line in reversed(lines):
        if line.strip().startswith("Action:"):
            action_line = line.strip()
            break

    if not action_line:
        return {}

    # 例：Action: list_group_files["VLC小组"]
    m = re.match(r"Action:\s*([a-zA-Z0-9_]+)\s*\[(.*)\]\s*$", action_line)
    if not m:
        return {}

    tool_name = m.group(1)
    raw_arg = m.group(2).strip()

    # 去掉首尾引号
    if (raw_arg.startswith('"') and raw_arg.endswith('"')) or (
        raw_arg.startswith("'") and raw_arg.endswith("'")
    ):
        raw_arg = raw_arg[1:-1]

    return {"tool_name": tool_name, "arg": raw_arg}


def parse_answer(text: str) -> str:
    """
    从模型输出中解析最终 Answer:
    形如: Answer: xxx
    如果没有显式 Answer: 则返回全文。
    """
    match = re.search(r"(?:^|\n)\s*Answer:\s*([\s\S]*)$", text.strip())
    if match:
        return match.group(1).strip()
    # 没有显式 Answer，则直接返回原文
    return text.strip()


def _initial_messages(question: str, chat_history: Optional[List[Dict[str, str]]] = None):
    messages: List[Dict[str, str]] = [{"role": "system", "content": REACT_SYSTEM_PROMPT}]
    for item in (chat_history or [])[-20:]:
        role = item.get("role")
        content = str(item.get("content", "")).strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})
    return messages


def run_react_once(
    llm: ChatOpenAI,
    question: str,
    max_steps: Optional[int] = None,
    debug: bool = True,
    username: str = "legacy-agent",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    手写一个简单 ReAct 循环：
    - 系统提示 + 用户问题 → LLM 输出 Thought/Action/Answer
    - 解析 Action，如果有工具调用则执行，并把 Observation 追加到对话中，再次让 LLM 推理
    - 最多 max_steps 次工具调用
    """
    messages = _initial_messages(question, chat_history)
    cfg = _agent_config()
    if max_steps is None:
        max_steps = max(1, int(cfg.get("AGENT_LEGACY_MAX_STEPS", 5)))
    guard_repeat = bool(cfg.get("AGENT_REPEAT_ACTION_GUARD", True))
    seen_actions = set()

    observation_text = ""
    for step in range(1, max_steps + 1):
        if debug:
            debug_log(f"ReAct Step {step}")

        # 调用模型
        if debug:
            debug_log(f"准备调用 LLM, step={step}, message_count={len(messages)}")
            total_chars = sum(len(m.get("content", "")) for m in messages)
            debug_log(f"messages 总字符数: {total_chars}")
            for i, m in enumerate(messages[-4:], 1):  # 只打印最后几条，避免刷屏
                debug_log(f"最近消息{i}: role={m['role']}, chars={len(m['content'])}")

        llm_start = time.perf_counter()
        resp = llm.invoke(messages)
        llm_cost = time.perf_counter() - llm_start

        content = resp.content if hasattr(resp, "content") else str(resp)

        if debug:
            debug_log(f"LLM 调用完成, step={step}, 耗时={llm_cost:.3f}s, 输出字符数={len(content)}")
            debug_log("LLM 输出：", content)

        # 尝试解析 Answer（如果模型已经给出最终回答）
        if "Answer:" in content:
            final_answer = parse_answer(content)
            if debug:
                debug_log("解析到最终 Answer，结束循环")
            return final_answer

        # 解析 Action
        action = parse_action(content)
        if not action:
            # 没有 Action，也没有 Answer，当成直接回答
            if debug:
                debug_log("未解析到 Action，直接将本次输出当作回答")
            return content.strip()

        tool_name = action["tool_name"]
        arg = action["arg"]

        if debug:
            debug_log(f"解析到 Action: tool={tool_name}, arg={arg}")

        action_key = (tool_name, arg)
        if guard_repeat and action_key in seen_actions:
            debug_log(f"[AGENT] 重复调用被抑制: tool={tool_name}, arg={arg}")
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": f"Observation: {_REPEAT_ACTION_HINT}"})
            continue
        seen_actions.add(action_key)

        tool = TOOLS.get(tool_name)
        if tool is None:
            observation_text = f"[工具错误] 未找到名为 {tool_name} 的工具。"
        else:
            tool_start = time.perf_counter()
            try:
                # langchain_core.tools.Tool 对象：用 .invoke 调用
                # 若你的版本不支持 .invoke，可以用 .run 或直接 .func 看实际类型
                tool_input_key = list(tool.args.keys())[0]  # 第一个参数名
                tool_input = {tool_input_key: arg}
                with agent_tool_context(username):
                    obs = tool.invoke(tool_input)
                tool_cost = time.perf_counter() - tool_start
                observation_text = str(obs)

                if debug:
                    debug_log(
                        f"工具调用完成: tool={tool_name}, 耗时={tool_cost:.3f}s, observation字符数={len(observation_text)}")
            except Exception as e:
                tool_cost = time.perf_counter() - tool_start
                observation_text = "[工具执行异常] 服务暂时不可用，请稍后重试。"
                if debug:
                    debug_log(f"工具调用异常: tool={tool_name}, 耗时={tool_cost:.3f}s, error={e}")

        if debug:
            debug_log("Observation:", observation_text)

        # 将本轮 LLM 输出 + Observation 追加到对话历史，再继续下一轮
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation: {observation_text}\n请继续推理，若可以则给出 Answer。"})


    # 超出 max_steps 仍无 Answer，则返回最后一次 observation + 提示
    return f"工具调用达到最大步数，最后一次观察结果为：\n{observation_text}"


def ask_agent(
    question: str,
    username: str = "legacy-agent",
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """
    应用侧统一入口：ReAct LLM + 工具。
    """
    route = rule_based_route(question)
    if route and route.get("route") == "rag_search":
        # 明确的内容型问题无需让 Agent 再猜工具，直接复用标准 RAG 链路。
        answer = ask_rag(question, username=username).get("answer", "")
    else:
        llm = build_qwen_llm()
        answer = run_react_once(
            llm,
            question,
            debug=True,
            username=username,
            chat_history=chat_history,
        )
    append_user_chat_history(username, "user", question)
    append_user_chat_history(username, "assistant", answer)
    return answer


def ask_agent_stream(
    question: str,
    username: str = "legacy-agent",
    chat_history: Optional[List[Dict[str, str]]] = None,
):
    """
    流式 ReAct Agent：
    - type=step_llm：每一步 LLM 的输出（Thought/Action 文本）
    - type=step_tool：每次工具调用及其Observation
    - type=final：最终 Answer 文本
    - type=error：错误信息
    """
    initial_route = rule_based_route(question)
    requires_rag_after_list = bool(initial_route and initial_route.get("route") == "hybrid")

    observation_text = ""
    used_list_tool = False
    used_rag_after_list = False
    latest_catalog_result = None
    cfg = _agent_config()
    guard_repeat = bool(cfg.get("AGENT_REPEAT_ACTION_GUARD", True))
    seen_actions = set()

    def finalize_answer(value: str) -> str:
        answer = sanitize_generated_answer(value)
        if latest_catalog_result and "## 目录 / 文件检索结果" not in answer:
            answer = f"{build_catalog_answer_section(latest_catalog_result)}\n\n## 文档内容说明\n\n{answer}".strip()
        return answer

    try:
        if initial_route and initial_route.get("route") == "rag_search":
            # 内容意图采用确定性工具选择，避免 Agent 把“格式怎么写”误当成模板文件枚举。
            thought = (
                "Thought: 该问题询问知识库中的格式、规范或方法，应直接检索文档正文。\n"
                f"Action: rag_qa[{json.dumps(question, ensure_ascii=False)}]"
            )
            yield {"type": "step_llm", "step": 1, "content": thought}
            rag_result = ask_rag(question, username=username)
            observation_text = rag_result.get("answer", "")
            latest_catalog_result = rag_result.get("file_result")
            yield {
                "type": "step_tool",
                "step": 1,
                "tool": "rag_qa",
                "arg": question,
                "observation": observation_text,
            }
            if latest_catalog_result:
                yield {
                    "type": "tool",
                    "tool_name": "list_catalog_entries",
                    "content": latest_catalog_result,
                }
            final_answer = finalize_answer(observation_text)
            append_user_chat_history(username, "user", question)
            append_user_chat_history(username, "assistant", final_answer)
            yield {
                "type": "final",
                "content": final_answer,
                "file_result": latest_catalog_result,
            }
            return

        llm = build_qwen_llm()
        messages = _initial_messages(question, chat_history)
        max_steps = max(1, int(cfg.get("AGENT_MAX_STEPS", 4)))

        for step in range(1, max_steps + 1):
            # 1) 调 LLM
            debug_log(
                f"[stream] step={step} 开始, messages={len(messages)}, total_chars={sum(len(m.get('content', '')) for m in messages)}")

            llm_start = time.perf_counter()
            resp = llm.invoke(messages)
            llm_cost = time.perf_counter() - llm_start

            content = resp.content if hasattr(resp, "content") else str(resp)
            debug_log(f"[stream] step={step} LLM耗时={llm_cost:.3f}s, 输出字符数={len(content)}")

            # 把本轮 LLM 输出先流出去
            yield {
                "type": "step_llm",
                "step": step,
                "content": content,
            }

            # 2) 判断是否是最终 Answer
            if "Answer:" in content:
                if requires_rag_after_list and used_list_tool and not used_rag_after_list:
                    # 拦截过早的 Answer，强制继续 rag_qa
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "原始问题同时要求清单和内容说明，但你还没有调用 rag_qa 阅读正文。"
                            f"请使用原始问题“{question}”及上面的目录结果构造准确检索问题并调用 rag_qa，"
                            "再根据 rag_qa 的 Observation 输出 Answer。"
                        )
                    })
                    continue

                final_answer = finalize_answer(parse_answer(content))
                append_user_chat_history(username, "user", question)
                append_user_chat_history(username, "assistant", final_answer)
                yield {
                    "type": "final",
                    "content": final_answer,
                    "file_result": latest_catalog_result,
                }
                return

            # 3) 解析 Action
            action = parse_action(content)
            if not action:
                # 没有 Action，也没有 Answer，当作直接回答
                final_answer = finalize_answer(content)
                append_user_chat_history(username, "user", question)
                append_user_chat_history(username, "assistant", final_answer)
                yield {
                    "type": "final",
                    "content": final_answer,
                    "file_result": latest_catalog_result,
                }
                return

            tool_name = action["tool_name"]
            arg = action["arg"]

            action_key = (tool_name, arg)
            if guard_repeat and action_key in seen_actions:
                debug_log(f"[AGENT][stream] 重复调用被抑制: tool={tool_name}, arg={arg}")
                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Observation: {_REPEAT_ACTION_HINT}"})
                continue
            seen_actions.add(action_key)

            tool = TOOLS.get(tool_name)
            catalog_result = None
            if tool is None:
                observation_text = f"[工具错误] 未找到名为 {tool_name} 的工具。"
            else:
                try:
                    tool_start = time.perf_counter()
                    if tool_name == "list_group_files":
                        catalog_result = list_catalog_entries(arg)
                        observation_text = build_file_context(catalog_result)
                    elif tool_name == "rag_qa":
                        rag_result = ask_rag(arg, username=username)
                        observation_text = rag_result.get("answer", "")
                        catalog_result = rag_result.get("file_result")
                    else:
                        tool_input_key = list(tool.args.keys())[0]
                        tool_input = {tool_input_key: arg}
                        with agent_tool_context(username):
                            obs = tool.invoke(tool_input)
                        observation_text = str(obs)
                    tool_cost = time.perf_counter() - tool_start
                    debug_log(
                        f"[stream] step={step} tool={tool_name} 耗时={tool_cost:.3f}s, observation字符数={len(observation_text)}")

                    if tool_name == "list_group_files":
                        used_list_tool = True
                    if used_list_tool and tool_name == "rag_qa":
                        used_rag_after_list = True
                    if catalog_result:
                        latest_catalog_result = catalog_result

                except Exception as e:
                    _logger.error(f"Agent 工具执行异常: tool={tool_name}, error={e!r}")
                    observation_text = "[工具执行异常] 服务暂时不可用，请稍后重试。"

            # 4) 把本次工具调用结果也流出去
            yield {
                "type": "step_tool",
                "step": step,
                "tool": tool_name,
                "arg": arg,
                "observation": observation_text,
            }
            if catalog_result:
                yield {
                    "type": "tool",
                    "tool_name": "list_catalog_entries",
                    "content": catalog_result,
                }

            # 5) 历史追加，进入下一轮
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"Observation: {observation_text}\n"
                    "请继续思考并决定下一步：如果信息尚不足以回答，继续选择合适的工具调用；"
                    "如果信息已经足够，并且（仅当原始问题同时要求清单和内容说明时）已在目录结果基础上调用过 rag_qa，"
                    "则可以给出 Answer。"
                )
            })

        # 超出最大步数未得到 Answer
        final_answer = finalize_answer(f"工具调用达到最大步数，最后一次观察结果为：\n{observation_text}")
        append_user_chat_history(username, "user", question)
        append_user_chat_history(username, "assistant", final_answer)
        yield {
            "type": "final",
            "content": final_answer,
            "file_result": latest_catalog_result,
        }

    except Exception as e:
        _logger.error(f"ReAct 执行异常: {e!r}")
        _logger.debug(traceback.format_exc())
        yield {
            "type": "error",
            "content": "服务暂时不可用，请稍后重试。",
        }


if __name__ == "__main__":
    q = input("请输入问题（例如：VLC小组有哪些设备？）：").strip()
    if not q:
        q = "VLC小组有哪些设备？"
    ans = ask_agent(q)
    _logger.info("=== Agent 最终回答 ===")
    _logger.info(ans)
