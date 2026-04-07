# agent_entry.py

import os
import json
import time
import traceback
from datetime import datetime
import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tools import rag_qa, list_group_files


load_dotenv()


def debug_log(*args):
    now = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{now}]", *args, flush=True)


def build_qwen_llm() -> ChatOpenAI:
    base_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.getenv("VLLM_API_KEY", "EMPTY")
    model_name = os.getenv("VLLM_MODEL_NAME", "qwen-7b-chat")

    llm = ChatOpenAI(
        model=model_name,
        openai_api_base=base_url,
        openai_api_key=api_key,
        temperature=0.1,
        max_tokens=1024,
        timeout=120,
    )
    return llm


TOOLS = {
    "rag_qa": rag_qa,
    "list_group_files": list_group_files,
}


REACT_SYSTEM_PROMPT = """你是实验室内部知识库助手，采用 ReAct 思维链 + 工具调用方式工作。

你可以使用以下工具：

1) rag_qa(query: str) -> str
   - 通用问答工具：基于实验室知识库进行 RAG 检索并回答问题。
   - 当用户提出具体技术问题、操作步骤、名词定义、原理说明、设备用法等时，应优先调用此工具。
   - 你也可以在拿到某些文件列表后，构造针对这些文件的具体问题，并用 rag_qa 进一步深入阅读和总结。

2) list_group_files(keyword: str) -> str
   - 通用“文件/目录搜索”工具：根据任意关键词列出相关的文件信息。
   - 关键词可以是：小组名称（如“VLC小组”）、设备/仪器名称（如“LETO-3”）、项目名称、实验名称、文档类型（如“设备操作指南”）等。
   - 返回内容中包含 file_name、doc_title、rel_path、chunks 数量，可用于了解有哪些相关文档、它们大致位于哪个目录/小组/项目下。
   - 当用户的问题属于“有哪些文档/有哪些设备/有哪些资料/某个主题下有哪些文件”等清单或目录类查询时，应优先调用本工具。

使用规范（ReAct 格式）：
- 你需要在回答前先“思考”(Thought)，再决定是否调用工具(Action)。
- 如果需要调用工具，请严格按照以下格式单独输出一行：
  Action: 工具名[参数值]

  例如：
  Action: list_group_files["VLC小组"]
  或：
  Action: rag_qa["根据知识库说明 HOLOEYE LETO-3 空间光调制器的主要参数和用途"]

- 工具执行完毕后，我会返回一段以 "Observation:" 开头的内容，包含工具的返回结果。
  你需要基于 Observation 继续 Thought/Action，或在信息足够时给出最终 Answer。

- 当你已经获得足够信息可以直接回答用户问题时，请输出：
  Answer: 你的最终回答内容

重要约束：
- 回答时必须优先依据工具返回结果，不要凭空编造。
- 当你在使用工具后进行总结时，请明确使用类似表述：“根据知识库内容可概括为”。
- 对于“有哪些/清单/列表”等问题：
  * 应在 Observation 中提供的所有文件信息基础上，尽量汇总并去重相关实体（如设备名称、文档名称、工具名称）。
  * 如用户还问“怎么用/如何使用/使用方法/内容是什么”，应在列出文档后，继续调用 rag_qa，基于相关文档总结，而不是停留在文件列表层面。
  * 不要仅依据单一文档片段作答，而应综合多个相关文件。
- 回答末尾请给出你主要参考的文档标题或路径（如果 Observation 或 rag_qa 结果中有相关信息）。"""


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
    lines = text.strip().splitlines()
    for line in lines:
        if line.strip().startswith("Answer:"):
            return line.split("Answer:", 1)[1].strip()
    # 没有显式 Answer，则直接返回原文
    return text.strip()


def run_react_once(llm: ChatOpenAI, question: str, max_steps: int = 5, debug: bool = True) -> str:
    """
    手写一个简单 ReAct 循环：
    - 系统提示 + 用户问题 → LLM 输出 Thought/Action/Answer
    - 解析 Action，如果有工具调用则执行，并把 Observation 追加到对话中，再次让 LLM 推理
    - 最多 max_steps 次工具调用
    """
    messages: List[Dict[str, str]] = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    observation_text = ""
    for step in range(1, max_steps + 1):
        if debug:
            print(f"\n----- ReAct Step {step} -----")

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
            print("LLM 输出：")
            print(content)

        # 尝试解析 Answer（如果模型已经给出最终回答）
        if "Answer:" in content:
            final_answer = parse_answer(content)
            if debug:
                print("\n解析到最终 Answer，结束循环。")
            return final_answer

        # 解析 Action
        action = parse_action(content)
        if not action:
            # 没有 Action，也没有 Answer，当成直接回答
            if debug:
                print("\n未解析到 Action，直接将本次输出当作回答。")
            return content.strip()

        tool_name = action["tool_name"]
        arg = action["arg"]

        if debug:
            print(f"\n解析到 Action: tool={tool_name}, arg={arg}")

        tool = TOOLS.get(tool_name)
        if tool is None:
            observation_text = f"[工具错误] 未找到名为 {tool_name} 的工具。"
        else:
            try:
                # langchain_core.tools.Tool 对象：用 .invoke 调用
                # 若你的版本不支持 .invoke，可以用 .run 或直接 .func 看实际类型
                tool_input_key = list(tool.args.keys())[0]  # 第一个参数名
                tool_input = {tool_input_key: arg}
                tool_start = time.perf_counter()
                obs = tool.invoke(tool_input)
                tool_cost = time.perf_counter() - tool_start
                observation_text = str(obs)

                if debug:
                    debug_log(
                        f"工具调用完成: tool={tool_name}, 耗时={tool_cost:.3f}s, observation字符数={len(observation_text)}")
            except Exception as e:
                tool_cost = time.perf_counter() - tool_start
                observation_text = f"[工具执行异常] {e}"
                if debug:
                    debug_log(f"工具调用异常: tool={tool_name}, 耗时={tool_cost:.3f}s, error={e}")

        if debug:
            print("\nObservation:")
            print(observation_text)

        # 将本轮 LLM 输出 + Observation 追加到对话历史，再继续下一轮
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": f"Observation: {observation_text}\n请继续推理，若可以则给出 Answer。"})


    # 超出 max_steps 仍无 Answer，则返回最后一次 observation + 提示
    return f"工具调用达到最大步数，最后一次观察结果为：\n{observation_text}"


def ask_agent(question: str) -> str:
    """
    应用侧统一入口：ReAct LLM + 工具。
    """
    llm = build_qwen_llm()
    answer = run_react_once(llm, question, max_steps=5, debug=True)
    return answer


def ask_agent_stream(question: str):
    """
    流式 ReAct Agent：
    - type=step_llm：每一步 LLM 的输出（Thought/Action 文本）
    - type=step_tool：每次工具调用及其Observation
    - type=final：最终 Answer 文本
    - type=error：错误信息
    """
    llm = build_qwen_llm()

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    observation_text = ""
    used_list_tool = False
    used_rag_after_list = False

    try:
        max_steps = 4

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
                if used_list_tool and not used_rag_after_list:
                    # 拦截过早的 Answer，强制继续 rag_qa
                    messages.append({"role": "assistant", "content": content})
                    messages.append({
                        "role": "user",
                        "content": (
                            "你已经看到了相关文件列表，但还没有基于这些文件调用 rag_qa 深入阅读、总结‘分别怎么用’。"
                            "请先调用 rag_qa 工具，构造一个包含‘根据上面的 Observation，总结这些工具分别怎么用’的具体问题，"
                            "再根据 rag_qa 的 Observation 输出 Answer。"
                        )
                    })
                    continue

                final_answer = parse_answer(content)
                yield {
                    "type": "final",
                    "content": final_answer,
                }
                return

            # 3) 解析 Action
            action = parse_action(content)
            if not action:
                # 没有 Action，也没有 Answer，当作直接回答
                yield {
                    "type": "final",
                    "content": content.strip(),
                }
                return

            tool_name = action["tool_name"]
            arg = action["arg"]

            tool = TOOLS.get(tool_name)
            if tool is None:
                observation_text = f"[工具错误] 未找到名为 {tool_name} 的工具。"
            else:
                try:
                    tool_input_key = list(tool.args.keys())[0]
                    tool_input = {tool_input_key: arg}
                    tool_start = time.perf_counter()
                    obs = tool.invoke(tool_input)
                    tool_cost = time.perf_counter() - tool_start
                    observation_text = str(obs)
                    debug_log(
                        f"[stream] step={step} tool={tool_name} 耗时={tool_cost:.3f}s, observation字符数={len(observation_text)}")

                    if tool_name == "list_group_files":
                        used_list_tool = True
                    if used_list_tool and tool_name == "rag_qa":
                        used_rag_after_list = True

                except Exception as e:
                    observation_text = f"[工具执行异常] {e}"

            # 4) 把本次工具调用结果也流出去
            yield {
                "type": "step_tool",
                "step": step,
                "tool": tool_name,
                "arg": arg,
                "observation": observation_text,
            }

            # 5) 历史追加，进入下一轮
            messages.append({"role": "assistant", "content": content})
            messages.append({
                "role": "user",
                "content": (
                    f"Observation: {observation_text}\n"
                    "请继续思考并决定下一步：如果信息尚不足以回答，继续选择合适的工具调用；"
                    "如果信息已经足够，并且（若你之前调用过 list_group_files）已经在此基础上调用过 rag_qa，"
                    "则可以给出 Answer。"
                )
            })

        # 超出最大步数未得到 Answer
        yield {
            "type": "final",
            "content": f"工具调用达到最大步数，最后一次观察结果为：\n{observation_text}",
        }

    except Exception as e:
        traceback.print_exc()
        yield {
            "type": "error",
            "content": f"ReAct 执行异常: {e}",
        }


if __name__ == "__main__":
    q = input("请输入问题（例如：VLC小组有哪些设备？）：").strip()
    if not q:
        q = "VLC小组有哪些设备？"
    ans = ask_agent(q)
    print("\n=== Agent 最终回答 ===")
    print(ans)