# app_test.py

from agent_entry import ask_agent


def pretty_print_title(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def run_single_query(question: str):
    pretty_print_title("应用侧测试开始")
    print(f"用户问题：{question}")

    pretty_print_title("ReAct 过程与中间结果")
    answer = ask_agent(question)

    pretty_print_title("Agent 最终回答")
    print(answer)


if __name__ == "__main__":
    q = input("请输入测试问题：").strip()
    run_single_query(q)