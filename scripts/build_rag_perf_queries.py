#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""构建并校验 RAG 链路测试集（可重复运行，输出 tests/rag_perf_queries.json）。

校验内容：
1. 路由一致性：用 rag_agent.rule_based_route 实测每条 query；
   - rag_search / file_list / hybrid 必须命中且与 expected_route 一致
   - llm_fallback 必须返回 None（规则未命中，交 LLM 兜底）
2. 落点真实性：grounding 指向的 Markdown 文件或目录必须在 DOCS_PATH 下存在
   （对含 CJK 兼容字符的真实文件名做 NFKC 归一容错）。
3. 契约完整性：id 唯一、必填字段齐全、expected_behavior 与 grounding_type 自洽。

任一项失败打印明细并以非零退出，避免把不可信测试集用于性能评估。
"""
import os
import sys
import json
import unicodedata
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lab_rag import rag_agent

DOCS = "/mnt/cpu_share"
OUT = os.path.join(ROOT, "tests", "rag_perf_queries.json")
HOLDOUT_OUT = os.path.join(ROOT, "tests", "rag_perf_holdout.json")
REAL_ROUTES = {"rag_search", "file_list", "hybrid"}
NO_GROUNDING = "（知识库无对应文档）"


def Q(qid, query, route, doc_type, grounding, topic,
      case_tag="", behavior="hit_source"):
    return {
        "id": qid,
        "query": query,
        "expected_route": route,
        "doc_type": doc_type,
        "grounding": grounding,
        "expected_topic": topic,
        "expected_behavior": behavior,
        "case_tag": case_tag,
    }


ROWS = [
    # ============ 原有 36 条：全文档类型 + 多重路由 ============
    Q("P01", "人工智能驱动的科研范式变革对研究方法有什么影响？", "rag_search", "论文/科研报告",
      "20260323_韦子扬_人工智能驱动的科研范式变革_v0.5a.md", "科研范式变革"),
    Q("P02", "这篇科研范式变革报告提到的核心技术和研究方案是什么？", "rag_search", "论文/科研报告",
      "20260323_韦子扬_人工智能驱动的科研范式变革_v0.5a.md", "核心技术/研究方案"),
    Q("P03", "水下可见光通信的关键技术和系统架构有哪些？", "rag_search", "论文/科研报告",
      "参考书/水下可见光通信关键技术.md", "关键技术/系统架构"),
    Q("P04", "水下无线光通信的信道模型和调制方法是什么？", "rag_search", "论文/科研报告",
      "参考书/水下无线光通信.md", "信道模型/调制方法"),

    Q("B01", "水下可见光通信技术与应用这本书覆盖了哪些关键技术？", "rag_search", "参考书",
      "参考书/水下可见光通信关键技术.md", "关键技术覆盖"),
    Q("B02", "水下无线光通信的信号模型和信道特性是什么？", "rag_search", "参考书",
      "参考书/水下无线光通信.md", "信号模型/信道特性"),

    Q("N01", "团队文档提交的命名与格式要求是什么？", "rag_search", "团队规范/要求",
      "团队工作及文档要求汇总_v0.1a_20250224.md", "命名/格式要求"),
    Q("N02", "实验拍照有哪些具体要求？", "rag_search", "团队规范/要求",
      "实验拍照要求.md", "实验拍照要求"),
    Q("N03", "论文图形格式对分辨率和字体有什么规定？", "rag_search", "团队规范/要求",
      "论文图形格式要求.md", "图形分辨率/字体规定"),
    Q("N04", "论文、报告撰写格式规范（Latex 版）包含哪些要求？", "rag_search", "团队规范/要求",
      "论文、报告撰写格式规范Latex版.md", "Latex 撰写规范"),

    Q("T01", "visio 图形如何转 eps？", "rag_search", "工具教程",
      "将visio图形转eps.md", "visio 转 eps"),
    Q("T02", "论文写作辅助工具有哪些，如何使用？", "rag_search", "工具教程",
      "论文辅助工具介绍-黄颂康.md", "论文写作辅助工具"),
    Q("T03", "论文写作经验分享里提到的写作方法有哪些？", "rag_search", "工具教程",
      "论文写作经验分享-武晓鸽.md", "写作方法"),

    Q("G01", "组会资料里关于实验方案设计的总结有哪些？", "rag_search", "组会/学期总结",
      "组会资料/20220121期末组会/小组期末总结/2021年秋季X2X组会总结.md", "实验方案设计总结"),
    Q("G02", "20230106 期末组会中 VLC小组 汇报了哪些实验进展和研究成果？", "rag_search", "组会/学期总结",
      "组会资料/20230106期末组会/20230106期末组会_许健鹏.md", "实验进展/研究成果"),
    Q("G03", "组会期末总结里提到的论文写作方法有哪些建议？", "rag_search", "组会/学期总结",
      "组会资料/20220121期末组会/论文写作_v2.md", "论文写作建议"),

    Q("D01", "MSO-X 3054A 示波器的触发参数如何设置？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Agilent MSO-X 3054A 示波器/MSOX3054A-数据手册-pdf-www.findic.com.md", "触发参数设置"),
    Q("D02", "Keysight N9322C 频谱分析仪的 RBW 参数怎么设置？", "rag_search", "设备手册",
      "设备操作指南/A2X小组/Keysight N9322C 频谱分析仪/N9322C基础型频谱分析仪.md", "RBW 参数"),
    Q("D03", "Keysight EDU36311A 电源的三路输出参数如何设置？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Keysight EDU36311A 电源/EDU36311A-三路输出可编程直流电源用户指南_CH.md", "三路输出参数"),
    Q("D04", "Newport 843-R 光功率计的测量不确定度如何评估？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Newport 843-R 光功率计/Understanding-Total-Measurement-Uncertainty.md", "测量不确定度"),
    Q("D05", "HOLOEYE LETO-3 空间光调制器的相位调制原理是什么？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/HOLOEYE LETO-3 空间光调制器/Manuals/LETO-3-Manual.md", "相位调制原理"),
    Q("D06", "Mini-Circuits ZHL-6A-S+ 功率放大器的增益参数怎么设置？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Mini-Circuits ZHL-6A-S+ 功率放大器/Mini-Circuits ZHL-6A-S+ datasheet.md", "增益参数"),
    Q("D07", "Agilent N5171 矢量信号发生器的输出功率范围如何设置？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Agilent N5171 矢量信号发生器/Aglient N5171、N5172B中文资料--旧的版本.md", "输出功率范围"),
    Q("D08", "SIGLENT SVA 1032X 矢量网络分析仪的校准步骤是什么？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/SIGLENT SVA 1032X 矢量网络分析仪/SVA1000X-series-usermanual.md", "校准步骤"),

    Q("F01", "列出 VLC小组 的所有设备手册清单", "file_list", "设备手册",
      "设备操作指南/VLC小组", "VLC 设备清单"),
    Q("F02", "A2X小组 有哪些设备？", "file_list", "设备手册",
      "设备操作指南/A2X小组", "A2X 设备清单"),
    Q("F03", "VLC小组 一共有多少台设备？", "file_list", "设备手册",
      "设备操作指南/VLC小组", "VLC 设备数量"),
    Q("F04", "VLC小组 的 Manuals 文件夹里有哪些文件？", "file_list", "设备手册",
      "设备操作指南/VLC小组/HOLOEYE LETO-3 空间光调制器/Manuals", "Manuals 目录文件"),

    Q("H01", "VLC小组有哪些设备，并说明 MSO-X 3054A 触发参数如何设置？", "hybrid", "设备手册",
      "设备操作指南/VLC小组 | 设备操作指南/VLC小组/Agilent MSO-X 3054A 示波器/MSOX3054A-数据手册-pdf-www.findic.com.md",
      "设备清单 + 触发参数"),
    Q("H02", "A2X小组有哪些设备，N9322C 频谱分析仪的 RBW 参数怎么设置？", "hybrid", "设备手册",
      "设备操作指南/A2X小组 | 设备操作指南/A2X小组/Keysight N9322C 频谱分析仪/N9322C基础型频谱分析仪.md",
      "设备清单 + RBW 参数"),
    Q("H03", "列出 VLC小组 的设备，并说明 HOLOEYE LETO-3 的相位调制原理", "hybrid", "设备手册",
      "设备操作指南/VLC小组 | 设备操作指南/VLC小组/HOLOEYE LETO-3 空间光调制器/Manuals/LETO-3-Manual.md",
      "设备清单 + 相位调制原理"),
    Q("H04", "VLC小组有哪些设备，Newport 843-R 光功率计的测量方法是什么？", "hybrid", "设备手册",
      "设备操作指南/VLC小组 | 设备操作指南/VLC小组/Newport 843-R 光功率计/Understanding-Total-Measurement-Uncertainty.md",
      "设备清单 + 测量方法"),

    Q("L01", "可见光通信的发展现状", "llm_fallback", "参考书",
      "参考书/水下可见光通信关键技术.md", "主题级问法（无明确动作）"),
    Q("L02", "我想了解水下光通信的研究", "llm_fallback", "论文/科研报告",
      "参考书/水下无线光通信技术与应用.md", "主题级问法（无明确动作）"),
    Q("L03", "VLC 系统怎么搭建？", "llm_fallback", "设备手册",
      "设备操作指南/VLC小组", "歧义 + 单学术词"),
    Q("L04", "帮我找一下关于信号处理的资料", "llm_fallback", "通用",
      "（跨文档）", "无规则命中词"),

    # ============ 新增：跨设备同类手册消歧 ============
    Q("X01", "USRP-2944 Specifications 中发射机通道的参数是多少？", "rag_search", "设备手册",
      "设备操作指南/A2X小组/NI大规模MIMO系统/USRP-2944 Specifications.md", "USRP-2944 发射机通道",
      case_tag="跨设备同类手册"),
    Q("X02", "APD210-Manual 中适用型号的参数说明有哪些？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Menlos Systems APD 210 光电探测器/APD210-Manual.md", "APD210 适用型号",
      case_tag="跨设备同类手册"),
    Q("X03", "SLM-DISPLAY-SDK-V3.0 版本的 Advanced API 使用方法是什么？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/HOLOEYE LETO-3 空间光调制器/SLM-DISPLAY-SDK-V3.0-for-MATLAB-Windows/SLM Display SDK.md",
      "SDK V3.0 Advanced API", case_tag="跨设备同类手册"),
    Q("X04", "SVA1000X_datasheet_QG0701X_C05B 中频率范围的参数是多少？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/SIGLENT SVA 1032X 矢量网络分析仪/SVA1000X_datasheet_QG0701X_C05B.md",
      "SVA1000X 频率范围", case_tag="跨设备同类手册"),
    Q("X05", "T1G-Manual 与 APD210-Manual 的操作步骤有什么区别？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Thorlabs T1G 激光驱动板Bias Tee/T1G-Manual.md | "
      "设备操作指南/VLC小组/Menlos Systems APD 210 光电探测器/APD210-Manual.md",
      "两份 Manual 差异", case_tag="跨设备同类手册"),
    Q("X06", "LETO-3 Configuration Manager Manual 的配置方法是什么？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/HOLOEYE LETO-3 空间光调制器/LETO-3_Configuration_Manager/LETO-3 Configuration Manager Manual.md",
      "Configuration Manager 配置", case_tag="跨设备同类手册"),

    # ============ 新增：纯英文手册（中文提问 / 英文提问） ============
    Q("E01", "N9322C Basic Spectrum Analyzer 中 DANL 的参数是多少？", "rag_search", "设备手册",
      "设备操作指南/A2X小组/Keysight N9322C 频谱分析仪/N9322C Basic Spectrum Analyzer.md",
      "DANL 指标", case_tag="纯英文手册"),
    Q("E02", "EDU36311A 编程指南中 SCPI common commands 如何使用？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Keysight EDU36311A 电源/EDU36311A-编程指南_En.md",
      "SCPI 命令", case_tag="纯英文手册"),
    Q("E03", "USRP RIO device Getting Started Guide 的操作步骤是什么？", "rag_search", "设备手册",
      "设备操作指南/A2X小组/NI大规模MIMO系统/USRP RIO device Getting Started Guide.md",
      "USRP RIO 入门", case_tag="纯英文手册"),
    Q("E04", "Understanding-Total-Measurement-Uncertainty 文档中测量不确定度的评估方法是什么？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Newport 843-R 光功率计/Understanding-Total-Measurement-Uncertainty.md",
      "不确定度评估", case_tag="纯英文手册"),
    Q("E05", "Propsim User Reference 中 FS16 信道仿真配置的说明是什么？", "rag_search", "设备手册",
      "设备操作指南/A2X小组/Keysight PROPSIM FS16 信道仿真仪 F8820A/Propsim User Reference.md",
      "FS16 信道仿真配置", case_tag="纯英文手册"),
    Q("E06", "How to set trigger on Infiniium 9000 series oscilloscope?", "llm_fallback", "设备手册",
      "设备操作指南/VLC小组/Agilent MSO 9404A 混合信号示波器/Infiniium 9000系列示波器用户指南_En.md",
      "英文问法 → LLM 路由", case_tag="纯英文手册"),

    # ============ 新增：OCR 残缺文档 ============
    Q("O01", "MSOX3054A 其他数据使用手册中 3000T X-Series 的 Zone Touch Trigger 功能如何描述？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Agilent MSO-X 3054A 示波器/MSOX3054A-其他数据使用手册-www.findic.com.md",
      "Zone Touch Trigger（U+FFFD 密集）", case_tag="OCR残缺"),
    Q("O02", "MIMO Application Framework Manual 中 Reciprocity Calibration 的实现方法是什么？", "rag_search", "设备手册",
      "设备操作指南/A2X小组/NI大规模MIMO系统/20201017_NI massive MIMO 开发培训/MIMO_Application_Framework_Manual_ver_1.5.md",
      "Reciprocity Calibration", case_tag="OCR残缺"),

    # ============ 新增：多跳（需跨文档组合） ============
    Q("M01", "USRP-2943 和 USRP-2944 的规格参数有什么区别？", "rag_search", "设备手册",
      "设备操作指南/A2X小组/NI大规模MIMO系统/USRP-2943 Specifications.md | "
      "设备操作指南/A2X小组/NI大规模MIMO系统/USRP-2944 Specifications.md",
      "两份规格书对比", case_tag="多跳"),
    Q("M02", "论文图形格式要求与论文、报告撰写格式规范对图片格式的要求有什么区别？", "rag_search", "团队规范/要求",
      "论文图形格式要求.md | 论文、报告撰写格式规范Latex版.md",
      "两份规范对比", case_tag="多跳"),
    Q("M03", "VLC小组 的示波器与信号发生器在触发设置和输出参数上分别有什么要求？", "rag_search", "设备手册",
      "设备操作指南/VLC小组/Agilent MSO-X 3054A 示波器/MSOX3054A-数据手册-pdf-www.findic.com.md | "
      "设备操作指南/VLC小组/Agilent N5171 矢量信号发生器/Aglient N5171、N5172B中文资料--旧的版本.md",
      "两类设备参数", case_tag="多跳"),

    # ============ 新增：否定语义 ============
    Q("NEG01", "团队文档中哪些格式是不允许提交的？", "rag_search", "团队规范/要求",
      "团队工作及文档要求汇总_v0.1a_20250224.md", "否定式提问", case_tag="否定"),
    Q("NEG02", "N9322C 频谱分析仪不支持哪些测量方法？", "rag_search", "设备手册",
      "设备操作指南/A2X小组/Keysight N9322C 频谱分析仪/N9322C基础型频谱分析仪.md",
      "否定式提问", case_tag="否定"),

    # ============ 新增：知识库无答案（期望拒答） ============
    Q("U01", "阿司匹林临床用药剂量的使用说明是什么？", "rag_search", "无答案",
      NO_GROUNDING, "临床医学", case_tag="无答案", behavior="refuse"),
    Q("U02", "Django ORM 的查询方法是什么？", "rag_search", "无答案",
      NO_GROUNDING, "通用 Web 框架", case_tag="无答案", behavior="refuse"),
    Q("U03", "今天上海天气的实时温度如何？", "rag_search", "无答案",
      NO_GROUNDING, "实时数据", case_tag="无答案", behavior="refuse"),
    Q("U04", "Kubernetes 集群部署的操作步骤是什么？", "rag_search", "无答案",
      NO_GROUNDING, "云原生运维", case_tag="无答案", behavior="refuse"),
    Q("U05", "红烧肉的烹饪步骤是什么？", "rag_search", "无答案",
      NO_GROUNDING, "日常生活", case_tag="无答案", behavior="refuse"),

    # ============ 新增：路由边界（深层目录 / 文件枚举 / 复合意图） ============
    Q("F05", "列出 HOLOEYE LETO-3 空间光调制器 下的所有 SDK 文件夹", "file_list", "设备手册",
      "设备操作指南/VLC小组/HOLOEYE LETO-3 空间光调制器", "SDK 子目录枚举",
      case_tag="路由边界"),
    Q("F06", "A2X小组 NI大规模MIMO系统 目录里有哪些 Specifications 文件？", "file_list", "设备手册",
      "设备操作指南/A2X小组/NI大规模MIMO系统", "规格书文件枚举",
      case_tag="路由边界"),
    Q("H05", "A2X小组有哪些设备，USRP-2944 和 USRP-2943 的规格有什么区别？", "hybrid", "设备手册",
      "设备操作指南/A2X小组 | 设备操作指南/A2X小组/NI大规模MIMO系统/USRP-2944 Specifications.md",
      "设备清单 + 规格对比", case_tag="路由边界"),
]


def nfkc(text):
    return unicodedata.normalize("NFKC", text)


def locate(rel_path):
    """校验 grounding 落点存在；对 CJK 兼容字符文件名做 NFKC 归一容错。"""
    target = os.path.join(DOCS, rel_path)
    if os.path.exists(target):
        return True
    parent = os.path.dirname(target)
    if not os.path.isdir(parent):
        return False
    wanted = nfkc(os.path.basename(target))
    try:
        for name in os.listdir(parent):
            if nfkc(name) == wanted:
                return True
    except OSError:
        return False
    return False


def grounding_parts(grounding, gtype):
    if gtype == "none":
        return []
    if gtype == "multi":
        return [p.strip() for p in grounding.split("|") if p.strip()]
    return [grounding]


def infer_grounding_type(grounding):
    if grounding in (NO_GROUNDING, "（跨文档）"):
        return "none"
    if "|" in grounding:
        return "multi"
    if grounding.endswith(".md"):
        return "file"
    return "dir"


def validate_rows(rows, label, errors, corrections):
    """路由/落点/契约校验；返回补齐 grounding_type 后的行（复制，不改原表）。

    路由以 rule_based_route 实测为准：与数据表设计意图不一致时按实测校正
    expected_route，并把校正项写入 corrections 供审计（避免测试集与真实逻辑漂移）。
    """
    validated = []
    for raw in rows:
        r = dict(raw)
        gtype = infer_grounding_type(r["grounding"])
        r["grounding_type"] = gtype

        res = rag_agent.rule_based_route(r["query"])
        rule_route = res["route"] if isinstance(res, dict) else None
        measured = rule_route if rule_route is not None else "llm_fallback"
        designed = r["expected_route"]
        r["designed_route"] = designed
        r["expected_route"] = measured
        if designed != measured:
            corrections.append({"id": r["id"], "set": label, "query": r["query"],
                                "designed": designed, "measured": measured})

        for part in grounding_parts(r["grounding"], gtype):
            if not locate(part):
                errors.append((r["id"], r["query"], f"[{label}] grounding 落点不存在: {part}"))

        if r["expected_behavior"] == "refuse" and gtype != "none":
            errors.append((r["id"], r["query"], f"[{label}] refuse 案例的 grounding_type 必须为 none"))
        if r["expected_behavior"] == "hit_source" and r["expected_route"] not in REAL_ROUTES | {"llm_fallback"}:
            errors.append((r["id"], r["query"], f"[{label}] 非法 expected_route: {r['expected_route']}"))
        validated.append(r)
    return validated


def distribution(rows):
    return {
        "total": len(rows),
        "route_distribution": dict(Counter(q["expected_route"] for q in rows)),
        "doc_type_distribution": dict(Counter(q["doc_type"] for q in rows)),
        "case_tag_distribution": dict(Counter(q["case_tag"] or "原有基线" for q in rows)),
        "behavior_distribution": dict(Counter(q["expected_behavior"] for q in rows)),
        "grounding_type_distribution": dict(Counter(q["grounding_type"] for q in rows)),
    }


def validate_distribution(meta, errors):
    """分布均衡门槛（收敛门槛的构建期校验，避免生成不可信测试集）。"""
    dist = meta["doc_type_distribution"]
    total = meta["total"]
    manual_ratio = dist.get("设备手册", 0) / total
    meta["device_manual_ratio"] = round(manual_ratio, 4)
    non_rag = 1 - meta["route_distribution"].get("rag_search", 0) / total
    meta["non_rag_route_ratio"] = round(non_rag, 4)
    tags = meta["case_tag_distribution"]

    if total < 120:
        errors.append(("META", "", f"主集条数不足 120: {total}"))
    if manual_ratio > 0.40:
        errors.append(("META", "", f"设备手册占比 {manual_ratio:.3f} 超过 0.40"))
    min_doc = min(dist.values())
    if min_doc < 8:
        errors.append(("META", "", f"存在 doc_type 少于 8 条: min={min_doc} {dist}"))
    if len(tags) < 14:
        errors.append(("META", "", f"case_tag 类别不足 14: {len(tags)}"))
    min_tag = min(tags.values())
    if min_tag < 3:
        errors.append(("META", "", f"存在 case_tag 少于 3 条: min={min_tag} {tags}"))
    if non_rag < 0.25:
        errors.append(("META", "", f"非 rag_search 路由占比 {non_rag:.3f} 低于 0.25"))


def main():
    # 延迟导入避免循环：rag_perf_cases 需要本模块的 Q() / NO_GROUNDING
    from rag_perf_cases import EXTRA_MAIN, HOLDOUT

    errors = []
    corrections = []
    queries = validate_rows(ROWS + EXTRA_MAIN, "main", errors, corrections)
    holdout = validate_rows(HOLDOUT, "holdout", errors, corrections)

    # 主集与留出集双向隔离：id 与 query 文本均不得重复
    main_ids = {q["id"] for q in queries}
    holdout_ids = {q["id"] for q in holdout}
    if len(queries) != len(main_ids) or len(holdout) != len(holdout_ids):
        errors.append(("META", "", "id 重复（主集或留出集）"))
    if main_ids & holdout_ids:
        errors.append(("META", "", f"主集与留出集 id 重叠: {sorted(main_ids & holdout_ids)}"))
    main_texts = {q["query"].strip() for q in queries}
    overlap = sorted({q["query"].strip() for q in holdout} & main_texts)
    if overlap:
        errors.append(("META", "", f"留出集 query 与主集重复: {overlap}"))
    if len(holdout) < 20:
        errors.append(("META", "", f"留出集不足 20 条: {len(holdout)}"))

    meta = distribution(queries)
    validate_distribution(meta, errors)

    if errors:
        print("=== 校验失败 ===")
        for qid, query, msg in errors:
            print(f"  [{qid}] {query}\n       -> {msg}")
        print(f"共 {len(errors)} 项失败，未写出文件。")
        sys.exit(1)

    meta.update({
        "description": "RAG 链路性能测试集：覆盖全部文档类型、多重路由与 14+ 类边界案例，附独立留出集。",
        "contract": {
            "expected_behavior": "hit_source=期望引用/目录命中 grounding；refuse=期望拒答（无相关内容）",
            "case_tag": "边界案例分类，空串为原有基线案例",
            "grounding_type": "file=单文件；dir=目录；multi=多落点（| 分隔，任一命中即算）；none=无落点",
        },
        "holdout": {"path": "tests/rag_perf_holdout.json", "total": len(holdout),
                    "note": "留出集不参与调优，仅用于泛化评估；query 与主集零重叠"},
        "route_corrections": corrections,
        "route_corrections_count": len(corrections),
        "note": "全部 query 已用 rule_based_route 实测校验；全部 grounding 已校验真实存在（NFKC 容错）；分布均衡与留出集隔离校验通过。",
    })
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "queries": queries}, f, ensure_ascii=False, indent=2)

    holdout_meta = distribution(holdout)
    holdout_meta.update({
        "description": "独立留出集：不参与任何调优，仅用于泛化评估。",
        "note": "与主集 id/query 零重叠；路由与落点校验同主集。",
    })
    with open(HOLDOUT_OUT, "w", encoding="utf-8") as f:
        json.dump({"meta": holdout_meta, "queries": holdout}, f, ensure_ascii=False, indent=2)

    print(f"OK: 写出 {OUT}，共 {len(queries)} 条")
    print(f"OK: 写出 {HOLDOUT_OUT}，共 {len(holdout)} 条")
    for key in ("total", "route_distribution", "doc_type_distribution", "case_tag_distribution",
                "behavior_distribution", "grounding_type_distribution",
                "device_manual_ratio", "non_rag_route_ratio"):
        print(f"  {key}: {meta[key]}")


if __name__ == "__main__":
    main()
