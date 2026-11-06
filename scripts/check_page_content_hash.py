#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""零重建判定：比较当前 clean_text 输出与快照的 SHA-256，判断是否必须全量重建。

用法：
    python scripts/check_page_content_hash.py --save     # 生成/更新快照
    python scripts/check_page_content_hash.py            # 与已有快照比对
    python scripts/check_page_content_hash.py --snapshot <path>  # 指定快照

退出码：0 = 完全一致（零重建）；1 = 有差异（需重建）；2 = 快照缺失或加载失败。
只读语料，不修改任何生产数据；--save 仅写 logs/ 下的快照文件。
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from lab_rag import ingest  # noqa: E402

DEFAULT_SNAPSHOT = os.path.join(ROOT, "logs", "clean_text_hashes_before.json")


def compute(cfg: dict) -> dict:
    from pathlib import Path
    root = Path(cfg["DOCS_PATH"])
    out = {}
    for f in sorted(root.rglob("*.md")):
        text, _ = ingest._decode_markdown(f.read_bytes())
        cleaned = ingest.clean_text(text, cfg) if text is not None else ""
        key = str(f).replace(str(root).rstrip("/") + "/", "")
        out[key] = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save", action="store_true", help="生成/更新快照")
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    args = ap.parse_args()

    cfg = ingest.load_config()
    current = compute(cfg)
    combined = hashlib.sha256("".join(sorted(current.values())).encode()).hexdigest()[:16]
    print(f"docs={len(current)} combined={combined}")

    if args.save or not os.path.isfile(args.snapshot):
        os.makedirs(os.path.dirname(args.snapshot), exist_ok=True)
        with open(args.snapshot, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"snapshot saved: {args.snapshot}")
        return 0 if args.save else 2

    with open(args.snapshot, encoding="utf-8") as f:
        baseline = json.load(f)
    changed = sorted(k for k in set(baseline) | set(current) if baseline.get(k) != current.get(k))
    if changed:
        print(f"CHANGED={len(changed)} (需要全量重建)")
        for name in changed[:20]:
            print(f"  - {name}")
        return 1
    print("IDENTICAL: page_content 逐字一致，本轮零重建")
    return 0


if __name__ == "__main__":
    sys.exit(main())
