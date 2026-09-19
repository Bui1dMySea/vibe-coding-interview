"""入口：主路 general 粗排+精排；没有可用相同再走 8889/8899。"""

import json
import os

from recall_filter import filter_recall_stage, truncate_prerank, filter_postrank

HERE = os.path.dirname(os.path.abspath(__file__))
HIST = {"8889", "8899"}


def _flat(buckets):
    return [item for v in buckets.values() if isinstance(v, list) for item in v]


with open(os.path.join(HERE, "mock_input.json"), encoding="utf-8") as f:
    data = json.load(f)

output = {}
for q in data["queries"]:
    qid = q["query_id"]
    stage1 = filter_recall_stage(q["query_res"])
    primary = {k: v for k, v in stage1.items() if k not in HIST}
    hist = {k: v for k, v in stage1.items() if k in HIST}
    stage2 = truncate_prerank(primary, topn=5)
    items = _flat(stage2)
    stage3 = filter_postrank(items, question_type=q.get("question_type", ""), topk=5)
    if not stage3:
        stage2 = truncate_prerank(hist, topn=5)
        items = _flat(stage2)
        stage3 = filter_postrank(
            items, question_type=q.get("question_type", ""), topk=5
        )
    output[qid] = stage3
    print(
        f"{qid}: recalled={sum(len(v) for v in stage1.values() if isinstance(v, list))} "
        f"into_rank={len(items)} final={len(stage3)}"
    )

with open(os.path.join(HERE, "output.json"), "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("written: output.json")
