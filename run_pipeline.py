"""入口脚本：对 mock_input.json 里的每个 query 跑三阶段管道，写出 output.json"""

import json
import os

from recall_filter import filter_recall_stage, truncate_prerank, filter_postrank

HERE = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(HERE, "mock_input.json"), encoding="utf-8") as f:
    data = json.load(f)

output = {}
for q in data["queries"]:
    qid = q["query_id"]
    stage1 = filter_recall_stage(q["query_res"])
    stage2 = truncate_prerank(stage1, topn=5)
    items = [item for bucket in stage2.values() for item in bucket]
    stage3 = filter_postrank(items, question_type=q.get("question_type", ""), topk=5)
    output[qid] = stage3
    print(
        f"{qid}: stage1={sum(len(v) for v in stage1.values() if isinstance(v, list))} "
        f"stage2={len(items)} final={len(stage3)}"
    )

with open(os.path.join(HERE, "output.json"), "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print("written: output.json")
