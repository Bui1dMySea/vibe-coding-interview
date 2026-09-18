"""
搜题候选三阶段过滤器（初版实现）

阶段1: 召回过滤    —— filter_recall_stage
阶段2: 粗排截断    —— truncate_prerank
阶段3: 精排后路由  —— filter_postrank

规格见 TASK.md。本实现按规格编写，已通过自测。
"""

import math
from typing import Dict, List, Optional

# ---------- 阶段1 默认参数 ----------
_RECALL_MIN_SCORE: float = 0.90
_RECALL_TARGET_DB_IDS: frozenset = frozenset({"8889", "8899"})
_RECALL_INVALID_API_IDS: frozenset = frozenset({"777"})

# ---------- 阶段2 默认参数 ----------
_PRERANK_TOPN: int = 10

# ---------- 阶段3 默认参数 ----------
_POSTRANK_TOPK: int = 5


# ========== 工具函数 ==========

def _get_ranker_score(item: dict) -> float:
    """提取 ranker_score；缺失时返回 -inf 排末位"""
    rs = item.get("ranker_score")
    if rs is None:
        return -math.inf
    return float(rs)


# ========== 阶段1: 召回后过滤 ==========

def filter_recall_stage(
    query_res: Dict[str, List[dict]],
    min_score: float = _RECALL_MIN_SCORE,
    target_db_ids: Optional[frozenset] = _RECALL_TARGET_DB_IDS,
    invalid_api_ids: frozenset = _RECALL_INVALID_API_IDS,
    per_db_topn: int = 20,
) -> Dict[str, List[dict]]:
    """
    召回后过滤，进粗排之前执行。

    目标库内候选须同时满足:
      - score 达到 min_score 阈值
      - is_same != 1
      - api_id 不在 invalid_api_ids 中
    过滤后每个目标库按 score 降序保留最多 per_db_topn 条。
    非目标库原样透传。
    """
    filtered: Dict[str, List[dict]] = {}

    for db_id, items in query_res.items():
        if not isinstance(items, list):
            filtered[db_id] = items
            continue

        if target_db_ids is not None and db_id not in target_db_ids:
            filtered[db_id] = items
            continue

        kept = []
        for item in items:
            if float(item.get("score", 0.0)) <= min_score:
                continue
            if item.get("is_same") == 1:
                continue
            if item.get("api_id") in invalid_api_ids:
                continue
            kept.append(item)

        kept.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
        filtered[db_id] = kept[:per_db_topn]

    return filtered


# ========== 阶段2: 粗排后截断 ==========

def truncate_prerank(
    pre_ranking_dict: Dict[str, List[dict]],
    topn: int = _PRERANK_TOPN,
    score_key: str = "pre_ranking_score",
) -> Dict[str, List[dict]]:
    """
    粗排后截断，进精排之前执行。

    按 pre_ranking_score 降序，取 topN 后重组 dict。
    """
    result: Dict[str, List[dict]] = {}
    for db_id, items in pre_ranking_dict.items():
        if not isinstance(items, list):
            continue
        ranked = sorted(
            items,
            key=lambda x: float(x.get(score_key, 0.0)),
            reverse=True,
        )
        result[db_id] = ranked[:topn]
    return result


# ========== 阶段3: 精排后过滤与路由 ==========

def filter_postrank(
    items: List[dict],
    question_type: str = "",
    topk: int = _POSTRANK_TOPK,
    thresholds: Optional[Dict[str, float]] = None,
) -> List[dict]:
    """
    精排后过滤，返回最终输出候选。

    按 ranker_score 降序取 topK；无候选时返回 [] 走 fallback 解题链路。
    """
    if not items:
        return []

    scored = sorted(items, key=_get_ranker_score, reverse=True)
    return scored[:topk]
