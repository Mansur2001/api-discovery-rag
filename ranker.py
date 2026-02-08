"""Two ranking strategies: baseline (cosine only) and QoS-aware (TOPSIS)."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from data_loader import APIRecord
from vector_store import RetrievalResult


@dataclass
class BaselineRankedResult:
    rank: int
    api_id: str
    name: str
    category: str
    method: str
    url: str
    description: str
    similarity_score: float


@dataclass
class QoSRankedResult:
    rank: int
    api_id: str
    name: str
    category: str
    method: str
    url: str
    description: str
    similarity_score: float
    topsis_score: float
    rt_ms: Optional[float]
    tp_rps: Optional[float]
    availability: Optional[float]
    valid_qos: bool


def rank_baseline(
    retrieval_results: list[RetrievalResult],
    no_qos_map: dict[str, APIRecord],
) -> list[BaselineRankedResult]:
    """Rank by cosine similarity only (preserves retrieval order)."""
    ranked = []
    for i, rr in enumerate(retrieval_results, 1):
        rec = no_qos_map.get(rr.api_id)
        if rec is None:
            continue
        ranked.append(
            BaselineRankedResult(
                rank=i,
                api_id=rr.api_id,
                name=rec.name,
                category=rec.category,
                method=rec.method,
                url=rec.url,
                description=rec.description.strip() or "(No description available)",
                similarity_score=round(rr.similarity_score, 4),
            )
        )
    return ranked


def rank_qos_aware(
    retrieval_results: list[RetrievalResult],
    qos_map: dict[str, APIRecord],
    weights: dict[str, float],
    cost_criteria: set[str],
) -> list[QoSRankedResult]:
    """Re-rank using TOPSIS on QoS metrics + similarity."""
    if not retrieval_results:
        return []

    # Pair retrieval results with QoS records
    pairs = []
    for rr in retrieval_results:
        rec = qos_map.get(rr.api_id)
        if rec is None:
            continue
        pairs.append((rr, rec))

    if not pairs:
        return []

    # Impute missing QoS values
    criteria_data = _impute_qos_values(pairs)

    # Build decision matrix: [rt_ms, tp_rps, availability, similarity]
    criteria_names = ["rt_ms", "tp_rps", "availability", "similarity"]
    n = len(criteria_data)
    matrix = np.zeros((n, 4))
    for i, cd in enumerate(criteria_data):
        matrix[i, 0] = cd["rt_ms"]
        matrix[i, 1] = cd["tp_rps"]
        matrix[i, 2] = cd["availability"]
        matrix[i, 3] = cd["similarity"]

    # Build weight and cost arrays in same column order
    w = np.array([weights[c] for c in criteria_names])
    is_cost = np.array([c in cost_criteria for c in criteria_names])

    # Compute TOPSIS scores
    scores = _topsis_score(matrix, w, is_cost)

    # Build results, sorted by TOPSIS score descending
    indexed = list(zip(pairs, criteria_data, scores))
    indexed.sort(key=lambda x: x[2], reverse=True)

    ranked = []
    for rank, ((rr, rec), cd, score) in enumerate(indexed, 1):
        qos = rec.qos
        ranked.append(
            QoSRankedResult(
                rank=rank,
                api_id=rr.api_id,
                name=rec.name,
                category=rec.category,
                method=rec.method,
                url=rec.url,
                description=rec.description.strip() or "(No description available)",
                similarity_score=round(rr.similarity_score, 4),
                topsis_score=round(float(score), 4),
                rt_ms=qos.rt_ms if qos else None,
                tp_rps=qos.tp_rps if qos else None,
                availability=qos.availability if qos else None,
                valid_qos=qos.valid_qos if qos else False,
            )
        )
    return ranked


def _topsis_score(
    decision_matrix: np.ndarray,
    weights: np.ndarray,
    is_cost: np.ndarray,
) -> np.ndarray:
    """Compute TOPSIS scores for a decision matrix.

    Steps:
    1. Normalize columns by L2 norm
    2. Apply weights
    3. Find ideal best (A+) and ideal worst (A-)
    4. Compute Euclidean distances to A+ and A-
    5. Score = D- / (D+ + D-)
    """
    n_rows, n_cols = decision_matrix.shape

    # Handle single-row edge case
    if n_rows == 1:
        return np.array([1.0])

    # Step 1: Normalize by column L2 norm
    norms = np.linalg.norm(decision_matrix, axis=0)
    # Avoid division by zero for all-zeros columns
    norms[norms == 0] = 1.0
    normalized = decision_matrix / norms

    # Step 2: Apply weights
    weighted = normalized * weights

    # Step 3: Ideal best and worst
    ideal_best = np.zeros(n_cols)
    ideal_worst = np.zeros(n_cols)
    for j in range(n_cols):
        if is_cost[j]:
            ideal_best[j] = weighted[:, j].min()
            ideal_worst[j] = weighted[:, j].max()
        else:
            ideal_best[j] = weighted[:, j].max()
            ideal_worst[j] = weighted[:, j].min()

    # Step 4: Euclidean distances
    d_plus = np.sqrt(np.sum((weighted - ideal_best) ** 2, axis=1))
    d_minus = np.sqrt(np.sum((weighted - ideal_worst) ** 2, axis=1))

    # Step 5: TOPSIS score
    denominator = d_plus + d_minus
    # Avoid division by zero
    scores = np.where(denominator > 0, d_minus / denominator, 0.5)

    return scores


def _impute_qos_values(
    pairs: list[tuple[RetrievalResult, APIRecord]],
) -> list[dict]:
    """Impute null/missing QoS values with worst-in-column defaults.

    null rt_ms -> max of valid values (worst latency)
    null tp_rps -> min of valid values (worst throughput)
    availability is always present (no nulls in data)
    """
    # Collect valid values for imputation
    valid_rt = [
        p[1].qos.rt_ms
        for p in pairs
        if p[1].qos and p[1].qos.rt_ms is not None
    ]
    valid_tp = [
        p[1].qos.tp_rps
        for p in pairs
        if p[1].qos and p[1].qos.tp_rps is not None
    ]

    # Fallback defaults if all values are null in the top-K
    default_rt = max(valid_rt) if valid_rt else 100.0
    default_tp = min(valid_tp) if valid_tp else 0.001

    results = []
    for rr, rec in pairs:
        qos = rec.qos
        if qos is None:
            results.append({
                "rt_ms": default_rt,
                "tp_rps": default_tp,
                "availability": 0.0,
                "similarity": rr.similarity_score,
            })
        else:
            results.append({
                "rt_ms": qos.rt_ms if qos.rt_ms is not None else default_rt,
                "tp_rps": qos.tp_rps if qos.tp_rps is not None else default_tp,
                "availability": qos.availability if qos.availability is not None else 0.0,
                "similarity": rr.similarity_score,
            })
    return results
