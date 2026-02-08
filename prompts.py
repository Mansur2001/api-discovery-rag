"""Prompt templates for baseline and QoS-aware LLM generation."""

from ranker import BaselineRankedResult, QoSRankedResult


def build_baseline_prompt(
    query: str, ranked_results: list[BaselineRankedResult]
) -> str:
    """Build prompt for baseline (cosine-similarity-only) explanation."""
    if not ranked_results:
        return f'No APIs were found for the query: "{query}". Please try a different search.'

    # Build table rows
    table_rows = []
    for r in ranked_results:
        table_rows.append(
            f"| {r.rank} | {r.name} | {r.category} | {r.method} | {r.similarity_score:.4f} |"
        )
    table = "\n".join(table_rows)

    top = ranked_results[0]

    return f"""You are an API recommendation assistant. A user searched for: "{query}"

The following APIs were found, ranked by semantic similarity to the query:

| Rank | Name | Category | Method | Similarity Score |
|------|------|----------|--------|-----------------|
{table}

Top result details:
- Name: {top.name}
- Description: {top.description}
- URL: {top.url}
- Category: {top.category}
- Method: {top.method}
- Similarity Score: {top.similarity_score:.4f}

Explain in 2-3 sentences why "{top.name}" is ranked #1 based on relevance to the query.
Then briefly note any interesting alternatives in the top results."""


def build_qos_aware_prompt(
    query: str, ranked_results: list[QoSRankedResult]
) -> str:
    """Build prompt for QoS-aware (TOPSIS) explanation."""
    if not ranked_results:
        return f'No APIs were found for the query: "{query}". Please try a different search.'

    # Build table rows
    table_rows = []
    for r in ranked_results:
        rt = f"{r.rt_ms:.2f}" if r.rt_ms is not None else "N/A"
        tp = f"{r.tp_rps:.2f}" if r.tp_rps is not None else "N/A"
        avail = f"{r.availability * 100:.1f}%" if r.availability is not None else "N/A"
        qos_flag = "" if r.valid_qos else " *"
        table_rows.append(
            f"| {r.rank} | {r.name} | {r.topsis_score:.4f} | "
            f"{r.similarity_score:.4f} | {rt} | {tp} | {avail}{qos_flag} |"
        )
    table = "\n".join(table_rows)

    top = ranked_results[0]
    top_rt = f"{top.rt_ms:.2f}ms" if top.rt_ms is not None else "N/A"
    top_tp = f"{top.tp_rps:.2f} req/s" if top.tp_rps is not None else "N/A"
    top_avail = f"{top.availability * 100:.1f}%" if top.availability is not None else "N/A"

    return f"""You are an API recommendation assistant specializing in Quality of Service analysis.
A user searched for: "{query}"

The following APIs were ranked using TOPSIS multi-criteria analysis considering latency, throughput, availability, and semantic relevance:

| Rank | Name | TOPSIS | Similarity | Latency(ms) | Throughput(rps) | Availability |
|------|------|--------|-----------|-------------|-----------------|-------------|
{table}

(* = QoS data may be incomplete for this API)

Top result details:
- Name: {top.name}
- Description: {top.description}
- TOPSIS Score: {top.topsis_score:.4f}
- Similarity Score: {top.similarity_score:.4f}
- Response Time: {top_rt}
- Throughput: {top_tp}
- Availability: {top_avail}

Explain in 2-3 sentences why "{top.name}" is ranked #1 considering both semantic relevance AND quality of service metrics.
If this API differs from what pure similarity would have chosen, explain the QoS tradeoff that led to the re-ranking."""
