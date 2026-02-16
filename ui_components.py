"""Reusable Streamlit UI components for the API Discovery dashboard."""

import streamlit as st

from config import DEFAULT_TOP_K, MIN_TOP_K, MAX_TOP_K
from ranker import BaselineRankedResult, QoSRankedResult
from llm_provider import get_provider, check_azure_availability


# Method badge colors
METHOD_COLORS = {
    "GET": "#28a745",
    "POST": "#007bff",
    "PUT": "#fd7e14",
    "PATCH": "#6f42c1",
    "DELETE": "#dc3545",
}


def render_sidebar(categories: list[str], available_models: list[dict]) -> dict:
    """Render the sidebar with all controls.

    Args:
        categories: List of API categories for filtering
        available_models: List of model dicts from Azure AI Foundry

    Returns dict with: query, model_id, top_k, category, search_clicked
    """
    with st.sidebar:
        st.title("API Discovery RAG")
        st.markdown("---")

        query = st.text_input(
            "Search for APIs",
            placeholder="e.g., weather forecast API",
        )

        st.markdown("#### LLM Model")

        # Check Azure availability
        azure_available, azure_reason = check_azure_availability()
        if not azure_available:
            st.warning(f"Azure AI Foundry: {azure_reason}")
            model_id = None
        elif not available_models:
            st.warning("No models available from Azure AI Foundry")
            model_id = None
        else:
            # Create model options
            model_options = {m["display_name"]: m["model_id"] for m in available_models}
            selected_display_name = st.selectbox(
                "Select Model",
                options=list(model_options.keys()),
                label_visibility="collapsed",
            )
            model_id = model_options[selected_display_name]

            # Show availability indicator
            st.markdown("##### Azure AI Status")
            st.markdown(f"&nbsp; :green_circle: &nbsp; Connected", unsafe_allow_html=True)
            st.caption(f"Model: `{model_id}`")

        st.markdown("---")

        top_k = st.slider(
            "Number of results (Top-K)",
            min_value=MIN_TOP_K,
            max_value=MAX_TOP_K,
            value=DEFAULT_TOP_K,
        )

        category_options = ["All Categories"] + categories
        category = st.selectbox("Filter by Category", options=category_options)
        if category == "All Categories":
            category = None

        st.markdown("---")
        search_clicked = st.button("Search", type="primary", use_container_width=True)

    return {
        "query": query,
        "model_id": model_id,
        "top_k": top_k,
        "category": category,
        "search_clicked": search_clicked,
    }


def _method_badge(method: str) -> str:
    """Return an HTML badge for the HTTP method."""
    color = METHOD_COLORS.get(method, "#6c757d")
    return (
        f'<span style="background-color:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:0.75em;font-weight:bold;">{method}</span>'
    )


def _score_bar(score: float, label: str, max_val: float = 1.0) -> str:
    """Return an HTML progress bar for a score."""
    pct = min(100, max(0, (score / max_val) * 100))
    return (
        f'<div style="margin:2px 0;">'
        f'<span style="font-size:0.8em;">{label}: {score:.4f}</span>'
        f'<div style="background:#e9ecef;border-radius:4px;height:8px;margin-top:2px;">'
        f'<div style="background:#007bff;width:{pct:.0f}%;height:100%;border-radius:4px;">'
        f"</div></div></div>"
    )


def render_baseline_card(result: BaselineRankedResult) -> None:
    """Render a single baseline result card."""
    with st.container():
        st.markdown(
            f"**#{result.rank}** &nbsp; {result.name} &nbsp; "
            f"{_method_badge(result.method)} &nbsp; "
            f'<span style="background:#e9ecef;padding:2px 8px;border-radius:4px;'
            f'font-size:0.75em;">{result.category}</span>',
            unsafe_allow_html=True,
        )
        st.markdown(_score_bar(result.similarity_score, "Similarity"), unsafe_allow_html=True)

        with st.expander("Details"):
            st.markdown(f"**Description:** {result.description}")
            st.markdown(f"**URL:** `{result.url}`")
            st.markdown(f"**API ID:** `{result.api_id}`")

        st.markdown("<hr style='margin:8px 0;border:none;border-top:1px solid #eee;'>", unsafe_allow_html=True)


def render_qos_card(result: QoSRankedResult) -> None:
    """Render a single QoS-aware result card."""
    with st.container():
        st.markdown(
            f"**#{result.rank}** &nbsp; {result.name} &nbsp; "
            f"{_method_badge(result.method)} &nbsp; "
            f'<span style="background:#e9ecef;padding:2px 8px;border-radius:4px;'
            f'font-size:0.75em;">{result.category}</span>',
            unsafe_allow_html=True,
        )

        st.markdown(_score_bar(result.topsis_score, "TOPSIS Score"), unsafe_allow_html=True)
        st.markdown(_score_bar(result.similarity_score, "Similarity"), unsafe_allow_html=True)

        # QoS metrics in columns
        c1, c2, c3 = st.columns(3)
        with c1:
            if result.rt_ms is not None:
                st.metric("Latency", f"{result.rt_ms:.2f} ms")
            else:
                st.metric("Latency", "N/A")
        with c2:
            if result.tp_rps is not None:
                st.metric("Throughput", f"{result.tp_rps:.2f} rps")
            else:
                st.metric("Throughput", "N/A")
        with c3:
            if result.availability is not None:
                st.metric("Availability", f"{result.availability * 100:.1f}%")
            else:
                st.metric("Availability", "N/A")

        if not result.valid_qos:
            st.caption("*QoS data may be incomplete for this API")

        with st.expander("Details"):
            st.markdown(f"**Description:** {result.description}")
            st.markdown(f"**URL:** `{result.url}`")
            st.markdown(f"**API ID:** `{result.api_id}`")

        st.markdown("<hr style='margin:8px 0;border:none;border-top:1px solid #eee;'>", unsafe_allow_html=True)


def render_llm_explanation(explanation: str, model_id: str, mode: str) -> None:
    """Render the LLM explanation in a styled container."""
    title = "Baseline Analysis" if mode == "baseline" else "QoS-Aware Analysis"

    st.markdown(f"### {title}")
    st.markdown(
        f'<span style="background:#f0f0f0;padding:4px 12px;border-radius:12px;'
        f'font-size:0.8em;">Powered by {model_id}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")

    if explanation.startswith("Error:") or explanation.startswith("LLM") or explanation.startswith("Azure"):
        st.warning(explanation)
    else:
        st.info(explanation)


def render_side_by_side(
    baseline_results: list[BaselineRankedResult],
    baseline_explanation: str,
    qos_results: list[QoSRankedResult],
    qos_explanation: str,
    model_id: str,
) -> None:
    """Main layout: two columns side by side."""
    col1, col2 = st.columns(2)

    with col1:
        render_llm_explanation(baseline_explanation, model_id, "baseline")
        st.markdown("---")
        st.markdown("#### Ranked by Similarity")
        for result in baseline_results:
            render_baseline_card(result)

    with col2:
        render_llm_explanation(qos_explanation, model_id, "qos")
        st.markdown("---")
        st.markdown("#### Ranked by TOPSIS (QoS-Aware)")
        for result in qos_results:
            render_qos_card(result)

    # Active LLM indicator
    st.markdown("---")
    st.markdown(
        f'<div style="text-align:center;padding:8px;background:#f8f9fa;'
        f'border-radius:8px;">'
        f'<strong>Active Model:</strong> {model_id}</div>',
        unsafe_allow_html=True,
    )


def render_error(message: str, context: str = "") -> None:
    """Show an error message with optional technical context."""
    st.error(message)
    if context:
        with st.expander("Technical Details"):
            st.code(context)


def render_no_results() -> None:
    """Show a message when no results are found."""
    st.info("No APIs found matching your query. Try a different search term or remove the category filter.")
