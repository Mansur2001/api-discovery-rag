"""Streamlit app entry point. Wires all pipeline modules together."""

import traceback

import streamlit as st

from config import TOPSIS_WEIGHTS, TOPSIS_COST_CRITERIA
from data_loader import load_all_data, get_categories, build_record_maps
from vector_store import VectorStore
from ranker import rank_baseline, rank_qos_aware
from llm_provider import get_provider, fetch_available_models, LLMError, LLMAuthError
from prompts import build_baseline_prompt, build_qos_aware_prompt
from ui_components import (
    render_sidebar,
    render_side_by_side,
    render_error,
    render_no_results,
)

st.set_page_config(
    page_title="API Discovery RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading API data...")
def init_data():
    """Load both JSONL datasets. Cached across all reruns."""
    no_qos_records, with_qos_records = load_all_data()
    no_qos_map, qos_map = build_record_maps(no_qos_records, with_qos_records)
    categories = get_categories(no_qos_records)
    return no_qos_records, with_qos_records, no_qos_map, qos_map, categories


@st.cache_resource(show_spinner="Building vector index (first run only)...")
def init_vector_store(_no_qos_records):
    """Initialize ChromaDB and build index if not already persisted."""
    vs = VectorStore()
    if not vs.index_exists():
        progress_bar = st.progress(0, text="Indexing APIs into vector database...")

        def update_progress(fraction):
            progress_bar.progress(fraction, text=f"Indexing... {fraction * 100:.0f}%")

        vs.build_index(_no_qos_records, progress_callback=update_progress)
        progress_bar.empty()
    return vs


def run_pipeline(
    query: str,
    model_id: str | None,
    top_k: int,
    category_filter: str | None,
    vector_store: VectorStore,
    no_qos_map: dict,
    qos_map: dict,
) -> dict:
    """Execute the full RAG pipeline.

    Returns dict with baseline_results, baseline_explanation,
    qos_results, qos_explanation, model_id.
    """
    # Stage 2: Retrieval (shared)
    retrieval_results = vector_store.query(query, top_k, category_filter)

    if not retrieval_results:
        return {
            "baseline_results": [],
            "baseline_explanation": "",
            "qos_results": [],
            "qos_explanation": "",
            "model_id": model_id,
        }

    # Stage 3: Ranking (two parallel modes)
    baseline_results = rank_baseline(retrieval_results, no_qos_map)
    qos_results = rank_qos_aware(
        retrieval_results, qos_map, TOPSIS_WEIGHTS, TOPSIS_COST_CRITERIA
    )

    # Stage 4: LLM Generation (two calls)
    baseline_explanation = _generate_explanation(
        model_id, query, baseline_results, mode="baseline"
    )
    qos_explanation = _generate_explanation(
        model_id, query, qos_results, mode="qos"
    )

    return {
        "baseline_results": baseline_results,
        "baseline_explanation": baseline_explanation,
        "qos_results": qos_results,
        "qos_explanation": qos_explanation,
        "model_id": model_id,
    }


def _generate_explanation(
    model_id: str | None,
    query: str,
    ranked_results: list,
    mode: str,
) -> str:
    """Generate LLM explanation, returning error message on failure."""
    if model_id is None:
        return "Azure AI Foundry not configured. Set AZURE_AI_ENDPOINT and AZURE_AI_KEY."

    try:
        provider = get_provider(model_id)
        available, reason = provider.is_available()
        if not available:
            return f"Azure AI unavailable: {reason}"

        if mode == "baseline":
            prompt = build_baseline_prompt(query, ranked_results)
        else:
            prompt = build_qos_aware_prompt(query, ranked_results)

        return provider.generate(prompt)

    except LLMAuthError as e:
        return f"Azure authentication error: {e}"
    except LLMError as e:
        return f"Azure AI error: {e}"
    except Exception as e:
        return f"Error generating explanation: {e}"


def main():
    """Main Streamlit app function."""
    # Initialize data
    try:
        no_qos_records, with_qos_records, no_qos_map, qos_map, categories = (
            init_data()
        )
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except ValueError as e:
        st.error(f"Data validation error: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

    # Initialize vector store
    try:
        vector_store = init_vector_store(no_qos_records)
    except Exception as e:
        render_error(
            "Failed to initialize vector database. "
            "Try deleting the chroma_db/ directory and restarting.",
            context=traceback.format_exc(),
        )
        st.stop()

    # Fetch available models from Azure AI Foundry
    available_models = fetch_available_models()

    # Render sidebar and get user inputs
    sidebar = render_sidebar(categories, available_models)

    # Header
    st.markdown("# API Discovery Dashboard")
    st.markdown(
        "Search for APIs using natural language. "
        "Compare **Baseline** (similarity only) vs **QoS-Aware** (TOPSIS) rankings side by side."
    )
    st.markdown("---")

    # Handle search
    if sidebar["search_clicked"]:
        query = sidebar["query"].strip()
        if not query:
            st.warning("Please enter a search query.")
            return

        # Reset pagination on new search
        st.session_state.baseline_page = 0
        st.session_state.qos_page = 0

        try:
            with st.spinner("Searching and ranking APIs..."):
                results = run_pipeline(
                    query=query,
                    model_id=sidebar["model_id"],
                    top_k=sidebar["top_k"],
                    category_filter=sidebar["category"],
                    vector_store=vector_store,
                    no_qos_map=no_qos_map,
                    qos_map=qos_map,
                )

            # Stage 5: Dashboard rendering
            if not results["baseline_results"] and not results["qos_results"]:
                render_no_results()
            else:
                render_side_by_side(
                    baseline_results=results["baseline_results"],
                    baseline_explanation=results["baseline_explanation"],
                    qos_results=results["qos_results"],
                    qos_explanation=results["qos_explanation"],
                    model_id=results["model_id"],
                )

        except Exception as e:
            render_error(
                "An unexpected error occurred. Please try again.",
                context=traceback.format_exc(),
            )
    else:
        # Show welcome state
        st.markdown(
            '<div style="text-align:center;padding:60px 20px;color:#6c757d;">'
            "<h3>Enter a query in the sidebar to discover APIs</h3>"
            "<p>Try queries like: <em>weather forecast</em>, "
            "<em>image recognition</em>, <em>payment processing</em>, "
            "<em>send SMS messages</em></p></div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
