"""Centralized configuration. Constants, paths, and environment loading."""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
NO_QOS_PATH = DATA_DIR / "api_repo.no_qos.jsonl"
WITH_QOS_PATH = DATA_DIR / "api_repo.with_qos.jsonl"
CHROMA_PERSIST_DIR = str(PROJECT_ROOT / "chroma_db")

# Embedding model (sentence-transformers, runs locally)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Default retrieval/ranking settings
DEFAULT_TOP_K = 10
MAX_TOP_K = 50
MIN_TOP_K = 3

# TOPSIS weights (sum to 1.0)
TOPSIS_WEIGHTS = {
    "rt_ms": 0.25,
    "tp_rps": 0.25,
    "availability": 0.30,
    "similarity": 0.20,
}
TOPSIS_COST_CRITERIA = {"rt_ms"}

# LLM configuration
LLM_MODELS = {
    "GPT-4o": {
        "provider": "openai",
        "model_id": "gpt-4o",
        "env_key": "OPENAI_API_KEY",
    },
    "Claude 3.5 Sonnet": {
        "provider": "anthropic",
        "model_id": "claude-3-5-sonnet-20241022",
        "env_key": "ANTHROPIC_API_KEY",
    },
    "Gemini 1.5 Pro": {
        "provider": "google",
        "model_id": "gemini-1.5-pro",
        "env_key": "GOOGLE_API_KEY",
    },
    "Llama 3 (Local)": {
        "provider": "ollama",
        "model_id": "llama3",
        "env_key": None,
    },
}


def get_api_key(env_key: str) -> str:
    """Get API key from environment or Streamlit secrets."""
    value = os.getenv(env_key, "")
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get(env_key, "")
    except Exception:
        return ""
