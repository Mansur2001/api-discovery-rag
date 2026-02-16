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

# Azure AI Foundry configuration
AZURE_AI_ENDPOINT = os.getenv("AZURE_AI_ENDPOINT", "")
AZURE_AI_KEY = os.getenv("AZURE_AI_KEY", "")

# Alternative: Use Azure credential for authentication
USE_AZURE_CREDENTIAL = os.getenv("USE_AZURE_CREDENTIAL", "false").lower() == "true"


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


def get_azure_endpoint() -> str:
    """Get Azure AI endpoint from environment or Streamlit secrets."""
    value = AZURE_AI_ENDPOINT
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get("AZURE_AI_ENDPOINT", "")
    except Exception:
        return ""


def get_azure_key() -> str:
    """Get Azure AI key from environment or Streamlit secrets."""
    value = AZURE_AI_KEY
    if value:
        return value
    try:
        import streamlit as st
        return st.secrets.get("AZURE_AI_KEY", "")
    except Exception:
        return ""
