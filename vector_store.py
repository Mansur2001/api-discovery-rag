"""ChromaDB vector store: index once, query per user request."""

from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME
from data_loader import APIRecord


COLLECTION_NAME = "api_discovery"
INDEX_BATCH_SIZE = 500


@dataclass
class RetrievalResult:
    api_id: str
    similarity_score: float


class VectorStore:
    """Manages ChromaDB collection for API embeddings."""

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        model_name: str = EMBEDDING_MODEL_NAME,
    ):
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def index_exists(self) -> bool:
        """Check if collection already has documents."""
        return self._collection.count() > 0

    def build_index(
        self,
        records: list[APIRecord],
        progress_callback=None,
    ) -> None:
        """Index all records into ChromaDB in batches.

        Args:
            records: List of APIRecord with embedding_text populated.
            progress_callback: Optional callable(fraction) for progress updates.
        """
        total = len(records)
        for start in range(0, total, INDEX_BATCH_SIZE):
            end = min(start + INDEX_BATCH_SIZE, total)
            batch = records[start:end]

            self._collection.add(
                ids=[r.api_id for r in batch],
                documents=[r.embedding_text for r in batch],
                metadatas=[
                    {
                        "category": r.category,
                        "name": r.name,
                        "method": r.method,
                        "has_description": bool(r.description.strip()),
                    }
                    for r in batch
                ],
            )

            if progress_callback:
                progress_callback(end / total)

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        category_filter: Optional[str] = None,
    ) -> list[RetrievalResult]:
        """Embed query and retrieve top-K most similar APIs.

        Returns list of RetrievalResult sorted by similarity descending.
        ChromaDB with cosine space returns distances in [0, 2].
        Similarity = 1 - distance.
        """
        if not query_text.strip():
            return []

        where_filter = None
        if category_filter:
            where_filter = {"category": category_filter}

        try:
            results = self._collection.query(
                query_texts=[query_text],
                n_results=top_k,
                where=where_filter,
                include=["distances"],
            )
        except Exception:
            # If category filter returns no results, ChromaDB may error
            return []

        if not results or not results["ids"] or not results["ids"][0]:
            return []

        ids = results["ids"][0]
        distances = results["distances"][0]

        retrieval_results = []
        for api_id, distance in zip(ids, distances):
            # Cosine distance in [0, 2]; similarity = 1 - distance
            similarity = max(0.0, 1.0 - distance)
            retrieval_results.append(
                RetrievalResult(api_id=api_id, similarity_score=similarity)
            )

        # Sort by similarity descending (should already be, but ensure)
        retrieval_results.sort(key=lambda r: r.similarity_score, reverse=True)
        return retrieval_results

    def reset(self) -> None:
        """Delete and recreate the collection. Use for re-indexing."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )
