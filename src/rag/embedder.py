"""
src/rag/embedder.py
====================
Text embedding using sentence-transformers (all-MiniLM-L6-v2).

This model runs on CPU — no GPU needed.
384-dimensional embeddings, fast and lightweight.

Used for:
  - Indexing GRI/TCFD/SASB/ISSB standard document chunks into FAISS
  - Embedding user claims at query time for semantic search

Usage:
    embedder = Embedder()
    vecs = embedder.embed_texts(["We reduced emissions by 40%."])
    query_vec = embedder.embed_query("carbon emission reduction target")
"""

from __future__ import annotations

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import hf_cfg


class Embedder:
    """
    Wraps sentence-transformers for consistent text embedding.
    Singleton pattern: call Embedder.get_instance() to reuse the loaded model.
    """

    _instance: "Embedder | None" = None

    def __init__(self, model_id: str = hf_cfg.embedding_model_id) -> None:
        logger.info(f"Loading embedding model: {model_id}")
        self.model = SentenceTransformer(model_id)
        self.model_id = model_id
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.success(
            f"Embedding model loaded | dim={self.embedding_dim} | model={model_id}"
        )

    @classmethod
    def get_instance(cls) -> "Embedder":
        """Return a shared Embedder instance (lazy singleton)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_texts(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Embed a list of texts.

        Args:
            texts:         List of strings to embed
            batch_size:    Sentences per batch (larger = faster but more RAM)
            show_progress: Show tqdm progress bar

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        logger.info(f"Embedding {len(texts)} texts (batch_size={batch_size})")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,   # L2-normalise for cosine similarity via dot product
        )
        logger.debug(f"Embedding shape: {embeddings.shape}")
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string.

        Returns:
            numpy array of shape (1, embedding_dim)
        """
        return self.embed_texts([query])
