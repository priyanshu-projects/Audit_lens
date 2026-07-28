from __future__ import annotations
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from config.settings import hf_cfg

class Embedder:
    _instance: 'Embedder | None' = None

    def __init__(self, model_id: str=hf_cfg.embedding_model_id) -> None:
        logger.info(f'Loading embedding model: {model_id}')
        self.model = SentenceTransformer(model_id)
        self.model_id = model_id
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.success(f'Embedding model loaded | dim={self.embedding_dim} | model={model_id}')

    @classmethod
    def get_instance(cls) -> 'Embedder':
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def embed_texts(self, texts: list[str], batch_size: int=32, show_progress: bool=False) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)
        logger.info(f'Embedding {len(texts)} texts (batch_size={batch_size})')
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=show_progress, convert_to_numpy=True, normalize_embeddings=True)
        logger.debug(f'Embedding shape: {embeddings.shape}')
        return embeddings.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])