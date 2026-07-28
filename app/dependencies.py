from __future__ import annotations
import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional
import pandas as pd
from loguru import logger
from config.settings import app_cfg, gemini_cfg
_claims_df: Optional[pd.DataFrame] = None
_documents: Optional[list] = None
_rag_chain = None
_verification_pipeline = None
_finbert = None

def get_claims_df() -> pd.DataFrame:
    global _claims_df
    if _claims_df is None:
        path = Path('data/processed/claims.parquet')
        if path.exists():
            _claims_df = pd.read_parquet(path)
            logger.info(f'Loaded {len(_claims_df)} claims from {path}')
        else:
            logger.warning('claims.parquet not found — returning empty DataFrame')
            _claims_df = pd.DataFrame()
    return _claims_df

def invalidate_claims_cache() -> None:
    global _claims_df
    _claims_df = None

def get_documents() -> list:
    global _documents
    if _documents is None:
        path = Path('data/processed/documents.json')
        if path.exists():
            with open(path, encoding='utf-8') as f:
                _documents = json.load(f)
            logger.info(f'Loaded {len(_documents)} documents from {path}')
        else:
            logger.warning('documents.json not found — returning empty list')
            _documents = []
    return _documents

def invalidate_documents_cache() -> None:
    global _documents
    _documents = None

def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        try:
            from src.rag.vector_store import VectorStore
            from src.rag.rag_chain import RagChain
            index_path = app_cfg.faiss_index_path
            if index_path.exists() and (index_path / 'faiss.index').exists():
                vs = VectorStore.load(index_path)
                _rag_chain = RagChain.build(vs, api_key=gemini_cfg.api_key)
                logger.info('RAG chain initialised from FAISS index')
            else:
                logger.warning('FAISS index not found — RAG chain unavailable')
        except Exception as exc:
            logger.error(f'Failed to initialise RAG chain: {exc}')
    return _rag_chain

def invalidate_rag_cache() -> None:
    global _rag_chain
    _rag_chain = None

def get_verification_pipeline():
    global _verification_pipeline
    if _verification_pipeline is None:
        try:
            from src.consistency.verifier import VerificationPipeline
            _verification_pipeline = VerificationPipeline()
            logger.info('VerificationPipeline initialised')
        except Exception as exc:
            logger.error(f'Failed to initialise VerificationPipeline: {exc}')
    return _verification_pipeline

def get_finbert():
    global _finbert
    if _finbert is None:
        try:
            from src.classification.finbert_classifier import FinBertClassifier
            _finbert = FinBertClassifier()
            logger.info('FinBERT classifier loaded into memory')
        except Exception as exc:
            logger.error(f'Failed to load FinBERT: {exc}')
    return _finbert