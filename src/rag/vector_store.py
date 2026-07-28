from __future__ import annotations
import json
import pickle
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import faiss
import numpy as np
import pdfplumber
from loguru import logger
from src.rag.embedder import Embedder

@dataclass
class DocumentChunk:
    text: str
    source_doc: str
    standard: str
    section: str
    page_number: int = 0
    chunk_id: int = 0

@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float
    rank: int
CHUNK_SIZE_TOKENS = 300
CHUNK_OVERLAP_TOKENS = 50
CHARS_PER_TOKEN = 4
CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN
STANDARD_DETECTION = {'gri': 'GRI', 'tcfd': 'TCFD', 'sasb': 'SASB', 'issb': 'ISSB', 'ifrs': 'ISSB', 'csrd': 'CSRD', 'eu_': 'CSRD'}

class VectorStore:

    def __init__(self) -> None:
        self.embedder = Embedder.get_instance()
        self.index: Optional[faiss.Index] = None
        self.chunks: list[DocumentChunk] = []
        self._is_built = False

    def build_from_directory(self, knowledge_base_dir: Path) -> None:
        pdf_files = list(knowledge_base_dir.glob('*.pdf'))
        if not pdf_files:
            logger.warning(f'No PDFs found in {knowledge_base_dir}. Download GRI/TCFD/SASB/ISSB PDFs first — see README for links.')
            return
        logger.info(f'Building FAISS index from {len(pdf_files)} PDFs')
        all_chunks: list[DocumentChunk] = []
        for pdf_path in pdf_files:
            chunks = self._chunk_pdf(pdf_path)
            all_chunks.extend(chunks)
            logger.info(f'  {pdf_path.name}: {len(chunks)} chunks')
        if not all_chunks:
            logger.error('No text extracted from any PDF — check files are not scanned images')
            return
        self._build_index(all_chunks)
        logger.success(f'FAISS index built: {len(all_chunks)} chunks | dim={self.embedder.embedding_dim}')

    def build_from_texts(self, texts: list[str], metadatas: list[dict] | None=None) -> None:
        metadatas = metadatas or [{}] * len(texts)
        chunks = [DocumentChunk(text=text, source_doc=meta.get('source_doc', 'unknown'), standard=meta.get('standard', 'UNKNOWN'), section=meta.get('section', ''), page_number=meta.get('page_number', 0), chunk_id=i) for i, (text, meta) in enumerate(zip(texts, metadatas))]
        self._build_index(chunks)

    def save(self, index_dir: Path) -> None:
        if not self._is_built:
            raise RuntimeError('No index to save — call build_from_directory() first')
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / 'faiss.index'))
        chunks_data = [asdict(c) for c in self.chunks]
        with open(index_dir / 'chunks.json', 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        logger.success(f'FAISS index saved to {index_dir} ({len(self.chunks)} chunks)')

    @classmethod
    def load(cls, index_dir: Path) -> 'VectorStore':
        store = cls()
        faiss_path = index_dir / 'faiss.index'
        chunks_path = index_dir / 'chunks.json'
        if not faiss_path.exists():
            raise FileNotFoundError(f'FAISS index not found at {faiss_path}. Run vector_store.build_from_directory() first.')
        store.index = faiss.read_index(str(faiss_path))
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        store.chunks = [DocumentChunk(**c) for c in chunks_data]
        store._is_built = True
        logger.success(f'FAISS index loaded from {index_dir} | {len(store.chunks)} chunks | dim={store.index.d}')
        return store

    def search(self, query: str, top_k: int=5, standard_filter: Optional[str]=None) -> list[SearchResult]:
        if not self._is_built or self.index is None:
            logger.error('Vector store not built — call build_from_directory() or load() first')
            return []
        query_vec = self.embedder.embed_query(query)
        search_k = top_k * 5 if standard_filter else top_k
        scores, indices = self.index.search(query_vec, search_k)
        results: list[SearchResult] = []
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if standard_filter and chunk.standard != standard_filter.upper():
                continue
            results.append(SearchResult(chunk=chunk, score=float(score), rank=rank + 1))
            if len(results) >= top_k:
                break
        logger.debug(f"FAISS search returned {len(results)} results for query: '{query[:60]}...'")
        return results

    def _chunk_pdf(self, pdf_path: Path) -> list[DocumentChunk]:
        standard = self._detect_standard(pdf_path.name)
        chunks: list[DocumentChunk] = []
        full_text_by_page: list[tuple[int, str]] = []
        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ''
                    if text.strip():
                        full_text_by_page.append((page_num, text))
        except Exception as exc:
            logger.error(f'Failed to extract {pdf_path.name}: {exc}')
            return chunks
        chunk_id = 0
        for page_num, page_text in full_text_by_page:
            first_line = page_text.split('\n')[0].strip()
            section = first_line[:80] if first_line else f'Page {page_num}'
            start = 0
            while start < len(page_text):
                end = min(start + CHUNK_SIZE_CHARS, len(page_text))
                chunk_text = page_text[start:end].strip()
                if len(chunk_text) > 50:
                    chunks.append(DocumentChunk(text=chunk_text, source_doc=pdf_path.name, standard=standard, section=section, page_number=page_num, chunk_id=chunk_id))
                    chunk_id += 1
                if end >= len(page_text):
                    break
                start = end - CHUNK_OVERLAP_CHARS
        return chunks

    def _build_index(self, chunks: list[DocumentChunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts, show_progress=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks
        self._is_built = True

    @staticmethod
    def _detect_standard(filename: str) -> str:
        lower = filename.lower()
        for keyword, standard in STANDARD_DETECTION.items():
            if keyword in lower:
                return standard
        return 'UNKNOWN'