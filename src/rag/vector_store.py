"""
src/rag/vector_store.py
========================
FAISS-based vector store for GRI/TCFD/SASB/ISSB standard document chunks.

The knowledge base is built ONCE from PDFs in data/knowledge_base/
and saved to disk. In production (Cloud Run), it's loaded from GCS.

Chunking strategy:
  - Each standard PDF is split into ~300 token chunks with 50-token overlap
  - Each chunk stores: text, source document, section, page number

Usage:
    store = VectorStore()
    store.build_from_directory(Path("data/knowledge_base"))
    store.save(Path("data/knowledge_base/faiss_index"))

    # Later / in production:
    store = VectorStore.load(Path("data/knowledge_base/faiss_index"))
    results = store.search("Scope 1 GHG emission disclosure requirements", top_k=5)
"""

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


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """A single chunk of a regulatory standard document."""
    text: str
    source_doc: str       # e.g. "GRI_305_Emissions.pdf"
    standard: str         # e.g. "GRI", "TCFD", "SASB", "ISSB", "CSRD"
    section: str          # e.g. "Disclosure 305-1"
    page_number: int = 0
    chunk_id: int = 0


@dataclass
class SearchResult:
    """A single search result from the vector store."""
    chunk: DocumentChunk
    score: float          # cosine similarity (higher = more relevant)
    rank: int


# ── Chunking config ───────────────────────────────────────────────────────────

CHUNK_SIZE_TOKENS = 300      # approximate tokens per chunk
CHUNK_OVERLAP_TOKENS = 50    # overlap to avoid losing context at boundaries
CHARS_PER_TOKEN = 4          # rough estimate for character→token conversion

CHUNK_SIZE_CHARS = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN       # 1200
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN  # 200

# Map filename patterns to standard labels
STANDARD_DETECTION = {
    "gri": "GRI",
    "tcfd": "TCFD",
    "sasb": "SASB",
    "issb": "ISSB",
    "ifrs": "ISSB",
    "csrd": "CSRD",
    "eu_": "CSRD",
}


# ── Main class ────────────────────────────────────────────────────────────────

class VectorStore:
    """
    FAISS vector store for regulatory standard document chunks.

    Index type: IndexFlatIP (inner product = cosine similarity when vectors
    are L2-normalised, which Embedder does by default).

    For ~500 chunks, flat index is fast enough (< 5ms per query on CPU).
    No need for HNSW or IVF at this scale.
    """

    def __init__(self) -> None:
        self.embedder = Embedder.get_instance()
        self.index: Optional[faiss.Index] = None
        self.chunks: list[DocumentChunk] = []
        self._is_built = False

    # ── Build ──────────────────────────────────────────────────────────────

    def build_from_directory(self, knowledge_base_dir: Path) -> None:
        """
        Build the FAISS index from all PDFs in the knowledge base directory.

        Args:
            knowledge_base_dir: Path to data/knowledge_base/
        """
        pdf_files = list(knowledge_base_dir.glob("*.pdf"))
        if not pdf_files:
            logger.warning(
                f"No PDFs found in {knowledge_base_dir}. "
                "Download GRI/TCFD/SASB/ISSB PDFs first — see README for links."
            )
            return

        logger.info(f"Building FAISS index from {len(pdf_files)} PDFs")
        all_chunks: list[DocumentChunk] = []

        for pdf_path in pdf_files:
            chunks = self._chunk_pdf(pdf_path)
            all_chunks.extend(chunks)
            logger.info(f"  {pdf_path.name}: {len(chunks)} chunks")

        if not all_chunks:
            logger.error("No text extracted from any PDF — check files are not scanned images")
            return

        self._build_index(all_chunks)
        logger.success(
            f"FAISS index built: {len(all_chunks)} chunks | "
            f"dim={self.embedder.embedding_dim}"
        )

    def build_from_texts(
        self,
        texts: list[str],
        metadatas: list[dict] | None = None,
    ) -> None:
        """
        Build index directly from a list of text strings.
        Useful for testing without PDFs.
        """
        metadatas = metadatas or [{}] * len(texts)
        chunks = [
            DocumentChunk(
                text=text,
                source_doc=meta.get("source_doc", "unknown"),
                standard=meta.get("standard", "UNKNOWN"),
                section=meta.get("section", ""),
                page_number=meta.get("page_number", 0),
                chunk_id=i,
            )
            for i, (text, meta) in enumerate(zip(texts, metadatas))
        ]
        self._build_index(chunks)

    # ── Persist ────────────────────────────────────────────────────────────

    def save(self, index_dir: Path) -> None:
        """
        Save the FAISS index and chunk metadata to disk.

        Saves two files:
          - index_dir/faiss.index   (FAISS binary)
          - index_dir/chunks.json   (chunk metadata)
        """
        if not self._is_built:
            raise RuntimeError("No index to save — call build_from_directory() first")

        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / "faiss.index"))

        chunks_data = [asdict(c) for c in self.chunks]
        with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)

        logger.success(f"FAISS index saved to {index_dir} ({len(self.chunks)} chunks)")

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        """
        Load a previously saved FAISS index from disk.

        Args:
            index_dir: Directory containing faiss.index + chunks.json
        """
        store = cls()

        faiss_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.json"

        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {faiss_path}. "
                "Run vector_store.build_from_directory() first."
            )

        store.index = faiss.read_index(str(faiss_path))

        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        store.chunks = [DocumentChunk(**c) for c in chunks_data]
        store._is_built = True

        logger.success(
            f"FAISS index loaded from {index_dir} | "
            f"{len(store.chunks)} chunks | "
            f"dim={store.index.d}"
        )
        return store

    # ── Search ─────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        standard_filter: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic search over the knowledge base.

        Args:
            query:           Natural language query or ESG claim text
            top_k:           Number of results to return
            standard_filter: Optional — restrict to one standard ("GRI", "TCFD", etc.)

        Returns:
            List of SearchResult sorted by relevance (highest score first)
        """
        if not self._is_built or self.index is None:
            logger.error("Vector store not built — call build_from_directory() or load() first")
            return []

        query_vec = self.embedder.embed_query(query)   # shape (1, dim)

        # Search more candidates if filtering by standard
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

    # ── Private helpers ────────────────────────────────────────────────────

    def _chunk_pdf(self, pdf_path: Path) -> list[DocumentChunk]:
        """Extract and chunk a single PDF into DocumentChunk objects."""
        standard = self._detect_standard(pdf_path.name)
        chunks: list[DocumentChunk] = []
        full_text_by_page: list[tuple[int, str]] = []

        try:
            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    if text.strip():
                        full_text_by_page.append((page_num, text))
        except Exception as exc:
            logger.error(f"Failed to extract {pdf_path.name}: {exc}")
            return chunks

        chunk_id = 0
        for page_num, page_text in full_text_by_page:
            # Detect section from first line
            first_line = page_text.split("\n")[0].strip()
            section = first_line[:80] if first_line else f"Page {page_num}"

            # Slide window across the page text
            start = 0
            while start < len(page_text):
                end = min(start + CHUNK_SIZE_CHARS, len(page_text))
                chunk_text = page_text[start:end].strip()

                if len(chunk_text) > 50:   # skip tiny chunks
                    chunks.append(DocumentChunk(
                        text=chunk_text,
                        source_doc=pdf_path.name,
                        standard=standard,
                        section=section,
                        page_number=page_num,
                        chunk_id=chunk_id,
                    ))
                    chunk_id += 1

                if end >= len(page_text):
                    break
                start = end - CHUNK_OVERLAP_CHARS

        return chunks

    def _build_index(self, chunks: list[DocumentChunk]) -> None:
        """Embed all chunks and build the FAISS index."""
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_texts(texts, show_progress=True)

        dim = embeddings.shape[1]
        # IndexFlatIP: exact search, inner product (= cosine for L2-normalised vecs)
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks
        self._is_built = True

    @staticmethod
    def _detect_standard(filename: str) -> str:
        """Detect which ESG standard a PDF belongs to from its filename."""
        lower = filename.lower()
        for keyword, standard in STANDARD_DETECTION.items():
            if keyword in lower:
                return standard
        return "UNKNOWN"
