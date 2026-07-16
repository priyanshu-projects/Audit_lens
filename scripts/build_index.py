"""
scripts/build_index.py
========================
One-time script to build the FAISS vector index from regulatory standard PDFs.

Run this ONCE after downloading GRI/TCFD/SASB/ISSB PDFs into data/knowledge_base/.

Usage:
    python scripts/build_index.py

Downloads to trigger (free, public):
  GRI 305: https://www.globalreporting.org/standards/standards-development/universal-standards/
  TCFD:    https://www.fsb-tcfd.org/publications/
  SASB:    https://sasb.org/standards/download/
  ISSB:    https://www.ifrs.org/groups/international-sustainability-standards-board/
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.rag.vector_store import VectorStore
from config.settings import app_cfg, DATA_DIR


def main():
    kb_dir = DATA_DIR / "knowledge_base"
    index_path = app_cfg.faiss_index_path

    pdf_files = list(kb_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(
            f"No PDFs found in {kb_dir}.\n"
            "Please download regulatory standard PDFs:\n"
            "  GRI 305: https://www.globalreporting.org/standards/\n"
            "  TCFD:    https://www.fsb-tcfd.org/publications/\n"
            "  SASB:    https://sasb.org/standards/download/\n"
            "  ISSB:    https://www.ifrs.org/groups/international-sustainability-standards-board/"
        )
        return

    logger.info(f"Building FAISS index from {len(pdf_files)} PDFs in {kb_dir}")
    store = VectorStore()
    store.build_from_directory(kb_dir)
    store.save(index_path)
    logger.success(f"Index built and saved to {index_path}")
    logger.info("You can now run: streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    main()
