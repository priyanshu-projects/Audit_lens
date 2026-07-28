"""
config/settings.py
==================
Central configuration for AuditLens using Pydantic BaseSettings.
All env vars are loaded from the .env file at project root.
"""

from __future__ import annotations

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Project paths ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


class GeminiConfig(BaseSettings):
    """Google Gemini LLM settings."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    api_key: str = Field("", alias="GEMINI_API_KEY")
    model_name: str = Field("gemini-2.0-flash", alias="GEMINI_MODEL_NAME")


class HuggingFaceConfig(BaseSettings):
    """HuggingFace Hub settings."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    finbert_model_id: str = Field("yiyanghkust/finbert-esg", alias="FINBERT_MODEL_ID")
    embedding_model_id: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_ID",
    )


class EdgarConfig(BaseSettings):
    """SEC EDGAR API settings."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    user_agent: str = Field(
        "AuditLens auditlens@example.com",
        alias="EDGAR_USER_AGENT",
    )
    base_url: str = "https://data.sec.gov"
    submissions_url: str = "https://data.sec.gov/submissions"
    search_url: str = "https://efts.sec.gov/LATEST/search-index"


class AppConfig(BaseSettings):
    """Top-level app settings."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    env: str = Field("development", alias="APP_ENV")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # FAISS index directory — baked into Docker image in production
    faiss_index_path: Path = DATA_DIR / "knowledge_base"

    # Number of standard chunks to retrieve per claim
    rag_top_k: int = 5

    @property
    def is_production(self) -> bool:
        return self.env == "production"


# ── Convenience singleton instances ──────────────────────────────────────────
gemini_cfg = GeminiConfig()
hf_cfg = HuggingFaceConfig()
edgar_cfg = EdgarConfig()
app_cfg = AppConfig()
