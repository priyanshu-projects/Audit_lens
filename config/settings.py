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
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"


class GeminiConfig(BaseSettings):
    """Google Gemini LLM settings."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    api_key: str = Field("", alias="GEMINI_API_KEY")
    model_name: str = Field("gemini-3.1-flash-lite", alias="GEMINI_MODEL_NAME")


class HuggingFaceConfig(BaseSettings):
    """HuggingFace Hub settings."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    token: str = Field("", alias="HF_TOKEN")
    username: str = Field("your-hf-username", alias="HF_USERNAME")
    # Using the publicly available ESG fine-tuned FinBERT model directly.
    # This model natively outputs Environmental / Social / Governance / None.
    finbert_model_id: str = Field("yiyanghkust/finbert-esg", alias="FINBERT_MODEL_ID")
    embedding_model_id: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        alias="EMBEDDING_MODEL_ID",
    )

    @property
    def finetuned_model_id(self) -> str:
        """Full HF Hub ID for a custom fine-tuned model (optional override)."""
        return f"{self.username}/finbert-esg"


class GcpConfig(BaseSettings):
    """Google Cloud Platform settings."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    project_id: str = Field("", alias="GCP_PROJECT_ID")
    region: str = Field("us-central1", alias="GCP_REGION")
    bucket: str = Field("auditlens-storage", alias="GCP_BUCKET")
    artifact_registry: str = Field("auditlens-registry", alias="GCP_ARTIFACT_REGISTRY")


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
    # How many days back to look for new filings on each daily run
    lookback_days: int = 7


class CdpConfig(BaseSettings):
    """CDP API settings (optional — Level 4 consistency check)."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    api_key: str = Field("", alias="CDP_API_KEY")
    base_url: str = "https://api.cdp.net/v0"
    enabled: bool = Field(False)   # Set True once you have CDP API access

    @property
    def is_available(self) -> bool:
        return bool(self.api_key) and self.enabled


class GitHubConfig(BaseSettings):
    """GitHub settings for retraining workflow dispatch."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    token: str = Field("", alias="GITHUB_TOKEN")
    repo: str = Field("", alias="GITHUB_REPO")
    retrain_workflow: str = "retrain.yml"


class AppConfig(BaseSettings):
    """Top-level app settings."""
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    env: str = Field("development", alias="APP_ENV")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # Drift threshold — if Evidently data drift share > this, trigger retrain
    drift_threshold: float = 0.3

    # FAISS index path (local cache; synced from GCS in production)
    faiss_index_path: Path = DATA_DIR / "knowledge_base" / "faiss_index"

    # Number of standard chunks to retrieve per claim
    rag_top_k: int = 5

    @property
    def is_production(self) -> bool:
        return self.env == "production"


# ── Convenience singleton instances ──────────────────────────────────────────
# Import these throughout the project instead of re-instantiating
gemini_cfg = GeminiConfig()
hf_cfg = HuggingFaceConfig()
gcp_cfg = GcpConfig()
edgar_cfg = EdgarConfig()
cdp_cfg = CdpConfig()
github_cfg = GitHubConfig()
app_cfg = AppConfig()
