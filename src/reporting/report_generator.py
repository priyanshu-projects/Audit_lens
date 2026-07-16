"""
src/reporting/report_generator.py
==================================
Generates structured audit observations and compiles the final audit report
(JSON and Markdown formats) from verification results.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

from src.extraction.claim_detector import Claim
from src.rag.rag_chain import RagChain
from src.rag.vector_store import VectorStore
from config.settings import app_cfg, gemini_cfg


class ReportGenerator:
    """
    Orchestrates the generation of structured audit observations via RAG and Gemini
    and writes out the consolidated audit report.
    """

    def __init__(self, rag_chain: Optional[RagChain] = None) -> None:
        self.rag_chain = rag_chain or self._initialize_default_rag_chain()

    def _initialize_default_rag_chain(self) -> Optional[RagChain]:
        """Load vector store and initialize the default RAG chain from config settings."""
        try:
            index_path = app_cfg.faiss_index_path
            if index_path.exists() and (index_path / "faiss.index").exists():
                vector_store = VectorStore.load(index_path)
                return RagChain.build(vector_store, api_key=gemini_cfg.api_key)
            else:
                return None
        except Exception:
            return None

    def generate(
        self,
        verification_results: list[dict],
        output_json_path: str | Path,
        output_md_path: Optional[str | Path] = None,
    ) -> list[dict]:
        """
        Generate RAG observations for high-risk/inconsistent claims and compile the final report.

        Args:
            verification_results: List of verification result dicts from VerificationPipeline.
            output_json_path: Path to write the final compiled JSON report.
            output_md_path: Optional path to write a readable Markdown report summary.

        Returns:
            The list of results with populated audit observation data.
        """
        final_results = []

        for item in verification_results:
            claim: Claim = item["claim"]
            verdict = item["verdict"]

            # Initialize empty observation dictionary
            obs_data = None

            # 1. Check if observation already exists in the item
            existing_obs = item.get("audit_observation")
            if existing_obs:
                if hasattr(existing_obs, "to_dict"):
                    obs_data = existing_obs.to_dict()
                elif isinstance(existing_obs, dict):
                    obs_data = existing_obs
                else:
                    obs_data = {
                        "structured_note": getattr(existing_obs, "structured_note", ""),
                        "standards_cited": getattr(existing_obs, "standards_cited", []),
                        "retrieval_confidence": getattr(existing_obs, "retrieval_confidence", 0.0),
                    }
            # 2. Generate RAG audit observation for flagged claims if not already present
            elif verdict in {"UNSUPPORTED", "INCONSISTENT", "HIGH_RISK", "PARTIALLY_CONSISTENT"}:
                if self.rag_chain:
                    try:
                        # surrounding context is in 'evidence' field
                        context_window = getattr(claim, "evidence", claim.text)
                        obs = self.rag_chain.generate_audit_observation(
                            claim=claim.text,
                            shap_narrative=context_window or "No SHAP explanation available (generated in batch).",
                        )
                        obs_data = {
                            "structured_note": obs.structured_note,
                            "standards_cited": obs.standards_cited,
                            "retrieval_confidence": obs.retrieval_confidence,
                        }
                    except Exception as e:
                        obs_data = {
                            "structured_note": f"⚠️ Error generating audit observation: {e}",
                            "standards_cited": [],
                            "retrieval_confidence": "low",
                        }
                else:
                    obs_data = {
                        "structured_note": "⚠️ RAG chain not available. Ensure FAISS index exists.",
                        "standards_cited": [],
                        "retrieval_confidence": "low",
                    }


            # Build serializable version of the result
            serialized_item = {
                "claim_id": getattr(claim, "claim_id", None),
                "claim_text": claim.text,
                "source_section": claim.source_section,
                "esg_label": claim.esg_label,
                "l1_status": item["l1"].status,
                "l1_note": item["l1"].note,
                "l2_status": item["l2"].status,
                "l2_note": item["l2"].note,
                "l3_status": item["l3"].status,
                "l3_note": item["l3"].note,
                "risk_score": item["risk_score"],
                "verdict": verdict,
                "final_note": item["final_note"],
                "audit_observation": obs_data,
            }
            final_results.append(serialized_item)

        # Write to JSON
        output_json_path = Path(output_json_path)
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(final_results, f, indent=2, ensure_ascii=False)

        # Write to MD if requested
        if output_md_path:
            self._write_markdown_report(final_results, output_md_path)

        return final_results

    def _write_markdown_report(self, results: list[dict], output_md_path: str | Path) -> None:
        """Create a beautiful human-readable markdown report of the audit."""
        output_md_path = Path(output_md_path)
        output_md_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# 🧾 ESG Audit Observation Report",
            "Generated automatically by the AuditLens MLOps pipeline.",
            "",
            "## 📊 Executive Summary Table",
            "",
            "| ID | ESG Label | Verdict | Risk Score | Claim Text |",
            "|---|---|---|---|---|",
        ]

        for item in results:
            lines.append(
                f"| {item['claim_id']} | {item['esg_label']} | **{item['verdict']}** | {item['risk_score']:.2f} | {item['claim_text']} |"
            )

        lines.append("")
        lines.append("## 🔍 Detailed Observations")
        lines.append("")

        flagged_items = [r for r in results if r["audit_observation"] is not None]
        if not flagged_items:
            lines.append("*No claims were flagged for audit review. All claims are Consistent.*")
        else:
            for item in flagged_items:
                obs = item["audit_observation"]
                lines.append(f"### Claim {item['claim_id']}: \"{item['claim_text']}\"")
                lines.append(f"- **Section**: `{item['source_section']}`")
                lines.append(f"- **Verdict**: `{item['verdict']}` (Risk: `{item['risk_score']:.2f}`)")
                lines.append(f"- **L1 Check**: `{item['l1_status']}`")
                lines.append(f"- **L2 Check**: `{item['l2_status']}`")
                lines.append(f"- **L3 Check**: `{item['l3_status']}`")
                lines.append("")
                lines.append("**Audit Note:**")
                lines.append(obs["structured_note"])
                lines.append("")
                lines.append("**Standards Cited:**")
                if obs["standards_cited"]:
                    for std in obs["standards_cited"]:
                        lines.append(f"- {std}")
                else:
                    lines.append("- *None cited*")
                lines.append("")
                lines.append("---")

        with open(output_md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
