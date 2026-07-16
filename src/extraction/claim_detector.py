"""
src/extraction/claim_detector.py
==================================
Detects verifiable ESG claim sentences from extracted text.

A "claim" in the ESG audit context is a sentence that:
  - Makes a quantitative assertion about an ESG metric (e.g. "reduced emissions by 40%")
  - States a forward ESG commitment (e.g. "target net-zero by 2040")
  - Makes a comparative ESG statement (e.g. "lower than industry average for emissions")
  - References ESG certifications/standards (e.g. "GRI certified", "ISO 14001")

Critically, pure financial performance statements (revenue, margins, EPS, net sales)
are NOT ESG claims and are explicitly filtered out.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

from loguru import logger
from src.classification.finbert_classifier import FinBertClassifier
from langchain_core.language_models import BaseLanguageModel

# spaCy is loaded lazily to keep import time low
_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
            _NLP = spacy.blank("en")
            _NLP.add_pipe("sentencizer")
    return _NLP


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class Claim:
    """A single detected ESG claim sentence."""
    text: str                         # The clean, atomic ESG claim
    start_char: int
    end_char: int
    evidence_sentence: str = ""       # The exact sentence containing the claim
    evidence: str = ""                # surrounding ±1 sentence context/evidence
    source_section: str = "general"
    claim_type: str = "quantitative"
    confidence: float = 0.5
    esg_label: str = "MIXED"
    matched_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "evidence_sentence": self.evidence_sentence,
            "evidence": self.evidence,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "source_section": self.source_section,
            "claim_type": self.claim_type,
            "confidence": self.confidence,
            "matched_patterns": self.matched_patterns,
        }


# ── BLOCKLIST: sentences matching any of these are NEVER ESG claims ──────────
# These catch financial performance statements, legal boilerplate, accounting notes

_FINANCIAL_BLOCKLIST: list[str] = [
    # Revenue / sales language
    r"\b(?:net sales|gross revenue|total revenue|net revenue|product revenue|service revenue)\b",
    r"\b(?:iPhone|Mac|iPad|Apple Watch|AirPods|HomePod|Vision Pro)\b.{0,60}(?:net sales|revenue|increased|decreased)",
    r"\b(?:Americas|Europe|Greater China|Japan|Asia Pacific)\b.{0,60}(?:net sales|revenue|increased|decreased)",
    r"\bsegment.{0,30}(?:net sales|revenue)\b",
    # Profit / margin financial language
    r"\b(?:gross margin|operating margin|net income|earnings per share|EPS|diluted EPS)\b",
    r"\b(?:operating income|operating expense|operating profit|net profit)\b",
    r"\b(?:selling, general and administrative|SG&A|R&D expense|research and development expense)\b",
    # Tax / accounting language
    r"\beffective tax rate\b",
    r"\bdeferred tax\b",
    r"\b(?:fiscal year|quarterly results|annual results)\b.{0,40}(?:increased|decreased|grew|declined)",
    # Legal / SEC boilerplate
    r"\b(?:Rule\s+\d+[a-z]?-\d+|15d-14|13a-14)\b",
    r"\bCertification of Chief (?:Executive|Financial) Officer\b",
    r"\bPursuant to (?:Section|Rule)\b",
    r"\bSarbanes-Oxley\b",
    r"\bhereby certif(?:y|ies)\b",
    r"\b(?:Exhibit|Item)\s+\d+\.?\d*\b",
    # Proxy statement / incorporated by reference boilerplate
    r"\bwill be included in the.{0,30}(?:Proxy Statement|proxy statement)\b",
    r"\bincorporated herein by reference\b",
    r"\b(?:Proxy Statement|DEF 14A)\b.{0,60}(?:incorporated|reference|included)\b",
    r"\bthe information required by this Item\b",
    # Auditor boilerplate
    r"\b(?:Ernst & Young|Deloitte|KPMG|PricewaterhouseCoopers|PwC)\b.{0,60}(?:auditor|served)",
    r"\bWe have served as the Company.{0,10}s auditor\b",
    # Stock / share language
    r"\b(?:shares? outstanding|stock repurchase|buyback|dividends? per share|share price)\b",
    r"\b(?:repurchased|reacquired).{0,30}shares\b",
    # Generic competition / business strategy (not ESG)
    r"\bprincipal competitive factors\b",
    r"\bimportant to the Company include price, product\b",
    # Pure financial comparison with no ESG keyword
    r"\b(?:net sales|revenue|income|profit|margin|expense)\b.{0,60}(?:increased|decreased|grew|declined).{0,60}\b20\d{2}\b",
]

_BLOCKLIST_COMPILED = [re.compile(p, re.IGNORECASE) for p in _FINANCIAL_BLOCKLIST]


# ── REQUIRED: sentence must contain at least one ESG anchor keyword ───────────
# This is the gatekeeper — if a sentence has no ESG relevance, skip it

_ESG_ANCHOR_PATTERN = re.compile(
    r"\b(?:"
    # Environmental
    r"emission|carbon|greenhouse|GHG|CO2|climate|net.?zero|carbon.?neutral|"
    r"renewable|clean energy|solar|wind|fossil fuel|biodiversity|deforestation|"
    r"scope [123]|scope one|scope two|scope three|TCFD|Paris Agreement|"
    r"water consumption|water usage|water withdrawal|water recycl|"
    r"waste divert|recycl|landfill|circular economy|sustainable packaging|"
    r"environmental|ecology|ecosystem|pollution|"
    # Social
    r"workforce diversity|gender (?:pay )?gap|pay equity|equal pay|"
    r"employee safety|workplace safety|injury rate|fatality|TRIR|LTIR|"
    r"living wage|fair wage|minimum wage|human rights|child labor|forced labor|"
    r"supply chain (?:labor|ethics|audit)|supplier diversity|"
    r"community investment|social impact|philanthrop|volunteering hours|"
    r"employee wellbeing|mental health|parental leave|paid leave|"
    r"training hours|learning and development|employee engagement|"
    r"DEI|diversity, equity|inclusion|underrepresented|BIPOC|"
    r"data privacy|customer privacy|"
    # Broader social — individual words (need these for sentences like "Women represent X% of workforce")
    r"women|female|gender|workforce|employees|worker|labour|labor|"
    r"safety incident|recordable incident|lost.?time|near miss|"
    # Governance
    r"board (?:diversity|independence|composition|oversight)|"
    r"independent director|audit committee|"
    r"executive compensation|CEO pay ratio|"
    r"anti-corruption|anti-bribery|whistleblower|ethics hotline|"
    r"GRI|SASB|ISSB|CSRD|SDG|ESG report|sustainability report|"
    r"third.?party audit|assurance|verification|"
    r"political contribution|lobbying disclosure"
    r")\b",
    re.IGNORECASE,
)


# ── ESG-SPECIFIC CLAIM PATTERNS ──────────────────────────────────────────────
# Only match if the sentence ALSO contains an ESG anchor (checked separately)

CLAIM_PATTERNS: list[tuple[str, str, str, float]] = [
    # Quantitative ESG metric claims — requires ESG unit words
    ("emission_quantity",
     r"\b(?:reduced?|decreas(?:ed?|ing)|cut|lower(?:ed?|ing)|achiev(?:ed?|ing)|increas(?:ed?|ing)|emitted?)\b"
     r".{0,80}?\b(?:tonne|tCO2|metric ton|MWh|GWh|gallon|litre|liter|MW|GW|kg CO2|m³|cubic met)\b",
     "quantitative", 0.9),

    ("emission_percentage",
     r"\b(?:reduced?|decreas(?:ed?|ing)|cut|lower(?:ed?|ing)|achiev(?:ed?|ing))\b"
     r".{0,60}?\b(\d+(?:\.\d+)?)\s*%"
     r".{0,80}?\b(?:emission|carbon|energy|water|waste|GHG|scope|renewabl)\b",
     "quantitative", 0.92),

    # ESG percentage of total
    ("esg_percentage_of",
     r"\b(\d+(?:\.\d+)?)\s*%\s+of\s+(?:our|total|all)?\s*"
     r"(?:energy|electricity|water|waste|fleet|employees?|supply chain|workforce|sourcing)\b",
     "quantitative", 0.85),

    # Net zero / carbon neutral targets
    ("net_zero",
     r"\bnet.{0,5}zero\b.{0,100}?(?:by|in|before|20\d{2})",
     "commitment", 0.95),

    ("carbon_neutral",
     r"\bcarbon.{0,10}neutral(?:ity)?\b",
     "commitment", 0.90),

    # Forward commitment with year — ESG context required (anchor check handles it)
    ("commitment_target",
     r"\b(?:target|goal|commit(?:ment|ted)|aim(?:ing)?|plan(?:ning)?|aspir(?:ing|ation)|pledge)\b"
     r".{0,100}?\b(?:by|in|before)\s+20\d{2}\b",
     "commitment", 0.85),

    # GHG scope claims
    ("ghg_scope",
     r"\bScope\s+[123I]+\s+(?:emissions?|GHG|greenhouse gas)\b",
     "quantitative", 0.90),

    # Standard / certification references
    ("standard_certification",
     r"\b(?:GRI\s+\d{3}|TCFD|SASB|ISSB|IFRS S[12]|ISO\s*(?:14001|26000|45001|50001)|LEED|ENERGY STAR|"
     r"B Corp|SA8000|CDP|Science Based Targets|SBTi|Task Force on Climate)\b",
     "compliance", 0.88),

    # ESG comparative (vs. industry/sector/baseline — not financial)
    ("esg_comparative",
     r"\b(?:lower|higher|better|above|below)\s+(?:than|the)\s+"
     r"(?:industry|sector|average|baseline|peer|benchmark)\b"
     r".{0,100}?\b(?:emission|carbon|safety|diversity|waste|energy|water|ESG)\b",
     "comparative", 0.78),

    # Workforce diversity / social metrics
    ("diversity_metric",
     r"\b(?:\d+(?:\.\d+)?)\s*%\s+(?:of\s+)?(?:our\s+)?(?:women|female|men|male|"
     r"minority|underrepresented|BIPOC|Hispanic|Black|Asian|veteran|disabled|"
     r"manag(?:ers?|ement)|leadership|board|executives?|workforce|employees?)\b",
     "quantitative", 0.88),

    # Safety metrics
    ("safety_metric",
     r"\b(?:TRIR|LTIR|DART|recordable injury|lost.?time injury|fatality|near miss|"
     r"safety incident|workplace accident)\b",
     "quantitative", 0.90),

    # Renewable energy claims
    ("renewable_energy",
     r"\b(?:renewable|clean|solar|wind|hydro|geotherm)\b.{0,60}?"
     r"(?:energy|electricity|power)\b",
     "quantitative", 0.85),
]

_COMPILED_PATTERNS = [
    (name, re.compile(pattern, re.IGNORECASE), claim_type, boost)
    for name, pattern, claim_type, boost in CLAIM_PATTERNS
]


# ── Main class ────────────────────────────────────────────────────────────────

CLAIM_EXTRACTION_PROMPT = """You are an expert ESG auditor.
You are given a list of candidate ESG sentences extracted from a company's financial filing, along with their surrounding sentence context.

Your task is to analyze each candidate and extract all distinct, verifiable atomic ESG (Environmental, Social, or Governance) claims.
An ESG claim is a statement that:
1. Makes a quantitative assertion about an ESG metric (e.g., emissions, water, waste, diversity pct, safety rates).
2. States a forward-looking ESG commitment or target.
3. References specific ESG standards or certifications (GRI, TCFD, SASB, etc.).
4. Describes a specific ESG policy, risk, or compliance action.

CRITICAL Granularity Rule:
- Split every independent factual assertion into a separate claim object. One claim = one independently verifiable ESG fact.
- Do NOT merge multiple independent policies, targets, risks, or governance statements into a single claim.
- If a sentence contains three distinct factual assertions, return three separate claim objects in the "claims" list.
- If two extracted claims express the same factual assertion with only minor wording differences, return only the more specific claim. Example: Keep "Reduce carbon footprint." and discard "Reduce environmental impact." because the second is implied by the first.
- Boilerplate statements, general company facts, and pure financial/economic metrics (revenue, profit, margins) are NOT ESG claims and must result in is_actual_claim = false.

Input candidates (JSON format):
{candidates}

For each candidate, output a JSON object containing:
- "sentence_index": (integer) matching the input candidate's index
- "is_actual_claim": (boolean) true if the candidate contains one or more valid ESG claims, false otherwise
- "claims": (array of objects, only if is_actual_claim is true) each object containing:
  - "normalized_claim": (string) a concise, normalized, and standalone version of the claim (e.g. "Board reviews ESG risks.")
  - "category": (string) "Environmental", "Social", "Governance", or "Mixed"
  - "claim_type": (string) "Policy", "Target", "Risk", "Performance", or "Compliance"
  - "evidence": (string) the exact substring/supporting text from the original sentence containing this claim

Return a valid JSON array of objects. Do not wrap in markdown code blocks or add any other text outside the JSON.
"""


class ClaimDetector:
    """
    Detects ESG claims in a document using a model-driven hybrid approach:
    1. Sentence Splitting (spaCy).
    2. Candidate Filtering: FinBERT-ESG sentence classification.
    3. Context Packaging: Packaging candidates with ±1 sentence context.
    4. Structured LLM extraction: Gemini verifies and structures claims.
    """

    def __init__(
        self,
        classifier: Optional[FinBertClassifier] = None,
        llm: Optional[BaseLanguageModel] = None,
        confidence_threshold: float = 0.80,
        batch_size: int = 10,
        use_keyword_filter: bool = True,
        max_candidates: Optional[int] = None,
    ) -> None:
        self.classifier = classifier
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.use_keyword_filter = use_keyword_filter
        self.max_candidates = max_candidates
        logger.info(
            f"ClaimDetector initialised | "
            f"threshold={confidence_threshold} | batch_size={batch_size} | "
            f"use_keyword_filter={use_keyword_filter} | max_candidates={max_candidates}"
        )

    def detect_claims(
        self,
        text: str,
        source_section: str = "general",
        char_offset: int = 0,
    ) -> list[Claim]:
        """
        Detect ESG claims in the provided text.

        Args:
            text:           Full or sectioned text to analyse
            source_section: Section label ("environmental", "social", "governance")
            char_offset:    Character offset if text is a substring of a larger doc

        Returns:
            List of Claim objects validated and normalized by Gemini.
        """
        if not text.strip():
            return []

        if self.llm is None:
            raise ValueError(
                "Gemini LLM is required for claim extraction. "
                "Ensure GEMINI_API_KEY is configured in your environment."
            )

        nlp = _get_nlp()
        doc = nlp(text)

        try:
            sentences = list(doc.sents)
        except Exception:
            sentences = self._fallback_sentences(text)
            logger.debug("Using fallback sentence splitter")

        # 1. Clean and filter sentences locally (fast checks)
        cleaned_sentences = []
        for i, sent in enumerate(sentences):
            sent_text = sent.text.strip() if hasattr(sent, "text") else str(sent).strip()

            # Basic cleanups
            if len(sent_text) < 30 or len(sent_text) > 500:
                continue
            if sent_text.isupper() and len(sent_text) < 100:
                continue
            if self._is_blocked(sent_text):
                continue
            if self.use_keyword_filter and not _ESG_ANCHOR_PATTERN.search(sent_text):
                continue

            if hasattr(sent, "start_char"):
                start = sent.start_char + char_offset
                end = sent.end_char + char_offset
            else:
                start = text.find(sent_text) + char_offset
                end = start + len(sent_text)

            cleaned_sentences.append({
                "index": i,
                "text": sent_text,
                "start": start,
                "end": end,
            })

        if not cleaned_sentences:
            return []

        # 2. FinBERT Filtering
        candidate_sentences = []
        if self.classifier:
            texts_to_classify = [s["text"] for s in cleaned_sentences]
            sections_list = [source_section] * len(texts_to_classify)
            classifications = self.classifier.classify_batch(
                texts_to_classify, source_sections=sections_list
            )

            for s, res in zip(cleaned_sentences, classifications):
                # Filter: must be Environmental, Social, or Governance, with confidence >= threshold
                if res.esg_label in {"E", "S", "G"} and res.confidence >= self.confidence_threshold:
                    s["finbert_confidence"] = res.confidence
                    s["finbert_esg_label"] = res.esg_label
                    candidate_sentences.append(s)
        else:
            # Fallback if no classifier is configured
            for s in cleaned_sentences:
                if _ESG_ANCHOR_PATTERN.search(s["text"]):
                    s["finbert_confidence"] = 0.5
                    s["finbert_esg_label"] = "MIXED"
                    candidate_sentences.append(s)

        if not candidate_sentences:
            return []

        # Sort and limit candidates if max_candidates is set
        if self.max_candidates is not None:
            candidate_sentences.sort(key=lambda x: x.get("finbert_confidence", 0.0), reverse=True)
            candidate_sentences = candidate_sentences[:self.max_candidates]

        # 3. Build extraction units with ±1 sentence context
        num_sentences = len(sentences)
        extraction_units = []
        for s in candidate_sentences:
            idx = s["index"]
            prev_text = ""
            if idx > 0:
                prev_sent = sentences[idx - 1]
                prev_text = prev_sent.text.strip() if hasattr(prev_sent, "text") else str(prev_sent).strip()
            next_text = ""
            if idx < num_sentences - 1:
                next_sent = sentences[idx + 1]
                next_text = next_sent.text.strip() if hasattr(next_sent, "text") else str(next_sent).strip()

            context_pieces = [prev_text, s["text"], next_text]
            context = " ".join([p for p in context_pieces if p])

            extraction_units.append({
                "sentence_index": idx,
                "evidence": s["text"],
                "context": context,
                "start": s["start"],
                "end": s["end"],
                "finbert_confidence": s["finbert_confidence"],
                "finbert_esg_label": s["finbert_esg_label"],
            })

        # 4. Batch and call Gemini
        claims: list[Claim] = []
        seen_texts: set[str] = set()

        for batch_start in range(0, len(extraction_units), self.batch_size):
            batch = extraction_units[batch_start : batch_start + self.batch_size]
            batch_claims = self._extract_claims_from_batch_with_retry(batch, source_section)

            for claim in batch_claims:
                norm = re.sub(r"\s+", " ", claim.text.lower()).strip()
                if norm in seen_texts:
                    continue
                seen_texts.add(norm)
                claims.append(claim)

        logger.info(
            f"Detected {len(claims)} ESG claims in '{source_section}' section "
            f"({len(text):,} chars processed)"
        )
        return self._quality_filter(claims)

    def _quality_filter(self, claims: list[Claim]) -> list[Claim]:
        """
        Filters out weak, generic, duplicates, or non-auditable claims
        (like operational boilerplate, generic HR policies, and cybersecurity oversight).
        """
        filtered = []

        # Keywords that indicate generic, non-auditable operational boilerplate
        generic_hr_phrases = [
            r"\bcompetitive pay\b",
            r"\bmentorship\b",
            r"\bflexible work\b",
            r"\bonline classes\b",
            r"\bemployee engagement\b",
            r"\bmechanisms to hire\b",
            r"\bdevelop, evaluate, and retain\b",
            r"\bskills training\b",
        ]

        operational_boilerplate = [
            r"\butilizes independent contractors\b",
            r"\bsupplement its workforce\b",
            r"\bsubject to labor union\b",
            r"\blabor union organizing\b",
            r"\blitigation regarding\b",
            r"\bclass actions\b",
        ]

        governance_cybersecurity_boilerplate = [
            r"\bcybersecurity risk\b",
            r"\binformation security\b",
            r"\bcybersecurity incident\b",
            r"\bsecurity committee\b",
            r"\baudit committee regularly\b",
            r"\boversight and monitoring\b",
        ]

        # Combine compiled regex patterns for speed
        all_generic_patterns = [
            re.compile(p, re.IGNORECASE)
            for p in (generic_hr_phrases + operational_boilerplate + governance_cybersecurity_boilerplate)
        ]

        for claim in claims:
            text = claim.text
            has_numbers = bool(re.search(r"\b\d+[\d,.]*\b", text))

            # 1. If it has numbers, always keep it (it's quantitative evidence)
            if has_numbers:
                filtered.append(claim)
                continue

            # 2. Check if it matches any boilerplate or generic rules
            is_generic = False
            for pattern in all_generic_patterns:
                if pattern.search(text):
                    is_generic = True
                    break

            if is_generic:
                logger.info(f"Quality filter removed claim: '{text}' (reason: generic/boilerplate)")
                continue

            # 3. Filter very short claims
            if len(text.split()) < 5:
                logger.info(f"Quality filter removed claim: '{text}' (reason: too short)")
                continue

            filtered.append(claim)

        logger.success(f"Quality filtering complete: kept {len(filtered)} of {len(claims)} claims.")
        return filtered

    def detect_from_document(self, sections: dict[str, str]) -> list[Claim]:
        """
        Run claim detection across all document sections.

        Args:
            sections: Dict from ExtractedDocument.sections
                      e.g. {"environmental": "...", "social": "..."}

        Returns:
            All ESG claims across all sections, sorted by confidence (desc)
        """
        all_claims: list[Claim] = []
        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue
            claims = self.detect_claims(section_text, source_section=section_name)
            all_claims.extend(claims)

        all_claims.sort(key=lambda c: c.confidence, reverse=True)
        logger.success(f"Total ESG claims detected: {len(all_claims)}")
        return all_claims

    def _extract_claims_from_batch_with_retry(
        self,
        batch: list[dict],
        source_section: str,
    ) -> list[Claim]:
        """Send a batch of extraction units to Gemini, parse JSON, and handle retries."""
        candidates = []
        for unit in batch:
            candidates.append({
                "sentence_index": unit["sentence_index"],
                "evidence": unit["evidence"],
                "context": unit["context"],
            })

        prompt_str = CLAIM_EXTRACTION_PROMPT.format(candidates=json.dumps(candidates, indent=2))

        try:
            return self._execute_extraction_run(prompt_str, batch, source_section)
        except Exception as exc:
            logger.warning(f"Failed extraction batch attempt. Error: {exc}. Retrying once...")
            try:
                return self._execute_extraction_run(prompt_str, batch, source_section)
            except Exception as final_exc:
                logger.error(f"Failed extraction batch retry. Error: {final_exc}")
                return []

    def _execute_extraction_run(
        self,
        prompt_str: str,
        batch: list[dict],
        source_section: str,
    ) -> list[Claim]:
        """Helper to invoke LLM, validate response against schema, and build Claim list."""
        try:
            response = self.llm.invoke(prompt_str)
            response_text = response.content if hasattr(response, "content") else str(response)
        except Exception as exc:
            exc_str = str(exc).lower()
            if "finishreason" in exc_str or "safety" in exc_str or "block" in exc_str or "finish_reason" in exc_str:
                logger.warning(
                    "Batch generation was blocked by safety filters or finish reason (e.g., FinishReason 19). "
                    "Skipping this batch to prevent pipeline crash."
                )
                return []
            raise exc

        cleaned_json = self._clean_json_response(response_text)

        try:
            results = json.loads(cleaned_json)
        except Exception as exc:
            raise ValueError(f"Invalid JSON returned by Gemini: {exc}\nResponse: {response_text[:300]}")

        if not isinstance(results, list):
            raise TypeError("LLM response must be a JSON array of objects.")

        claims = []
        batch_map = {item["sentence_index"]: item for item in batch}

        for obj in results:
            if not isinstance(obj, dict):
                continue

            is_actual = obj.get("is_actual_claim", False)
            if not is_actual:
                continue

            sent_idx = obj.get("sentence_index")
            if sent_idx is None or sent_idx not in batch_map:
                continue

            unit = batch_map[sent_idx]
            
            claims_list = obj.get("claims")
            if not isinstance(claims_list, list):
                continue

            for atomic in claims_list:
                if not isinstance(atomic, dict):
                    continue

                normalized_val = atomic.get("normalized_claim")
                if isinstance(normalized_val, list):
                    normalized = " ".join([str(x) for x in normalized_val]).strip()
                elif normalized_val is not None:
                    normalized = str(normalized_val).strip()
                else:
                    normalized = ""

                category = atomic.get("category")
                raw_claim_type = atomic.get("claim_type")

                if not normalized or not category or not raw_claim_type:
                    # Skip incomplete or malformed atomic claim objects
                    continue

                category = str(category).strip()
                raw_claim_type = str(raw_claim_type).strip()

                # Get exact evidence sentence, fallback to the original parent sentence
                evidence_val = atomic.get("evidence")
                if isinstance(evidence_val, list):
                    evidence_sentence = " ".join([str(x) for x in evidence_val]).strip()
                elif evidence_val is not None:
                    evidence_sentence = str(evidence_val).strip()
                else:
                    evidence_sentence = str(unit["evidence"]).strip()

                # Map category
                esg_label = "MIXED"
                category_upper = category.upper()
                if "ENVIRONMENT" in category_upper or category_upper == "E":
                    esg_label = "E"
                elif "SOCIAL" in category_upper or category_upper == "S":
                    esg_label = "S"
                elif "GOVERNANCE" in category_upper or category_upper == "G":
                    esg_label = "G"

                # Map claim type
                claim_type_lower = raw_claim_type.lower()
                if "target" in claim_type_lower or "commitment" in claim_type_lower:
                    claim_type = "commitment"
                elif "performance" in claim_type_lower or "quantitative" in claim_type_lower:
                    claim_type = "quantitative"
                elif "compliance" in claim_type_lower:
                    claim_type = "compliance"
                else:
                    claim_type = claim_type_lower

                claims.append(Claim(
                    text=normalized,
                    evidence_sentence=evidence_sentence,
                    evidence=unit["context"],
                    start_char=unit["start"],
                    end_char=unit["end"],
                    source_section=source_section,
                    claim_type=claim_type,
                    confidence=round(unit["finbert_confidence"], 3),
                    esg_label=esg_label,
                    matched_patterns=["model_extracted"],
                ))

        return claims

    @staticmethod
    def _clean_json_response(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            newline_idx = text.find("\n")
            if newline_idx != -1:
                text = text[newline_idx:].strip()
            if text.endswith("```"):
                text = text[:-3].strip()
        return text

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _is_blocked(text: str) -> bool:
        """Return True if sentence matches any financial/legal blocklist pattern."""
        for pattern in _BLOCKLIST_COMPILED:
            if pattern.search(text):
                return True
        return False

    @staticmethod
    def _fallback_sentences(text: str) -> list[str]:
        """Simple sentence splitter when spaCy pipeline is unavailable."""
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in raw if s.strip()]
