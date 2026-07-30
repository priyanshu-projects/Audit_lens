from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from loguru import logger
from src.classification.finbert_classifier import FinBertClassifier
from langchain_core.language_models import BaseLanguageModel
_NLP = None

def _get_nlp():
    global _NLP
    if _NLP is None:
        import spacy
        try:
            _NLP = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            _NLP = spacy.blank('en')
            _NLP.add_pipe('sentencizer')
    return _NLP

@dataclass
class Claim:
    text: str
    start_char: int
    end_char: int
    evidence_sentence: str = ''
    evidence: str = ''
    source_section: str = 'general'
    claim_type: str = 'quantitative'
    confidence: float = 0.5
    esg_label: str = 'MIXED'
    matched_patterns: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {'text': self.text, 'evidence_sentence': self.evidence_sentence, 'evidence': self.evidence, 'start_char': self.start_char, 'end_char': self.end_char, 'source_section': self.source_section, 'claim_type': self.claim_type, 'confidence': self.confidence, 'matched_patterns': self.matched_patterns}
_FINANCIAL_BLOCKLIST: list[str] = ['\\b(?:net sales|gross revenue|total revenue|net revenue|product revenue|service revenue)\\b', '\\b(?:iPhone|Mac|iPad|Apple Watch|AirPods|HomePod|Vision Pro)\\b.{0,60}(?:net sales|revenue|increased|decreased)', '\\b(?:Americas|Europe|Greater China|Japan|Asia Pacific)\\b.{0,60}(?:net sales|revenue|increased|decreased)', '\\bsegment.{0,30}(?:net sales|revenue)\\b', '\\b(?:gross margin|operating margin|net income|earnings per share|EPS|diluted EPS)\\b', '\\b(?:operating income|operating expense|operating profit|net profit)\\b', '\\b(?:selling, general and administrative|SG&A|R&D expense|research and development expense)\\b', '\\beffective tax rate\\b', '\\bdeferred tax\\b', '\\b(?:fiscal year|quarterly results|annual results)\\b.{0,40}(?:increased|decreased|grew|declined)', '\\b(?:Rule\\s+\\d+[a-z]?-\\d+|15d-14|13a-14)\\b', '\\bCertification of Chief (?:Executive|Financial) Officer\\b', '\\bPursuant to (?:Section|Rule)\\b', '\\bSarbanes-Oxley\\b', '\\bhereby certif(?:y|ies)\\b', '\\b(?:Exhibit|Item)\\s+\\d+\\.?\\d*\\b', '\\bwill be included in the.{0,30}(?:Proxy Statement|proxy statement)\\b', '\\bincorporated herein by reference\\b', '\\b(?:Proxy Statement|DEF 14A)\\b.{0,60}(?:incorporated|reference|included)\\b', '\\bthe information required by this Item\\b', '\\b(?:Ernst & Young|Deloitte|KPMG|PricewaterhouseCoopers|PwC)\\b.{0,60}(?:auditor|served)', '\\bWe have served as the Company.{0,10}s auditor\\b', '\\b(?:shares? outstanding|stock repurchase|buyback|dividends? per share|share price)\\b', '\\b(?:repurchased|reacquired).{0,30}shares\\b', '\\bprincipal competitive factors\\b', '\\bimportant to the Company include price, product\\b', '\\b(?:net sales|revenue|income|profit|margin|expense)\\b.{0,60}(?:increased|decreased|grew|declined).{0,60}\\b20\\d{2}\\b']
_BLOCKLIST_COMPILED = [re.compile(p, re.IGNORECASE) for p in _FINANCIAL_BLOCKLIST]
_ESG_ANCHOR_PATTERN = re.compile('\\b(?:emission|carbon|greenhouse|GHG|CO2|climate|net.?zero|carbon.?neutral|renewable|clean energy|solar|wind|fossil fuel|biodiversity|deforestation|scope [123]|scope one|scope two|scope three|TCFD|Paris Agreement|water consumption|water usage|water withdrawal|water recycl|waste divert|recycl|landfill|circular economy|sustainable packaging|environmental|ecology|ecosystem|pollution|workforce diversity|gender (?:pay )?gap|pay equity|equal pay|employee safety|workplace safety|injury rate|fatality|TRIR|LTIR|living wage|fair wage|minimum wage|human rights|child labor|forced labor|supply chain (?:labor|ethics|audit)|supplier diversity|community investment|social impact|philanthrop|volunteering hours|employee wellbeing|mental health|parental leave|paid leave|training hours|learning and development|employee engagement|DEI|diversity, equity|inclusion|underrepresented|BIPOC|data privacy|customer privacy|women|female|gender|workforce|employees|worker|labour|labor|safety incident|recordable incident|lost.?time|near miss|board (?:diversity|independence|composition|oversight)|independent director|audit committee|executive compensation|CEO pay ratio|anti-corruption|anti-bribery|whistleblower|ethics hotline|GRI|SASB|ISSB|CSRD|SDG|ESG report|sustainability report|third.?party audit|assurance|verification|political contribution|lobbying disclosure)\\b', re.IGNORECASE)
CLAIM_PATTERNS: list[tuple[str, str, str, float]] = [('emission_quantity', '\\b(?:reduced?|decreas(?:ed?|ing)|cut|lower(?:ed?|ing)|achiev(?:ed?|ing)|increas(?:ed?|ing)|emitted?)\\b.{0,80}?\\b(?:tonne|tCO2|metric ton|MWh|GWh|gallon|litre|liter|MW|GW|kg CO2|m³|cubic met)\\b', 'quantitative', 0.9), ('emission_percentage', '\\b(?:reduced?|decreas(?:ed?|ing)|cut|lower(?:ed?|ing)|achiev(?:ed?|ing))\\b.{0,60}?\\b(\\d+(?:\\.\\d+)?)\\s*%.{0,80}?\\b(?:emission|carbon|energy|water|waste|GHG|scope|renewabl)\\b', 'quantitative', 0.92), ('esg_percentage_of', '\\b(\\d+(?:\\.\\d+)?)\\s*%\\s+of\\s+(?:our|total|all)?\\s*(?:energy|electricity|water|waste|fleet|employees?|supply chain|workforce|sourcing)\\b', 'quantitative', 0.85), ('net_zero', '\\bnet.{0,5}zero\\b.{0,100}?(?:by|in|before|20\\d{2})', 'commitment', 0.95), ('carbon_neutral', '\\bcarbon.{0,10}neutral(?:ity)?\\b', 'commitment', 0.9), ('commitment_target', '\\b(?:target|goal|commit(?:ment|ted)|aim(?:ing)?|plan(?:ning)?|aspir(?:ing|ation)|pledge)\\b.{0,100}?\\b(?:by|in|before)\\s+20\\d{2}\\b', 'commitment', 0.85), ('ghg_scope', '\\bScope\\s+[123I]+\\s+(?:emissions?|GHG|greenhouse gas)\\b', 'quantitative', 0.9), ('standard_certification', '\\b(?:GRI\\s+\\d{3}|TCFD|SASB|ISSB|IFRS S[12]|ISO\\s*(?:14001|26000|45001|50001)|LEED|ENERGY STAR|B Corp|SA8000|CDP|Science Based Targets|SBTi|Task Force on Climate)\\b', 'compliance', 0.88), ('esg_comparative', '\\b(?:lower|higher|better|above|below)\\s+(?:than|the)\\s+(?:industry|sector|average|baseline|peer|benchmark)\\b.{0,100}?\\b(?:emission|carbon|safety|diversity|waste|energy|water|ESG)\\b', 'comparative', 0.78), ('diversity_metric', '\\b(?:\\d+(?:\\.\\d+)?)\\s*%\\s+(?:of\\s+)?(?:our\\s+)?(?:women|female|men|male|minority|underrepresented|BIPOC|Hispanic|Black|Asian|veteran|disabled|manag(?:ers?|ement)|leadership|board|executives?|workforce|employees?)\\b', 'quantitative', 0.88), ('safety_metric', '\\b(?:TRIR|LTIR|DART|recordable injury|lost.?time injury|fatality|near miss|safety incident|workplace accident)\\b', 'quantitative', 0.9), ('renewable_energy', '\\b(?:renewable|clean|solar|wind|hydro|geotherm)\\b.{0,60}?(?:energy|electricity|power)\\b', 'quantitative', 0.85)]
_COMPILED_PATTERNS = [(name, re.compile(pattern, re.IGNORECASE), claim_type, boost) for name, pattern, claim_type, boost in CLAIM_PATTERNS]
CLAIM_EXTRACTION_PROMPT = 'You are an expert ESG auditor.\nYou are given a list of candidate ESG sentences extracted from a company\'s financial filing, along with their surrounding sentence context.\n\nYour task is to analyze each candidate and extract all distinct, verifiable atomic ESG (Environmental, Social, or Governance) claims.\nAn ESG claim is a statement that:\n1. Makes a quantitative assertion about an ESG metric (e.g., emissions, water, waste, diversity pct, safety rates).\n2. States a forward-looking ESG commitment or target.\n3. References specific ESG standards or certifications (GRI, TCFD, SASB, etc.).\n4. Describes a specific ESG policy, risk, or compliance action.\n\nCRITICAL Granularity Rule:\n- Split every independent factual assertion into a separate claim object. One claim = one independently verifiable ESG fact.\n- Do NOT merge multiple independent policies, targets, risks, or governance statements into a single claim.\n- If a sentence contains three distinct factual assertions, return three separate claim objects in the "claims" list.\n- If two extracted claims express the same factual assertion with only minor wording differences, return only the more specific claim. Example: Keep "Reduce carbon footprint." and discard "Reduce environmental impact." because the second is implied by the first.\n- Boilerplate statements, general company facts, and pure financial/economic metrics (revenue, profit, margins) are NOT ESG claims and must result in is_actual_claim = false.\n\nInput candidates (JSON format):\n{candidates}\n\nFor each candidate, output a JSON object containing:\n- "sentence_index": (integer) matching the input candidate\'s index\n- "is_actual_claim": (boolean) true if the candidate contains one or more valid ESG claims, false otherwise\n- "claims": (array of objects, only if is_actual_claim is true) each object containing:\n  - "normalized_claim": (string) a concise, normalized, and standalone version of the claim (e.g. "Board reviews ESG risks.")\n  - "category": (string) "Environmental", "Social", "Governance", or "Mixed"\n  - "claim_type": (string) "Policy", "Target", "Risk", "Performance", or "Compliance"\n  - "evidence": (string) the exact substring/supporting text from the original sentence containing this claim\n\nReturn a valid JSON array of objects. Do not wrap in markdown code blocks or add any other text outside the JSON.\n'

class ClaimDetector:

    def __init__(self, classifier: Optional[FinBertClassifier]=None, llm: Optional[BaseLanguageModel]=None, confidence_threshold: float=0.8, batch_size: int=10, use_keyword_filter: bool=True, max_candidates: Optional[int]=None) -> None:
        self.classifier = classifier
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size
        self.use_keyword_filter = use_keyword_filter
        self.max_candidates = max_candidates
        logger.info(f'ClaimDetector initialised | threshold={confidence_threshold} | batch_size={batch_size} | use_keyword_filter={use_keyword_filter} | max_candidates={max_candidates}')

    def detect_claims(self, text: str, source_section: str='general', char_offset: int=0) -> list[Claim]:
        if not text.strip():
            return []
        if self.llm is None:
            raise ValueError('Gemini LLM is required for claim extraction. Ensure GEMINI_API_KEY is configured in your environment.')
        nlp = _get_nlp()
        doc = nlp(text)
        try:
            sentences = list(doc.sents)
        except Exception:
            sentences = self._fallback_sentences(text)
            logger.debug('Using fallback sentence splitter')
        cleaned_sentences = []
        for i, sent in enumerate(sentences):
            sent_text = sent.text.strip() if hasattr(sent, 'text') else str(sent).strip()
            if len(sent_text) < 30 or len(sent_text) > 500:
                continue
            if sent_text.isupper() and len(sent_text) < 100:
                continue
            if self._is_blocked(sent_text):
                continue
            if self.use_keyword_filter and (not _ESG_ANCHOR_PATTERN.search(sent_text)):
                continue
            if hasattr(sent, 'start_char'):
                start = sent.start_char + char_offset
                end = sent.end_char + char_offset
            else:
                start = text.find(sent_text) + char_offset
                end = start + len(sent_text)
            cleaned_sentences.append({'index': i, 'text': sent_text, 'start': start, 'end': end})
        if not cleaned_sentences:
            return []
        candidate_sentences = []
        if self.classifier:
            texts_to_classify = [s['text'] for s in cleaned_sentences]
            sections_list = [source_section] * len(texts_to_classify)
            classifications = self.classifier.classify_batch(texts_to_classify, source_sections=sections_list)
            for s, res in zip(cleaned_sentences, classifications):
                is_high_conf_esg = res.esg_label in {'E', 'S', 'G'} and res.confidence >= self.confidence_threshold
                has_esg_anchor = _ESG_ANCHOR_PATTERN.search(s['text']) is not None
                if is_high_conf_esg or has_esg_anchor:
                    s['finbert_confidence'] = res.confidence
                    s['finbert_esg_label'] = res.esg_label if is_high_conf_esg else (res.esg_label if res.esg_label in {'E', 'S', 'G'} else 'MIXED')
                    candidate_sentences.append(s)
        else:
            for s in cleaned_sentences:
                if _ESG_ANCHOR_PATTERN.search(s['text']):
                    s['finbert_confidence'] = 0.5
                    s['finbert_esg_label'] = 'MIXED'
                    candidate_sentences.append(s)
        if not candidate_sentences:
            return []
        if self.max_candidates is not None:
            candidate_sentences.sort(key=lambda x: x.get('finbert_confidence', 0.0), reverse=True)
            candidate_sentences = candidate_sentences[:self.max_candidates]
        num_sentences = len(sentences)
        extraction_units = []
        for s in candidate_sentences:
            idx = s['index']
            prev_text = ''
            if idx > 0:
                prev_sent = sentences[idx - 1]
                prev_text = prev_sent.text.strip() if hasattr(prev_sent, 'text') else str(prev_sent).strip()
            next_text = ''
            if idx < num_sentences - 1:
                next_sent = sentences[idx + 1]
                next_text = next_sent.text.strip() if hasattr(next_sent, 'text') else str(next_sent).strip()
            context_pieces = [prev_text, s['text'], next_text]
            context = ' '.join([p for p in context_pieces if p])
            extraction_units.append({'sentence_index': idx, 'evidence': s['text'], 'context': context, 'start': s['start'], 'end': s['end'], 'finbert_confidence': s['finbert_confidence'], 'finbert_esg_label': s['finbert_esg_label']})
        claims: list[Claim] = []
        seen_texts: set[str] = set()
        for batch_start in range(0, len(extraction_units), self.batch_size):
            batch = extraction_units[batch_start:batch_start + self.batch_size]
            batch_claims = self._extract_claims_from_batch_with_retry(batch, source_section)
            for claim in batch_claims:
                norm = re.sub('\\s+', ' ', claim.text.lower()).strip()
                if norm in seen_texts:
                    continue
                seen_texts.add(norm)
                claims.append(claim)
        logger.info(f"Detected {len(claims)} ESG claims in '{source_section}' section ({len(text):,} chars processed)")
        return self._quality_filter(claims)

    def _quality_filter(self, claims: list[Claim]) -> list[Claim]:
        filtered = []
        generic_hr_phrases = ['\\bcompetitive pay\\b', '\\bmentorship\\b', '\\bflexible work\\b', '\\bonline classes\\b', '\\bemployee engagement\\b', '\\bmechanisms to hire\\b', '\\bdevelop, evaluate, and retain\\b', '\\bskills training\\b']
        operational_boilerplate = ['\\butilizes independent contractors\\b', '\\bsupplement its workforce\\b', '\\bsubject to labor union\\b', '\\blabor union organizing\\b', '\\blitigation regarding\\b', '\\bclass actions\\b']
        governance_cybersecurity_boilerplate = ['\\bcybersecurity risk\\b', '\\bcybersecurity incident\\b']
        all_generic_patterns = [re.compile(p, re.IGNORECASE) for p in generic_hr_phrases + operational_boilerplate + governance_cybersecurity_boilerplate]
        for claim in claims:
            text = claim.text
            has_numbers = bool(re.search('\\b\\d+[\\d,.]*\\b', text))
            if has_numbers:
                filtered.append(claim)
                continue
            is_generic = False
            for pattern in all_generic_patterns:
                if pattern.search(text):
                    is_generic = True
                    break
            if is_generic:
                logger.info(f"Quality filter removed claim: '{text}' (reason: generic/boilerplate)")
                continue
            if len(text.split()) < 5:
                logger.info(f"Quality filter removed claim: '{text}' (reason: too short)")
                continue
            filtered.append(claim)
        logger.success(f'Quality filtering complete: kept {len(filtered)} of {len(claims)} claims.')
        return filtered

    def detect_from_document(self, sections: dict[str, str]) -> list[Claim]:
        all_claims: list[Claim] = []
        for section_name, section_text in sections.items():
            if not section_text.strip():
                continue
            claims = self.detect_claims(section_text, source_section=section_name)
            all_claims.extend(claims)
        all_claims.sort(key=lambda c: c.confidence, reverse=True)
        logger.success(f'Total ESG claims detected: {len(all_claims)}')
        return all_claims

    def _extract_claims_from_batch_with_retry(self, batch: list[dict], source_section: str) -> list[Claim]:
        candidates = []
        for unit in batch:
            candidates.append({'sentence_index': unit['sentence_index'], 'evidence': unit['evidence'], 'context': unit['context']})
        prompt_str = CLAIM_EXTRACTION_PROMPT.format(candidates=json.dumps(candidates, indent=2))
        try:
            return self._execute_extraction_run(prompt_str, batch, source_section)
        except Exception as exc:
            logger.warning(f'Failed extraction batch attempt. Error: {exc}. Retrying once...')
            try:
                return self._execute_extraction_run(prompt_str, batch, source_section)
            except Exception as final_exc:
                logger.error(f'Failed extraction batch retry. Error: {final_exc}')
                return []

    def _execute_extraction_run(self, prompt_str: str, batch: list[dict], source_section: str) -> list[Claim]:
        try:
            response = self.llm.invoke(prompt_str)
            response_text = response.content if hasattr(response, 'content') else str(response)
        except Exception as exc:
            exc_str = str(exc).lower()
            if 'finishreason' in exc_str or 'safety' in exc_str or 'block' in exc_str or ('finish_reason' in exc_str):
                logger.warning('Batch generation was blocked by safety filters or finish reason (e.g., FinishReason 19). Skipping this batch to prevent pipeline crash.')
                return []
            raise exc
        cleaned_json = self._clean_json_response(response_text)
        try:
            results = json.loads(cleaned_json)
        except Exception as exc:
            raise ValueError(f'Invalid JSON returned by Gemini: {exc}\nResponse: {response_text[:300]}')
        if not isinstance(results, list):
            raise TypeError('LLM response must be a JSON array of objects.')
        claims = []
        batch_map = {item['sentence_index']: item for item in batch}
        for obj in results:
            if not isinstance(obj, dict):
                continue
            is_actual = obj.get('is_actual_claim', False)
            if not is_actual:
                continue
            sent_idx = obj.get('sentence_index')
            if sent_idx is None or sent_idx not in batch_map:
                continue
            unit = batch_map[sent_idx]
            claims_list = obj.get('claims')
            if not isinstance(claims_list, list):
                continue
            for atomic in claims_list:
                if not isinstance(atomic, dict):
                    continue
                normalized_val = atomic.get('normalized_claim')
                if isinstance(normalized_val, list):
                    normalized = ' '.join([str(x) for x in normalized_val]).strip()
                elif normalized_val is not None:
                    normalized = str(normalized_val).strip()
                else:
                    normalized = ''
                category = atomic.get('category')
                raw_claim_type = atomic.get('claim_type')
                if not normalized or not category or (not raw_claim_type):
                    continue
                category = str(category).strip()
                raw_claim_type = str(raw_claim_type).strip()
                evidence_val = atomic.get('evidence')
                if isinstance(evidence_val, list):
                    evidence_sentence = ' '.join([str(x) for x in evidence_val]).strip()
                elif evidence_val is not None:
                    evidence_sentence = str(evidence_val).strip()
                else:
                    evidence_sentence = str(unit['evidence']).strip()
                esg_label = 'MIXED'
                category_upper = category.upper()
                if 'ENVIRONMENT' in category_upper or category_upper == 'E':
                    esg_label = 'E'
                elif 'SOCIAL' in category_upper or category_upper == 'S':
                    esg_label = 'S'
                elif 'GOVERNANCE' in category_upper or category_upper == 'G':
                    esg_label = 'G'
                claim_type_lower = raw_claim_type.lower()
                if 'target' in claim_type_lower or 'commitment' in claim_type_lower:
                    claim_type = 'commitment'
                elif 'performance' in claim_type_lower or 'quantitative' in claim_type_lower:
                    claim_type = 'quantitative'
                elif 'compliance' in claim_type_lower:
                    claim_type = 'compliance'
                else:
                    claim_type = claim_type_lower
                claims.append(Claim(text=normalized, evidence_sentence=evidence_sentence, evidence=unit['context'], start_char=unit['start'], end_char=unit['end'], source_section=source_section, claim_type=claim_type, confidence=round(unit['finbert_confidence'], 3), esg_label=esg_label, matched_patterns=['model_extracted']))
        return claims

    @staticmethod
    def _clean_json_response(text: str) -> str:
        text = text.strip()
        if text.startswith('```'):
            newline_idx = text.find('\n')
            if newline_idx != -1:
                text = text[newline_idx:].strip()
            if text.endswith('```'):
                text = text[:-3].strip()
        return text

    @staticmethod
    def _is_blocked(text: str) -> bool:
        for pattern in _BLOCKLIST_COMPILED:
            if pattern.search(text):
                return True
        return False

    @staticmethod
    def _fallback_sentences(text: str) -> list[str]:
        raw = re.split('(?<=[.!?])\\s+(?=[A-Z])', text)
        return [s.strip() for s in raw if s.strip()]