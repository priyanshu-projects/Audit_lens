"""
test_pipeline.py -- Manual pipeline correctness test for AuditLens.
Run with: conda run -n auditlens python test_pipeline.py
"""
import sys
import os
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
from pathlib import Path

failures = []

def check(label, condition, detail=""):
    mark = "[PASS]" if condition else "[FAIL]"
    extra = f"  ({detail})" if (detail and not condition) else ""
    print(f"  {mark}  {label}{extra}")
    if not condition:
        failures.append(label)

# === 1. ClaimDetector ===
print("\n=== 1. ClaimDetector tests ===")

class MockClassificationResult:
    def __init__(self, esg_label="E", confidence=0.9):
        self.esg_label = esg_label
        self.confidence = confidence

class MockClassifier:
    def classify_batch(self, texts, source_sections=None):
        return [MockClassificationResult() for _ in texts]

class MockLLM:
    def invoke(self, prompt_str, *args, **kwargs):
        import json
        import re
        match = re.search(r"Input candidates \(JSON format\):\s*(\[.*?\])", prompt_str, re.DOTALL)
        if not match:
            class Response:
                content = "[]"
            return Response()
        try:
            candidates = json.loads(match.group(1))
        except Exception:
            class Response:
                content = "[]"
            return Response()
        results = []
        for cand in candidates:
            text = cand["evidence"]
            is_bad = any(term in text.lower() for term in [
                "proxy statement", "competitive factors", "net sales increased", "incorporated herein by reference"
            ])
            if is_bad:
                results.append({
                    "sentence_index": cand["sentence_index"],
                    "is_actual_claim": False,
                    "claims": []
                })
            else:
                claim_type = "quantitative"
                if "net zero" in text.lower() or "targets net zero" in text.lower():
                    claim_type = "commitment"
                elif "trir" in text.lower():
                    claim_type = "quantitative"
                results.append({
                    "sentence_index": cand["sentence_index"],
                    "is_actual_claim": True,
                    "claims": [
                        {
                            "normalized_claim": text.replace("We reduced ", "Reduced ").replace("Apple targets ", "Targets "),
                            "category": "Environmental" if "workforce" not in text.lower() and "trir" not in text.lower() else "Social",
                            "claim_type": claim_type,
                            "evidence": text
                        }
                    ]
                })
        class Response:
            content = json.dumps(results)
        return Response()

from src.extraction.claim_detector import ClaimDetector
detector = ClaimDetector(classifier=MockClassifier(), llm=MockLLM())

GOOD = [
    ("Scope1 emission reduction",
     "We reduced Scope 1 and Scope 2 emissions by 40% since 2019."),
    ("Net-zero commitment",
     "Apple targets net zero across our entire supply chain by 2030."),
    ("Workforce diversity pct",
     "Women represent 35% of our global workforce as of fiscal year 2025."),
    ("Renewable energy 100pct",
     "Renewable energy accounted for 100% of our global electricity use."),
    ("Safety TRIR metric",
     "Our TRIR (Total Recordable Incident Rate) was 0.13 in fiscal 2024."),
]

BAD = [
    ("Proxy boilerplate",
     "The information required by this Item will be included in the Proxy Statement."),
    ("Competitive factors boilerplate",
     "Principal competitive factors important to the Company include price, product and service quality."),
    ("Revenue comparison financial",
     "Net sales increased 4% or 15.2 billion compared to 2024."),
    ("Incorporated by reference",
     "This information is incorporated herein by reference from the Proxy Statement."),
]

for label, text in GOOD:
    r = detector.detect_claims(text, "environmental")
    check(f"DETECTED: {label}", len(r) > 0, f"got 0 claims for: {text[:60]}")

for label, text in BAD:
    r = detector.detect_claims(text, "general")
    check(f"BLOCKED:  {label}", len(r) == 0, f"leaked {len(r)} claims for: {text[:60]}")

# === 2. InternalChecker (L1) ===
print("\n=== 2. InternalChecker (L1) tests ===")

from src.extraction.pdf_extractor import ExtractedDocument, TableData
from src.consistency.internal_checker import InternalChecker, ConsistencyResult

doc = ExtractedDocument(source_path=Path("test.pdf"))
doc.raw_text = "Scope 1 emissions: 850,000 tCO2e reduced by 40% since 2019."
table = TableData(
    page_number=5,
    headers=["Year", "Scope 1 (tCO2e)", "Scope 2 (tCO2e)"],
    rows=[["2019", "1,420,000", "2,390,000"], ["2024", "850,000", "950,000"]],
)
doc.tables = [table]
l1 = InternalChecker()

r = l1.check("We reduced Scope 1 emissions to 850,000 tCO2e, a 40% reduction since 2019.", 0, doc)
check("L1 SUPPORTED when number matches table", r.status == "SUPPORTED", f"got {r.status}")

r2 = l1.check("We are committed to reducing our environmental footprint.", 0, doc)
check("L1 SKIPPED for non-quantitative claim", r2.status == "SKIPPED", f"got {r2.status}")

doc2 = ExtractedDocument(source_path=Path("test.pdf"))
doc2.raw_text = "No relevant data here."
doc2.tables = []
r3 = l1.check("We achieved 999999 tonnes of CO2 reduction.", 0, doc2)
check("L1 UNSUPPORTED when number not in doc", r3.status == "UNSUPPORTED", f"got {r3.status}")

# === 3. Aggregator ===
print("\n=== 3. Aggregator tests ===")
from src.consistency.aggregator import aggregate_results

l_skip = ConsistencyResult(level=1, status="SKIPPED", note="No numeric.")
agg = aggregate_results(l1=l_skip, l2=l_skip, nlp_flag="LIKELY_CONSISTENT", esg_label="E", nlp_confidence=0.9)
check("All SKIPPED + LIKELY_CONSISTENT => CONSISTENT", agg.verdict == "CONSISTENT", f"got {agg.verdict}")

agg2 = aggregate_results(l1=l_skip, l2=l_skip, nlp_flag="HIGH_RISK", esg_label="E", nlp_confidence=0.85)
check("All SKIPPED + HIGH_RISK => INCONSISTENT", agg2.verdict == "INCONSISTENT", f"got {agg2.verdict}")

l1_s = ConsistencyResult(level=1, status="SUPPORTED", note="Found table.", evidence="850000")
l2_s = ConsistencyResult(level=2, status="SKIPPED", note="No prior.")
agg3 = aggregate_results(l1=l1_s, l2=l2_s, esg_label="E")
check("SUPPORTED L1 + SKIPPED L2 => CONSISTENT", agg3.verdict == "CONSISTENT", f"got {agg3.verdict}, risk={agg3.risk_score}")

l1_u = ConsistencyResult(level=1, status="UNSUPPORTED", note="Missing.")
l2_u = ConsistencyResult(level=2, status="UNSUPPORTED", note="Changed 35pct.")
agg4 = aggregate_results(l1=l1_u, l2=l2_u, esg_label="E")
check("UNSUPPORTED L1+L2 => INCONSISTENT or HIGH_RISK",
      agg4.verdict in ("INCONSISTENT", "HIGH_RISK"), f"got {agg4.verdict}")

l3_hr = ConsistencyResult(level=3, status="HIGH_RISK", note="SBTi conflict: claimed 2030, verified 2050.")
agg5 = aggregate_results(l1=l1_s, l2=l2_s, l3=l3_hr, esg_label="E")
check("L3 HIGH_RISK always => HIGH_RISK verdict", agg5.verdict == "HIGH_RISK", f"got {agg5.verdict}")

agg6 = aggregate_results(esg_label="MIXED", nlp_confidence=0.92)
check("MIXED + confidence>0.8 => NOT_ESG_CLAIM", agg6.verdict == "NOT_ESG_CLAIM", f"got {agg6.verdict}")

# === 4. SbtiChecker ===
print("\n=== 4. SbtiChecker tests ===")
from src.consistency.sbti_checker import SbtiChecker
from src.extraction.claim_detector import Claim

sbti = SbtiChecker()
c_quant = Claim(text="We reduced emissions by 40%.", start_char=0, end_char=30, claim_type="quantitative")
r = sbti.check(c_quant, "Apple Inc.")
check("Non-commitment claim => SKIPPED", r.status == "SKIPPED", f"got {r.status}")

c_commit = Claim(text="We target net zero by 2040.", start_char=0, end_char=27, claim_type="commitment")
r2 = sbti.check(c_commit, "XYZ Nonexistent Corp 999")
check("Unknown company => UNSUPPORTED", r2.status == "UNSUPPORTED", f"got {r2.status}")

# === Summary ===
print("\n" + "=" * 45)
total = len(GOOD) + len(BAD) + 3 + 6 + 2
if failures:
    print(f"FAILED {len(failures)}/{total} checks:")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
else:
    print(f"ALL {total} CHECKS PASSED")
