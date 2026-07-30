from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from config.settings import app_cfg, gemini_cfg
from src.rag.vector_store import SearchResult, VectorStore

@dataclass
class AuditObservation:
    claim_text: str
    structured_note: str
    standards_cited: list[str] = field(default_factory=list)
    retrieved_chunks: list[SearchResult] = field(default_factory=list)
    retrieval_confidence: float = 0.0
    low_confidence_flag: bool = False
    shap_narrative: str = ''

    def to_dict(self) -> dict:
        return {'claim_text': self.claim_text, 'structured_note': self.structured_note, 'standards_cited': self.standards_cited, 'retrieval_confidence': round(self.retrieval_confidence, 4), 'low_confidence_flag': self.low_confidence_flag, 'shap_narrative': self.shap_narrative}
AUDIT_OBSERVATION_TEMPLATE = 'You are a senior ESG auditor at a Big 4 firm.\nYou are reviewing a sustainability claim extracted from a company\'s ESG report.\n\nRELEVANT REGULATORY STANDARDS (retrieved from GRI/TCFD/SASB/ISSB knowledge base):\n{context}\n\nESG CLAIM UNDER REVIEW:\n"{claim}"\n\nEXPLAINABILITY NOTE (from FinBERT SHAP analysis):\n{shap_narrative}\n\n{auditor_question}\n\nGenerate a structured audit observation in exactly this format:\n\n**Standard Reference:** [cite the most relevant standard from the context above]\n\n**Compliance Assessment:** [COMPLIANT / PARTIALLY COMPLIANT / NON-COMPLIANT / INSUFFICIENT EVIDENCE]\n\n**Evidence Gaps Identified:**\n- [List specific missing disclosures or data points required by the cited standard]\n- [Each gap on its own line]\n\n**Risk Level:** [LOW / MEDIUM / HIGH / CRITICAL]\n\n**Recommended Auditor Action:**\n[Concrete next step — e.g. "Request the company\'s CDP submission for comparison" or "Verify the 2019 baseline figure against the prior-year 10-K"]\n\n**Grounding Note:** Only use information from the regulatory standards provided above. Do not introduce external knowledge.\n'
AUDITOR_QUESTION_PREFIX = 'AUDITOR QUESTION: '

class RagChain:
    LOW_CONFIDENCE_THRESHOLD = 0.4

    def __init__(self, vector_store: Optional[VectorStore], llm: ChatGoogleGenerativeAI) -> None:
        self.vector_store = vector_store
        self.llm = llm
        self._prompt = ChatPromptTemplate.from_template(AUDIT_OBSERVATION_TEMPLATE)
        self._output_parser = StrOutputParser()
        logger.info('RagChain initialised')

    @classmethod
    def build(cls, vector_store: Optional[VectorStore]=None, api_key: str=gemini_cfg.api_key, model_name: str=gemini_cfg.model_name) -> 'RagChain':
        models_to_try = ['gemini-2.0-flash', 'gemini-2.5-flash']
        ordered_models = []
        for m in [model_name] + models_to_try:
            if m and m not in ordered_models:
                ordered_models.append(m)
        llm_instances = [ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0.1, max_output_tokens=8192) for model in ordered_models]
        primary_llm = llm_instances[0]
        llm = primary_llm.with_fallbacks(llm_instances[1:]) if len(llm_instances) > 1 else primary_llm
        logger.info(f'Gemini LLM configured with model: {ordered_models[0]} (fallbacks: {ordered_models[1:]})')
        return cls(vector_store=vector_store, llm=llm)

    def query(self, question: str, top_k: int=5) -> dict:
        results = self.vector_store.search(question, top_k=top_k)
        if not results:
            return {'answer': 'The knowledge base does not contain enough information to answer this question.', 'source_documents': []}
        context = self._format_context(results)
        prompt = ChatPromptTemplate.from_template('You are an expert in ESG regulatory standards (GRI, TCFD, SASB, ISSB, EU CSRD).\n\nAnswer the following question using ONLY the provided standard excerpts.\nIf the answer is not in the excerpts, say so explicitly.\n\nREGULATORY STANDARD EXCERPTS:\n{context}\n\nQUESTION: {question}\n\nAnswer:')
        chain = prompt | self.llm | self._output_parser
        try:
            answer = chain.invoke({'context': context, 'question': question})
        except Exception as exc:
            logger.error(f'Query failed: {exc}')
            answer = f'Error generating answer: {exc}'
        source_docs = [{'page_content': r.chunk.text, 'score': r.score, 'metadata': {'document': r.chunk.source_doc, 'page': r.chunk.page_number, 'standard': r.chunk.standard, 'section': r.chunk.section}} for r in results]
        return {'answer': answer, 'source_documents': source_docs}

    def generate_audit_observation(self, claim: str, shap_narrative: str='', question: Optional[str]=None, top_k: int=app_cfg.rag_top_k, standard_filter: Optional[str]=None) -> AuditObservation:
        logger.info(f"Generating audit observation for: '{claim[:60]}...'")
        search_results = self.vector_store.search(query=claim, top_k=top_k, standard_filter=standard_filter)
        if not search_results:
            logger.warning('No standard chunks retrieved — knowledge base may be empty')
            return AuditObservation(claim_text=claim, structured_note='⚠️ Unable to generate audit observation: knowledge base is empty. Please run vector_store.build_from_directory() with GRI/TCFD/SASB/ISSB PDFs.', shap_narrative=shap_narrative)
        context = self._format_context(search_results)
        avg_score = sum((r.score for r in search_results)) / len(search_results)
        low_confidence = avg_score < self.LOW_CONFIDENCE_THRESHOLD
        if low_confidence:
            logger.warning(f'Low retrieval confidence (avg={avg_score:.3f}) — standard chunks may not be closely relevant')
        auditor_q = ''
        if question:
            auditor_q = f'{AUDITOR_QUESTION_PREFIX}{question}'
        prompt_input = {'context': context, 'claim': claim, 'shap_narrative': shap_narrative or 'No SHAP explanation available.', 'auditor_question': auditor_q}
        try:
            chain = self._prompt | self.llm | self._output_parser
            structured_note = chain.invoke(prompt_input)
            logger.success('Audit observation generated')
        except Exception as exc:
            logger.error(f'Gemini generation failed: {exc}')
            structured_note = f'Generation failed: {exc}'
        standards_cited = self._extract_standards(structured_note, search_results)
        return AuditObservation(claim_text=claim, structured_note=structured_note, standards_cited=standards_cited, retrieved_chunks=search_results, retrieval_confidence=round(avg_score, 4), low_confidence_flag=low_confidence, shap_narrative=shap_narrative)

    def answer_question(self, question: str, top_k: int=app_cfg.rag_top_k) -> str:
        results = self.vector_store.search(question, top_k=top_k)
        if not results:
            return 'Knowledge base is empty. Please build the FAISS index first.'
        context = self._format_context(results)
        chatbot_prompt = ChatPromptTemplate.from_template('You are an expert in ESG regulatory standards (GRI, TCFD, SASB, ISSB, EU CSRD).\n\nAnswer the following question using ONLY the provided standard excerpts.\nIf the answer is not in the excerpts, say so explicitly.\n\nREGULATORY STANDARD EXCERPTS:\n{context}\n\nQUESTION: {question}\n\nAnswer:')
        chain = chatbot_prompt | self.llm | self._output_parser
        try:
            return chain.invoke({'context': context, 'question': question})
        except Exception as exc:
            logger.error(f'Chatbot generation failed: {exc}')
            return f'Error generating answer: {exc}'

    @staticmethod
    def _format_context(results: list[SearchResult]) -> str:
        lines = []
        for r in results:
            lines.append(f'[{r.rank}] {r.chunk.standard} — {r.chunk.section} (source: {r.chunk.source_doc}, page {r.chunk.page_number}, relevance: {r.score:.3f})\n{r.chunk.text}')
        return '\n\n---\n\n'.join(lines)

    @staticmethod
    def _extract_standards(note: str, results: list[SearchResult]) -> list[str]:
        standards = set()
        for standard in ['GRI', 'TCFD', 'SASB', 'ISSB', 'CSRD', 'IFRS']:
            if standard in note.upper():
                standards.add(standard)
        for r in results:
            if r.chunk.standard and r.chunk.standard != 'UNKNOWN':
                standards.add(r.chunk.standard)
        return sorted(standards)