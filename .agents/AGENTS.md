# Project Specification: Intelligent Task Assistant using Advanced RAG

## Objective
Build a highly accurate Retrieval-Augmented Generation (RAG) assistant that specializes in answering questions from a collection of task guidelines, instruction manuals, policy documents, and reference PDFs.
The assistant should prioritize factual correctness over creativity and minimize hallucinations. Every answer must be grounded in retrieved evidence from the uploaded documents.
The system is intended for day-to-day task assistance where the user asks questions about rules, edge cases, procedures, exceptions, or instructions, and the assistant returns accurate answers with supporting citations.

---

## Core Goals
The assistant must:
- Understand all uploaded PDFs.
- Retrieve the most relevant information.
- Handle conflicting instructions gracefully.
- Avoid making up information.
- Always explain where the answer came from.
- Respond quickly.
- Be easy to update by simply adding or replacing PDFs.

---

## Knowledge Source
The knowledge base consists only of uploaded PDFs.
Examples include:
- Task guidelines
- Rule books
- Instruction manuals
- SOPs
- FAQs
- Policy documents
- Training material

The system should support adding new PDFs without rebuilding the entire application.

---

## High-Level Architecture
```
PDFs ──→ PDF Parser ──→ Semantic Chunking ──→ Embedding Model ──→ FAISS Vector Database ──→ Hybrid Retriever (BM25 + Dense) ──→ Cross-Encoder Re-Ranker ──→ Top Context ──→ LLM ──→ Answer + Sources + Confidence
```

---

## Technical Details

### PDF Processing
Extract text while preserving:
- headings
- page numbers
- lists
- tables where possible
- document name

Metadata for every chunk:
- `document`
- `page`
- `section`
- `chunk_id`

### Intelligent Chunking
Avoid fixed-size chunking.
- Preserve logical sections and rule boundaries.
- Overlap adjacent chunks.
- Configurable chunk size.
- Avoid splitting lists or procedures.

### Embedding Pipeline
Generate embeddings for every chunk.
Store:
- original text
- embedding
- metadata
Embedding model should be replaceable without modifying the rest of the pipeline.

### FAISS Vector Store
Use FAISS as the vector database.
Support:
- create index
- save index
- load index
- incremental document addition
- document replacement

### Hybrid Retrieval
Combine:
- Dense Retrieval (Embeddings)
- BM25 Keyword Search
Merge results before reranking.

### Cross-Encoder Re-Ranking
- Initial retrieval: Top 20 chunks.
- Re-ranker: Cross Encoder.
- Output: Best 5 chunks for LLM context.

### Prompt Engineering
The system prompt must instruct the LLM to:
- Answer only from retrieved context.
- Never invent rules.
- Say "The documents do not contain enough information" if evidence is insufficient.
- Mention uncertainty when applicable.
- Prefer exact wording for important policies.
- Keep answers concise but complete.

### Citations
Every response must include sources (e.g. `Guidelines.pdf — Page 17`).

### Confidence Score
Estimate confidence (High, Medium, Low) using:
- retrieval similarity
- reranker score
- agreement among retrieved chunks

### Conflict Detection
If conflict is detected:
```
Possible conflicting instructions detected.
Document A: ...
Document B: ...
Please verify which version is current.
```

### Explainability Mode
Optional developer mode displaying:
- retrieved chunks
- similarity and reranker scores
- retrieval latency
- reasoning for source selection

### Search Features
Support queries such as:
- "What should I do if..."
- "Is this allowed?"
- "Show all rules regarding..."
- "Which document discusses..."
- "Summarize the policy about..."
- "What exceptions exist for..."
- "Compare the instructions in Document A and Document B."

### Document Management
Allow upload, remove, replace, rebuild index, and view documents.

### Logging
Log: query, retrieved chunks, latency, generated answer, and citations.

### Evaluation
Benchmark script to measure:
- Precision@k, Recall@k, MRR
- Response latency, citation accuracy, hallucination rate

---

## Tech Stack
- Python, FastAPI
- FAISS, Sentence Transformers, Rank-BM25, Cross-Encoder
- PyMuPDF, SQLite, Transformers

---

## Folder Structure
```
app/
├── ingestion/
├── retrieval/
├── reranker/
├── embeddings/
├── vectorstore/
├── prompts/
├── api/
├── evaluation/
├── models/
├── utils/
└── tests/
```
