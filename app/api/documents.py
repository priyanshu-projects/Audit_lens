from __future__ import annotations
import shutil
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks
from loguru import logger
from app.models import DocumentOut, DocumentsListResponse, UploadResponse
from app.dependencies import get_documents, get_claims_df, invalidate_claims_cache, invalidate_documents_cache, invalidate_rag_cache
router = APIRouter(prefix='/documents', tags=['Documents'])
RAW_DIR = Path('data/raw')

def _build_doc_out(doc: dict, claims_df) -> DocumentOut:
    filename = doc.get('filename', doc.get('doc_id', ''))
    doc_id = doc.get('doc_id', filename)
    company = doc.get('company', doc.get('ticker', filename.split('_')[0].upper()))
    year = int(doc.get('year', 0))
    pages = int(doc.get('total_pages', len(doc.get('pages', []))))
    if not claims_df.empty and 'document' in claims_df.columns:
        doc_claims = claims_df[claims_df['document'].str.contains(filename, case=False, na=False)]
        total_claims = len(doc_claims)
        esg_breakdown = doc_claims['esg_label'].value_counts().to_dict() if total_claims else {}
    else:
        total_claims = 0
        esg_breakdown = {}
    return DocumentOut(doc_id=doc_id, filename=filename, company=company, year=year, total_pages=pages, total_claims=total_claims, esg_breakdown=esg_breakdown)

@router.get('', response_model=DocumentsListResponse)
async def list_documents() -> DocumentsListResponse:
    docs = get_documents()
    df = get_claims_df()
    out = [_build_doc_out(d, df) for d in docs]
    return DocumentsListResponse(total=len(out), documents=out)

@router.get('/{doc_id}', response_model=DocumentOut)
async def get_document(doc_id: str) -> DocumentOut:
    docs = get_documents()
    df = get_claims_df()
    match = next((d for d in docs if d.get('doc_id') == doc_id or d.get('filename', '').startswith(doc_id)), None)
    if not match:
        raise HTTPException(status_code=404, detail=f"Document '{doc_id}' not found.")
    return _build_doc_out(match, df)

def _run_pipeline_for_file(filename: str) -> None:
    import subprocess
    logger.info(f'Starting extraction pipeline for {filename}')
    result = subprocess.run(['python', 'scripts/run_pipeline_cli.py', '--input', 'data/raw', '--output', 'data/processed'], capture_output=True, text=True, cwd=str(Path.cwd()))
    if result.returncode == 0:
        logger.success(f'Extraction complete for {filename}')
        invalidate_claims_cache()
        invalidate_documents_cache()
        invalidate_rag_cache()
    else:
        logger.error(f'Extraction failed for {filename}: {result.stderr[-500:]}')

@router.post('/upload', response_model=UploadResponse)
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile=File(...)) -> UploadResponse:
    allowed = {'.pdf', '.htm', '.html'}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported file type '{suffix}'. Allowed: {allowed}")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DIR / file.filename
    content = await file.read()
    with open(dest, 'wb') as f:
        f.write(content)
    logger.info(f'Uploaded {file.filename} ({len(content)} bytes) → {dest}')
    background_tasks.add_task(_run_pipeline_for_file, file.filename)
    return UploadResponse(filename=file.filename, size_bytes=len(content), status='queued', message=f"File saved. Extraction pipeline queued for '{file.filename}'.")

@router.delete('/{doc_id}', status_code=204)
async def delete_document(doc_id: str) -> None:
    raw_files = list(RAW_DIR.glob(f'*{doc_id}*'))
    if not raw_files:
        raise HTTPException(status_code=404, detail=f"No raw file found matching '{doc_id}'.")
    for f in raw_files:
        f.unlink()
        logger.info(f'Deleted {f}')
    invalidate_claims_cache()
    invalidate_documents_cache()
    invalidate_rag_cache()