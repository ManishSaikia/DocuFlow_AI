import shutil
import uuid
from pathlib import Path
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.models.chunk import DocumentChunk
from app.models.job import IngestionJob, JobStatus
from app.schemas.document import IngestResponse, JobStatusResponse
from app.worker.tasks import process_document_task

import time
from app.services.embedder import embedding_service
from app.services.retriever import hybrid_search
from app.services.reranker import reranker_service
from app.schemas.search import SearchRequest, SearchResponse, SearchResultItem, RetrievalDebugMetrics

from app.agents.graph import extraction_graph
from app.schemas.extraction import ExtractionRequest, ExtractionResponse

router = APIRouter()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload document for asynchronous ingestion"
)
async def ingest_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    # File Validation
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "Only PDF documents are supported currently."
        )
    
    # Generate Job ID and Save file to disk
    job_id = uuid.uuid4()
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create Injestion record in DB
    job = IngestionJob(
        id=job_id,
        filename=file.filename,
        status=JobStatus.PENDING
    )
    db.add(job)
    await db.commit()
    await db.refresh(job)

    # Dispatch Celery Background Task
    process_document_task.delay(str(job_id), file_path=str(file_path))

    return IngestResponse(
        job_id=job.id,
        filename=job.filename,
        status=job.status
    )

@router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll document ingestion status and chunk count"
)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db)
):
    job = await db.get(IngestionJob, job_id)

    if not job:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail= f"Job {job_id} not found"
        )
    
    count_query = select(func.count(DocumentChunk.id)).where(
        DocumentChunk.job_id == job_id
    )
    count_result = await db.execute(count_query)
    chunks_count = count_result.scalar() or 0

    return JobStatusResponse(
        job_id=job.id,
        filename=job.filename,
        status=job.status,
        chunks_count=chunks_count,
        retry_count=job.retry_count,
        error_log=job.error_log,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Diagnostic Hybrid Search + Reranking endpoint"
)
async def search_documents(
    payload: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    start_time = time.perf_counter()

    query_embedding = embedding_service.embed_query(payload.query)

    candidates = await hybrid_search(
        session=db,
        query_text=payload.query,
        query_embedding=query_embedding,
        job_id=payload.job_id,
        top_k=20,
        limit=20,
    )

    if payload.use_reranker and candidates:
        final_candidates = reranker_service.rerank(
            query=payload.query,
            candidates=candidates,
            top_n=payload.top_k,
        )
    else:
        final_candidates = candidates[: payload.top_k]
    
    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    result = [
        SearchResultItem(
            id=c["id"],
            content=c["content"],
            metadata=c["metadata"],
            retrieval_debug_metrics=RetrievalDebugMetrics(
                dense_rank=c.get("dense_rank"),
                sparse_rank=c.get("sparse_rank"),
                rrf_score=c.get("rrf_score", 0.0),
                rerank_score=c.get("rerank_score"),
            ),
        )
        for c in final_candidates
    ]

    return SearchResponse(
        query = payload.query,
        total_result = len(result),
        latency_ms = latency_ms,
        results = result,
    )

@router.post(
    "/extract",
    response_model=ExtractionResponse,
    summary="Run Stateful Multi-Agent Extraction Workflow"
)
async def extract_document_data(
    payload: ExtractionRequest,
    db: AsyncSession = Depends(get_db)
):
    job = await db.get(IngestionJob, payload.job_id)
    if not job:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail=f"Job {payload.job_id} not found"
        )
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail=f"Job {payload.job_id} status is '{job.status.value}'. Ingestion must be COMPLETED before extraction."
        )
    
    initial_state = {
        "job_id": str(payload.job_id),
        "query": payload.query,
        "retrieved_chunks": [],
        "extracted_data": None,
        "validation_errors": [],
        "retry_count": 0,
        "status": "IN_PROGRESS"
    }

    final_state = await extraction_graph.ainvoke(initial_state)

    extracted = final_state.get("extracted_data")
    citations = extracted.get("citations", []) if extracted else []

    final_status = final_state.get("status", "IN_PROGRESS")

    if final_state.get("validation_errors") and final_state.get("retry_count", 0) >= 3:
        final_status = "FAILED_MAX_RETRIES"
    
    return ExtractionResponse(
        job_id=str(payload.job_id),
        status=final_status,
        retry_count=final_state.get("retry_count", 0),
        extracted_data=extracted,
        citations=citations,
        validation_errors=final_state.get("validation_errors", [])
    )
    