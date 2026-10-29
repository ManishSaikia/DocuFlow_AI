import uuid
import logging
from celery import Task
from app.worker.celery_app import celery_app
from app.db.session import SyncSessionLocal
from app.models.job import IngestionJob, JobStatus

from typing import Optional
from pathlib import Path
from app.services.chunker import parse_and_chunk_pdf

from app.services.embedder import embedding_service
from sqlalchemy import func

logger = logging.getLogger(__name__)

class IngestionTaskWithRetry(Task):

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        job_id = args[0] if args else kwargs.get("job_id")
        logger.warning(f"Task {task_id} (Job {job_id}) retrying: {exc}")

        if job_id:
            with SyncSessionLocal() as session:
                job = session.get(IngestionJob, uuid.UUID(str(job_id)))

                if job:
                    job.retry_count = self.request.retries
                    session.commit()
        
        super().on_retry(exc, task_id, args, kwargs, einfo)
    
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        job_id = args[0] if args else kwargs.get("job_id")
        logger.error(f"Task {task_id} (Job {job_id}) permanently failed: {exc}")
        
        if job_id:
            with SyncSessionLocal() as session:
                job = session.get(IngestionJob, uuid.UUID(str(job_id)))

                if job:
                    job.status = JobStatus.FAILED
                    job.error_log = str(einfo)
                    session.commit()
        
        super().on_failure(exc, task_id, args, kwargs, einfo)

@celery_app.task(
    base=IngestionTaskWithRetry,
    bind=True,
    autoretry_for=(ValueError, ConnectionError),
    retry_backoff=2,
    retry_jitter=True,
    max_retries=3
)
def process_document_task(
    self, 
    job_id: str,
    file_path: Optional[str] = None,
    simulate_failure: bool = False
    ):
    logger.info(f"Processing job {job_id} (Attempt {self.request.retries + 1}/4)...")
    parsed_uuid = uuid.UUID(str(job_id))

    with SyncSessionLocal() as session:
        job = session.get(IngestionJob, parsed_uuid)
        if not job:
            raise ValueError(f"Job {job_id} not found")
        job.status = JobStatus.PROCESSING
        session.commit()
    
    if simulate_failure:
        raise ValueError(f"Simulated extraction failure on attempt {self.request.retries + 1}")
    
    chunks_count = 0
    if file_path:
        path_obj = Path(file_path)
        if not path_obj.exists():
            raise ValueError(f"Uploaded file not found at: {file_path}")
        
        logger.info(f"Extracting chunks from: {file_path}")
        chunks = parse_and_chunk_pdf(path_obj, parsed_uuid)
        chunks_count = len(chunks)
        logger.info(f"Generated {chunks_count} chunks for job {job_id}")
        
        if chunks:
            texts = [chunk.content for chunk in chunks]

            logger.info(f"Generating dense embeddings for {chunks_count} chunks...")
            embeddings = embedding_service.embed_documents(texts, batch_size=64)

            for chunk, vector in zip(chunks, embeddings):
                chunk.embedding = vector
                chunk.search_vector = func.to_tsvector('english', chunk.content)

        with SyncSessionLocal() as session:
            session.add_all(chunks)
            job = session.get(IngestionJob, parsed_uuid)
            if job:
                job.status = JobStatus.COMPLETED
            session.commit()
    else:
        with SyncSessionLocal() as session:
            job = session.get(IngestionJob, parsed_uuid)
            if job:
                job.status = JobStatus.COMPLETED
            session.commit()
        
    return {"status": "SUCCESS", "job_id": job_id, "chunks_count": chunks_count}