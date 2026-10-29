import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict
from app.models.job import JobStatus

class IngestResponse(BaseModel):
    job_id: uuid.UUID
    filename: str
    status: JobStatus = JobStatus.PENDING
    message: str = "Document accepted for asynchronous processing"

    model_config = ConfigDict(from_attributes=True)

class JobStatusResponse(BaseModel):
    job_id: uuid.UUID
    filename: str
    status: JobStatus
    chunks_count: int = 0
    retry_count: int = 0
    error_log: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)