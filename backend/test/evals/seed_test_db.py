import asyncio
import uuid
from pathlib import Path
from sqlalchemy import select

from app.db.session import SyncSessionLocal, AsyncSessionLocal
from app.models.job import IngestionJob, JobStatus
from app.models.chunk import DocumentChunk
from app.worker.tasks import process_document_task

TARGET_JOB_ID = uuid.UUID("47dcd284-6420-4e37-869d-33e055b5da6f")
SAMPLE_PDF = Path(__file__).resolve().parents[3] / "samples" / "sample_invoice.pdf"

async def check_job_exists() -> bool:
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(IngestionJob.id).where(IngestionJob.id == TARGET_JOB_ID)
        )
        return result.scalar_one_or_none() is not None

def seed_database():
    exists = asyncio.run(check_job_exists())
    if exists:
        print(f"✅ Job {TARGET_JOB_ID} already exists in database. Skipping seed.")
        return

    print(f"🌱 Seeding database with job {TARGET_JOB_ID} using {SAMPLE_PDF}...")
    if not SAMPLE_PDF.exists():
        raise FileNotFoundError(f"Sample PDF not found at {SAMPLE_PDF}")

    with SyncSessionLocal() as session:
        job = IngestionJob(
            id=TARGET_JOB_ID,
            filename="sample_invoice.pdf",
            status=JobStatus.PENDING,
        )
        session.add(job)
        session.commit()

    # Process ingestion synchronously
    result = process_document_task(
        job_id=str(TARGET_JOB_ID),
        file_path=str(SAMPLE_PDF)
    )
    print(f"✅ Successfully seeded: {result}")

if __name__ == "__main__":
    seed_database()

