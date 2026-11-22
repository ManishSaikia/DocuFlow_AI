import asyncio
import json
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.config import settings
from app.db.session import AsyncSessionLocal
from app.models.chunk import DocumentChunk
from app.models.job import IngestionJob, JobStatus

OUTPUT_FILE = Path(__file__).parent / "golden_dataset.json"

class GoldenTestCase(BaseModel):
    test_id: str
    archetype: str  # "EXACT_LOOKUP" | "MULTI_HOP_SEMANTIC" | "ADVERSARIAL_NEGATIVE"
    job_id: str
    input_query: str
    ground_truth_chunks: List[int] = Field(
        ..., description="List of chunk_index integers containing the true answer"
    )
    expected_extracted_output: Dict[str, Any] = Field(
        default_factory=dict, description="Expected key-value extractions"
    )
    is_adversarial: bool = False

async def get_latest_completed_job_id() -> Optional[uuid.UUID]:
    """Fetch the most recent successfully completed ingestion job."""
    async with AsyncSessionLocal() as session:
        query = (
            select(IngestionJob.id)
            .where(IngestionJob.status == JobStatus.COMPLETED)
            .order_by(IngestionJob.created_at.desc())
            .limit(1)
        )
        result = await session.execute(query)
        return result.scalar_one_or_none()

async def fetch_document_chunks(job_id: uuid.UUID) -> List[Dict[str, Any]]:
    """Retrieve all chunks for the given job_id ordered by chunk_index."""
    async with AsyncSessionLocal() as session:
        query = (
            select(DocumentChunk)
            .where(DocumentChunk.job_id == job_id)
            .order_by(DocumentChunk.chunk_index.asc())
        )
        result = await session.execute(query)
        chunks = result.scalars().all()
        return [
            {
                "chunk_index": c.chunk_index,
                "content": c.content,
                "metadata": c.chunk_metadata or {},
            }
            for c in chunks
        ]

def generate_golden_dataset(job_id: str, chunks: List[Dict[str, Any]]) -> List[GoldenTestCase]:
    """
    Generates 25 structured evaluation test cases across 3 archetypes:
    1. Direct Alphanumeric / Exact Lookup (10 cases)
    2. Multi-Hop / Semantic Queries (10 cases)
    3. Adversarial / Negative Probes (5 cases)
    """
    test_cases: List[GoldenTestCase] = []
    case_num = 1

    # Safe chunk index references
    chunk_indices = [c["chunk_index"] for c in chunks] or [0]
    first_chunk = chunks[0] if chunks else {"chunk_index": 0, "content": ""}
    last_chunk = chunks[-1] if chunks else first_chunk

    # -------------------------------------------------------------
    # Archetype 1: Direct Alphanumeric / Exact Lookup (10 cases)
    # Stresses BM25 tsvector keyword matching
    # -------------------------------------------------------------
    exact_queries = [
        ("What is the exact document title or header?", {"document_title": "Income Statement (Product)"}, [first_chunk["chunk_index"]]),
        ("What is the total net income amount reported?", {"total_amount": 945.0}, [first_chunk["chunk_index"]]),
        ("What is the exact reporting date on the document?", {"document_date": "September 30, 2021"}, [first_chunk["chunk_index"]]),
        ("What is the line item description for net income?", {"line_items": [{"description": "Net income"}]}, [first_chunk["chunk_index"]]),
        ("What is the unit price and quantity for the primary net income line?", {"unit_price": 945.0, "quantity": 1.0}, [first_chunk["chunk_index"]]),
        ("Provide the total monetary amount in USD or dollars.", {"total_amount": 945.0}, [first_chunk["chunk_index"]]),
        ("What is the classified document type?", {"document_type": "FINANCIAL_STATEMENT"}, [first_chunk["chunk_index"]]),
        ("Extract chunk metadata regarding page numbers.", {"page_number": 1}, [first_chunk["chunk_index"]]),
        ("Verify if any sales tax or VAT is explicitly billed.", {"tax_amount": None}, [first_chunk["chunk_index"]]),
        ("What is the pre-tax subtotal listed in the document?", {"subtotal": None}, [first_chunk["chunk_index"]]),
    ]

    for q, expected, ground_truth in exact_queries:
        test_cases.append(
            GoldenTestCase(
                test_id=f"eval_{case_num:03d}",
                archetype="EXACT_LOOKUP",
                job_id=job_id,
                input_query=q,
                ground_truth_chunks=ground_truth,
                expected_extracted_output=expected,
                is_adversarial=False,
            )
        )
        case_num += 1

    # -------------------------------------------------------------
    # Archetype 2: Multi-Hop / Semantic Queries (10 cases)
    # Stresses Dense Embeddings + Cross-Encoder Reranking
    # -------------------------------------------------------------
    semantic_queries = [
        ("Can you summarize the bottom-line profitability for the company?", {"total_amount": 945.0}, chunk_indices[:2]),
        ("How did product revenues translate into net earnings in Q3?", {"total_amount": 945.0}, chunk_indices[:2]),
        ("Pull out all financial figures mentioned across the product statement.", {"total_amount": 945.0}, chunk_indices),
        ("What are the key financial performance metrics highlighted in this report?", {"document_title": "Income Statement (Product)"}, [first_chunk["chunk_index"]]),
        ("Provide a breakdown of all individual billed items and line expenses.", {"line_items": [{"description": "Net income"}]}, [first_chunk["chunk_index"]]),
        ("Which entity or recipient is this statement prepared for?", {"recipient": None}, [first_chunk["chunk_index"]]),
        ("What was the financial status of the business as of late 2021?", {"document_date": "September 30, 2021", "total_amount": 945.0}, [first_chunk["chunk_index"]]),
        ("Identify the issuing vendor or corporate entity responsible for this filing.", {"vendor_or_sender": None}, [first_chunk["chunk_index"]]),
        ("Summarize the overarching financial position outlined in the text.", {"total_amount": 945.0}, chunk_indices),
        ("What confidence score does the evidence in the text support?", {"confidence_score": 0.95}, [first_chunk["chunk_index"]]),
    ]

    for q, expected, ground_truth in semantic_queries:
        test_cases.append(
            GoldenTestCase(
                test_id=f"eval_{case_num:03d}",
                archetype="MULTI_HOP_SEMANTIC",
                job_id=job_id,
                input_query=q,
                ground_truth_chunks=ground_truth,
                expected_extracted_output=expected,
                is_adversarial=False,
            )
        )
        case_num += 1

    # -------------------------------------------------------------
    # Archetype 3: Adversarial / Negative Queries (5 cases)
    # Stresses Validator Grounding & Anti-Hallucination
    # -------------------------------------------------------------
    adversarial_queries = [
        ("What is the company's European VAT registration tax number in Germany?", {"vat_number": None}, []),
        ("Extract the IBAN and SWIFT wire transfer bank routing code.", {"iban": None, "swift": None}, []),
        ("What is the penalty clause for late delivery mentioned in Section 14.2?", {"penalty_clause": None}, []),
        ("What was the total marketing expenditure spent in Tokyo during Q4 2023?", {"tokyo_marketing": None}, []),
        ("Extract the physical shipping address of the third-party logistics courier.", {"shipping_address": None}, []),
    ]

    for q, expected, ground_truth in adversarial_queries:
        test_cases.append(
            GoldenTestCase(
                test_id=f"eval_{case_num:03d}",
                archetype="ADVERSARIAL_NEGATIVE",
                job_id=job_id,
                input_query=q,
                ground_truth_chunks=ground_truth,
                expected_extracted_output=expected,
                is_adversarial=True,
            )
        )
        case_num += 1

    return test_cases

async def build_and_save_dataset():
    print("🔍 Fetching latest completed ingestion job from PostgreSQL...")
    job_id = await get_latest_completed_job_id()

    if not job_id:
        print("⚠️ No completed job found in DB. Falling back to default test UUID.")
        job_id = uuid.UUID("47dcd284-6420-4e37-869d-33e055b5da6f")

    print(f"📄 Target Job ID: {job_id}")
    chunks = await fetch_document_chunks(job_id)
    print(f"📦 Retrieved {len(chunks)} chunks from database.")

    print("⚡ Synthesizing 25 golden evaluation test cases across 3 archetypes...")
    test_cases = generate_golden_dataset(str(job_id), chunks)

    # Serialize to JSON
    dataset_dict = [tc.model_dump() for tc in test_cases]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset_dict, f, indent=2)

    print(f"✅ Successfully exported {len(test_cases)} test cases to:")
    print(f"   {OUTPUT_FILE.resolve()}")

if __name__ == "__main__":
    asyncio.run(build_and_save_dataset())