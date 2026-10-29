from typing import Any, Dict, List, Optional, TypedDict

class ExtractionGraphState(TypedDict):
    """
    State dictonary that flows through every node in the LangGraph workflow
    """
    job_id: str
    query: str
    retrieved_chunks: List[Dict[str, Any]]
    extracted_data: Optional[Dict[str, Any]]
    validation_errors: List[str]
    retry_count: int
    status: str # "IN_PROGRESS", "VALIDATED", "FAILED_MAX_RETRIES"