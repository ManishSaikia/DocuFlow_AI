import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    query: str = Field(..., description="The search query text")
    job_id: Optional[uuid.UUID] = Field(None, description="Optional document job ID to filter search")
    top_k: int = Field(5, ge=1, le=50, description="Number of final chunks to return")
    use_reranker: bool = Field(True, description="Whether to apply Stage 2 Cross-Encoder reranking")

class RetrievalDebugMetrics(BaseModel):
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rrf_score: float
    rerank_score: Optional[float] = None

class SearchResultItem(BaseModel):
    id: uuid.UUID
    content: str
    metadata: Dict[str, Any]
    retrieval_debug_metrics: RetrievalDebugMetrics

class SearchResponse(BaseModel):
    query: str
    total_result: int
    latency_ms: float
    results: List[SearchResultItem]