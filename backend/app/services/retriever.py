import uuid
from typing import Any, Dict, List, Optional
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

HYBRID_SEARCH_SQL = text("""
WITH dense_search AS (
    SELECT id, chunk_index, content, metadata, ROW_NUMBER() OVER(ORDER BY embedding <=> CAST(:query_embedding AS vector)) AS dense_rank
    FROM document_chunks
    WHERE (CAST(:job_id AS uuid) IS NULL or job_id = :job_id) AND embedding IS NOT NULL
    LIMIT :top_k
),
sparse_search AS (
    SELECT id, chunk_index, content, metadata, ROW_NUMBER() OVER(ORDER BY ts_rank_cd(search_vector, websearch_to_tsquery('english', :query_text)) DESC) AS sparse_rank
    FROM document_chunks
    WHERE (CAST(:job_id AS uuid) IS NULL or job_id = :job_id) AND search_vector @@ websearch_to_tsquery('english', :query_text)
    LIMIT :top_k
)
SELECT
    COALESCE(d.id, s.id) as id,
    COALESCE(d.content, s.content) as content,
    COALESCE(d.metadata, s.metadata) as metadata,
    COALESCE(d.chunk_index, s.chunk_index) as chunk_index,
    d.dense_rank,
    s.sparse_rank,
    (COALESCE(1.0 / (60.0 + d.dense_rank), 0.0) + COALESCE(1.0 / (60.0 + s.sparse_rank), 0.0)) AS rrf_score
FROM dense_search as d
FULL OUTER JOIN sparse_search as s ON d.id = s.id
ORDER BY rrf_score DESC
LIMIT :limit;
""")

async def hybrid_search(
    session: AsyncSession,
    query_text: str,
    query_embedding: List[float],
    job_id: Optional[uuid.UUID] = None,
    top_k: int = 20,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """
    Executes a hybrid search combining dense semantic vector search (Cosine Distance)
    and sparse lexical search (tsvector/BM25) using Reciprocal Rank Fusion (RRF) directly in PostgreSQL.
    """

    embedding_str = f"[{','.join(str(x) for x in query_embedding)}]"

    result = await session.execute(
        HYBRID_SEARCH_SQL,
        {
            "query_text": query_text,
            "query_embedding": embedding_str,
            "job_id": job_id,
            "top_k": top_k,
            "limit": limit,
        },
    )

    rows = result.mappings().all()
    return [dict(row) for row in rows]