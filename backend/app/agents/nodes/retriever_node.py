import uuid
from typing import Any, Dict
from app.agents.state import ExtractionGraphState
from app.db.session import AsyncSessionLocal
from app.services.embedder import embedding_service
from app.services.retriever import hybrid_search
from app.services.reranker import reranker_service

from app.core.tracing import tracer

import time

async def retrieve_context_node(state: ExtractionGraphState) -> Dict[str, Any]:
    """
    Entry node of the LangGraph workflow:
    1. Embeds the user query.
    2. Runs Stage 1 Hybrid RRF search in PostgreSQL.
    3. Runs Stage 2 Cross-Encoder reranking down to Top 5.
    4. Formats each chunk with explicit citation headers ([Chunk Index: X | Page: Y]).
    """
    with tracer.start_as_current_span("node.retriever") as span:
        start_time = time.perf_counter()
        job_id_str = state["job_id"]
        query = state.get("query") or "document title date vendor line items total amount"
        job_uuid = uuid.UUID(str(job_id_str))

        # Embed Query
        query_embedding = embedding_service.embed_query(query)

        # Hybrid RRF Search
        async with AsyncSessionLocal() as session:
            candidates = await hybrid_search(
                session=session,
                query_text=query,
                query_embedding=query_embedding,
                job_id=job_uuid,
                top_k=20,
                limit=20,
            )
        
        # Cross-Encoder Reranking
        reranked_chunks = reranker_service.rerank(
            query=query,
            candidates=candidates,
            top_n=5
        )

        # Format Chunks with explicit citation tags
        formatted_chunks = []
        for c in reranked_chunks:
            chunk_idx = c.get("chunk_index", 0)
            page_num = c.get("metadata", {}).get("page_number", 1)
            rerank_score = c.get("rerank_score", 0.0)
            raw_text = c.get("content", "").strip()

            formatted_chunk = {
                "chunk_index": chunk_idx,
                "page_number": page_num,
                "rerank_score": rerank_score,
                "content": raw_text,
                "formatted_text": (
                    f"[Chunk Index: {chunk_idx} | Page: {page_num} | Score: {rerank_score:.2f}]\n"
                    f"{raw_text}"
                ),
            }
            formatted_chunks.append(formatted_chunk)

        top_score = formatted_chunks[0]["rerank_score"] if formatted_chunks else 0.0

        span.set_attribute("rag.query", query)
        span.set_attribute("rag.retrieved_chunks_count", len(formatted_chunks))
        span.set_attribute("rag.top_rerank_score", float(top_score))

        retrieval_ms = round((time.perf_counter() - start_time) * 1000, 2)
        latencies = dict(state.get("node_latencies_ms") or {})
        latencies["retrieval_and_rerank_ms"] = retrieval_ms
        
        return {
            "retrieved_chunks": formatted_chunks,
            "retry_count": 0,
            "validation_errors": [],
            "status": "IN_PROGRESS",
            "node_latencies_ms": latencies
        }