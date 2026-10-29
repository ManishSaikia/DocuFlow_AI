import asyncio
from app.db.session import AsyncSessionLocal
from app.models.job import IngestionJob, JobStatus
from app.services.embedder import embedding_service
from app.services.retriever import hybrid_search
from app.services.reranker import reranker_service
from sqlalchemy import select


async def main():
    async with AsyncSessionLocal() as session:
        # 1. Grab the latest completed job
        result = await session.execute(
            select(IngestionJob)
            .where(IngestionJob.status == JobStatus.COMPLETED)
            .order_by(IngestionJob.created_at.desc())
        )
        job = result.scalars().first()
        if not job:
            print("No completed jobs found in database to search against!")
            return

        print(f"=== Running Two-Stage Retrieval on: {job.filename} ===")
        query = "sample PDF file"
        print(f"User Query: '{query}'\n")

        # ---------------------------------------------------------
        # STAGE 1: Hybrid RRF Search (pgvector + full-text in PostgreSQL)
        # ---------------------------------------------------------
        print("[Stage 1] Executing PostgreSQL Hybrid RRF Search (Top 20)...")
        query_embedding = embedding_service.embed_query(query)
        candidates = await hybrid_search(
            session=session,
            query_text=query,
            query_embedding=query_embedding,
            job_id=job.id,
            top_k=20,
            limit=20,
        )
        print(f"-> Stage 1 returned {len(candidates)} candidates.\n")

        # ---------------------------------------------------------
        # STAGE 2: Cross-Encoder Reranking
        # ---------------------------------------------------------
        print("[Stage 2] Running Cross-Encoder Semantic Reranker (Top 3)...")
        final_chunks = reranker_service.rerank(
            query=query,
            candidates=candidates,
            top_n=3,
        )

        print("\n=== FINAL TOP 3 RERANKED CHUNKS ===")
        for idx, chunk in enumerate(final_chunks, 1):
            print(f"Rank #{idx}")
            print(f"  Rerank Score: {chunk['rerank_score']:.4f}")
            print(f"  RRF Score:    {chunk['rrf_score']:.5f}")
            print(f"  Dense Rank:   {chunk['dense_rank']} | Sparse Rank: {chunk['sparse_rank']}")
            print(f"  Preview:      {chunk['content'][:120]}...\n")


if __name__ == "__main__":
    asyncio.run(main())
