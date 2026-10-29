import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import redis.asyncio as aioredis

from app.core.config import settings
from app.services.embedder import embedding_service

SCHEMA_VERSION = "v1"
SIMILARITY_THRESHOLD = 0.95 # Cosine similarity >= 0.95 triggers semantic hit
DEFAULT_TTL = 3600 # 1 hour

class SemanticCacheService:
    def __init__(self):
        self.redis: aioredis.Redis = aioredis.from_url(
            settings.redis_url,
            decode_responses = True
        )
    
    def _hash_query(self, query: str) -> str:
        """Normalized SHA-256 hash for exact match lookup."""
        normalized = " ".join(query.strip().lower().split())
        key_raw = f"{SCHEMA_VERSION}:{normalized}"
        return hashlib.sha256(key_raw.encode("utf-8")).hexdigest()

    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Fast cosine similarity using NumPy."""
        a = np.array(vec_a, dtype=np.float32)
        b = np.array(vec_b, dtype=np.float32)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    async def lookup(
        self,
        job_id: str,
        query: str
    ) -> Tuple[Optional[Dict[str, Any]], Optional[List[float]], Optional[str]]:    
        """
        Two-Tier Cache Lookup:
        Returns:
            (cached_payload, query_embedding, cache_tier_hit)
            - If Tier 1 hit: (payload, None, "TIER_1_EXACT")
            - If Tier 2 hit: (payload, query_embedding, "TIER_2_SEMANTIC")
            - If Miss:       (None, query_embedding, None)
        """
        query_hash = self._hash_query(query)
        exact_key = f"cache:{job_id}:exact:{query_hash}"

        # TIER 1: Exact Match
        exact_cached = await self.redis.get(exact_key)
        if exact_cached:
            return json.loads(exact_cached), None, "TIER_1_EXACT"

        # TIER 2: Document-Scoped Semantic Vector Match
        query_embedding = embedding_service.embed_query(query)
        semantic_entries_raw = await self.redis.hgetall(f"cache:{job_id}:semantic_queries")

        if semantic_entries_raw:
            best_similarity = -1.0
            best_payload = None

            for q_hash, entry_json in semantic_entries_raw.items():
                entry = json.loads(entry_json)
                cached_vec = entry["embedding"]
                sim = self._cosine_similarity(query_embedding, cached_vec)

                if sim > best_similarity:
                    best_similarity = sim
                    best_payload = entry["response"]
            
            if best_similarity >= SIMILARITY_THRESHOLD and best_payload is not None:
                return best_payload, query_embedding, f"TIER_2_SEMANTIC (sim={best_similarity:.3f})"
        
        return None, query_embedding, None
    
    async def store(
        self,
        job_id: str,
        query: str,
        query_embedding: List[float],
        response_payload: Dict[str, Any],
        ttl: int = DEFAULT_TTL
    ) -> None:
        """
        Populate both Tier 1 and Tier 2 caches for the document.
        """
        query_hash = self._hash_query(query)
        exact_key = f"cache:{job_id}:exact:{query_hash}"
        semantic_hash_key = f"cache:{job_id}:semantic_queries"

        await self.redis.setex(
            exact_key,
            ttl,
            json.dumps(response_payload)
        )

        semantic_entry = {
            "query": query,
            "embedding": query_embedding,
            "response": response_payload
        }
        await self.redis.hset(
            semantic_hash_key,
            query_hash,
            json.dumps(semantic_entry)
        )
        await self.redis.expire(semantic_hash_key, ttl)

semantic_cache = SemanticCacheService()