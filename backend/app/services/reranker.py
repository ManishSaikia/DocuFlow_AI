from typing import Any, Dict, List
from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class RerankerService:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        print(f"Loading Cross-Encoder model: {model_name}...")
        self.model = CrossEncoder(model_name)
        print(f"Cross-Encoder model loaded successfully")

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Takes a list of retrieved chunk dictionaries from Stage 1 (RRF),
        scores them jointly with the user query using cross-attention,
        and returns the top_n most relevant chunks.
        """

        if not candidates:
            return []

        pairs = [[query, candidate["content"]] for candidate in candidates]

        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores):
            candidate["rerank_score"] = float(score)
        
        reranked_results = sorted(
            candidates,
            key = lambda x: x["rerank_score"],
            reverse = True
        )

        return reranked_results[:top_n]

reranker_service = RerankerService()