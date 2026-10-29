from typing import List, Generator
from fastembed import TextEmbedding

DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"

class EmbeddingService:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        print(f"Loading embedding mode: {model_name}...")
        self.model = TextEmbedding(model_name=model_name)
        print("Embedding model loaded successfully")
    
    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """
        Embeds a list of document chunk strings in batches of batch_size.
        Returns a list of 768-dimensional float vectors
        """
        if not texts:
            return []
        
        embeddings_generator: Generator = self.model.embed(texts, batch_size=batch_size)
        
        return [embedding.tolist() for embedding in embeddings_generator]
    
    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single search query string.
        """

        generator: Generator = self.model.embed(query)
        
        return next(generator).tolist()

embedding_service = EmbeddingService()