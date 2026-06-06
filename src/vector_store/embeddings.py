from typing import List
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL


_embeddings_instance = None


def get_embeddings() -> SentenceTransformer:
    """
    Get or create a singleton local embedding model.
    
    Uses sentence-transformers with multilingual model:
    - paraphrase-multilingual-MiniLM-L12-v2 (dimensi 384)
    - Support Bahasa Indonesia + 50+ bahasa lain
    - 100% offline setelah download pertama
    - Zero API cost, zero latency
    
    Returns:
        SentenceTransformer instance loaded locally.
    """
    global _embeddings_instance
    
    if _embeddings_instance is None:
        print(f"⏳ Loading embedding model: {EMBEDDING_MODEL}...")
        _embeddings_instance = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✅ Embedding model loaded (dimensi: {_embeddings_instance.get_sentence_embedding_dimension()})")
    
    return _embeddings_instance


class LocalEmbeddings:
    """
    Wrapper class to match the interface expected by the FAISS store.
    Provides embed_documents() and embed_query() methods.
    """
    
    def __init__(self):
        self.model = get_embeddings()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts into vectors.
        
        Args:
            texts: List of text strings to embed.
        
        Returns:
            List of embedding vectors (list of floats).
        """
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return embeddings.tolist()
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text into a vector.
        
        Args:
            text: Query text string.
        
        Returns:
            Embedding vector (list of floats).
        """
        embedding = self.model.encode(text, show_progress_bar=False)
        return embedding.tolist()