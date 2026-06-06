"""
Unit tests for the FAISS vector store.
Tests mock LocalEmbeddings to avoid loading the full model.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from unittest.mock import patch
from src.vector_store.faiss_db import FAISSDocumentStore
from src.config import EMBEDDING_DIMENSION


class MockEmbeddings:
    """Mock LocalEmbeddings that returns random vectors for testing."""
    
    def embed_documents(self, texts):
        """Return random embedding vectors."""
        return [np.random.randn(EMBEDDING_DIMENSION).tolist() for _ in texts]
    
    def embed_query(self, text):
        """Return a random query embedding vector."""
        return np.random.randn(EMBEDDING_DIMENSION).tolist()


@patch('src.vector_store.faiss_db.LocalEmbeddings')
def test_empty_store(mock_embeddings):
    """Searching an empty store should return empty list."""
    mock_embeddings.return_value = MockEmbeddings()
    store = FAISSDocumentStore()
    results = store.similarity_search("test query")
    
    assert results == []
    assert store.total_documents == 0


@patch('src.vector_store.faiss_db.LocalEmbeddings')
def test_add_single_document(mock_embeddings):
    """Adding a single document should work."""
    mock_embeddings.return_value = MockEmbeddings()
    store = FAISSDocumentStore()
    
    chunks = [{
        "chunk_id": "test_1_0",
        "text": "Laporan keuangan menunjukkan laba bersih Rp 100 miliar.",
        "metadata": {
            "filename": "laporan.pdf",
            "page_num": 1,
            "chunk_index": 0
        }
    }]
    
    count = store.add_documents(chunks)
    assert count == 1
    assert store.total_documents == 1


@patch('src.vector_store.faiss_db.LocalEmbeddings')
def test_add_multiple_documents(mock_embeddings):
    """Adding multiple documents should increase total count."""
    mock_embeddings.return_value = MockEmbeddings()
    store = FAISSDocumentStore()
    
    chunks = [
        {
            "chunk_id": "doc1_1_0",
            "text": "Pendapatan naik 20% menjadi Rp 500 miliar.",
            "metadata": {"filename": "doc1.pdf", "page_num": 1, "chunk_index": 0}
        },
        {
            "chunk_id": "doc2_2_0",
            "text": "Beban operasional turun 15%.",
            "metadata": {"filename": "doc2.pdf", "page_num": 2, "chunk_index": 0}
        },
        {
            "chunk_id": "doc1_3_0",
            "text": "Laba bersih mencapai Rp 200 miliar.",
            "metadata": {"filename": "doc1.pdf", "page_num": 3, "chunk_index": 0}
        }
    ]
    
    count = store.add_documents(chunks)
    assert count == 3
    assert store.total_documents == 3


@patch('src.vector_store.faiss_db.LocalEmbeddings')
def test_add_empty_chunks(mock_embeddings):
    """Adding empty chunks list should return 0."""
    mock_embeddings.return_value = MockEmbeddings()
    store = FAISSDocumentStore()
    count = store.add_documents([])
    
    assert count == 0
    assert store.total_documents == 0


@patch('src.vector_store.faiss_db.LocalEmbeddings')
def test_clear_store(mock_embeddings):
    """Clearing the store should reset everything."""
    mock_embeddings.return_value = MockEmbeddings()
    store = FAISSDocumentStore()
    
    chunks = [{
        "chunk_id": "test_1_0",
        "text": "Test content",
        "metadata": {"filename": "test.pdf", "page_num": 1, "chunk_index": 0}
    }]
    
    store.add_documents(chunks)
    assert store.total_documents == 1
    
    store.clear()
    assert store.total_documents == 0
    
    results = store.similarity_search("test")
    assert results == []


@patch('src.vector_store.faiss_db.LocalEmbeddings')
def test_similarity_search_returns_results(mock_embeddings):
    """Similarity search should return results with correct structure."""
    mock_embeddings.return_value = MockEmbeddings()
    store = FAISSDocumentStore()
    
    chunks = [
        {
            "chunk_id": "doc_1_0",
            "text": "Laba bersih perusahaan naik 30% tahun ini.",
            "metadata": {"filename": "report.pdf", "page_num": 5, "chunk_index": 0}
        },
        {
            "chunk_id": "doc_2_0",
            "text": "Total aset mencapai Rp 1 triliun.",
            "metadata": {"filename": "report.pdf", "page_num": 7, "chunk_index": 0}
        }
    ]
    
    store.add_documents(chunks)
    results = store.similarity_search("laba perusahaan", k=2)
    
    assert len(results) > 0
    assert "chunk_text" in results[0]
    assert "score" in results[0]
    assert "metadata" in results[0]
    assert "filename" in results[0]["metadata"]
    assert "page_num" in results[0]["metadata"]


if __name__ == "__main__":
    test_empty_store()
    test_add_single_document()
    test_add_multiple_documents()
    test_add_empty_chunks()
    test_clear_store()
    test_similarity_search_returns_results()
    print("✅ All vector store tests passed!")