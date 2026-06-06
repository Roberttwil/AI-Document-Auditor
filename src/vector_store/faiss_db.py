import os
import pickle
from typing import List, Dict, Optional
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from src.config import EMBEDDING_DIMENSION, TOP_K_RESULTS, MIN_SIMILARITY_THRESHOLD
from src.vector_store.embeddings import LocalEmbeddings


class FAISSDocumentStore:
    """
    In-memory FAISS vector store for document chunks.
    
    Stores embeddings + metadata entirely in RAM.
    No persistence — data is lost when the server restarts.
    """
    
    def __init__(self, persist_dir="src/vector_store/faiss_index"):
        """Initialize FAISS index and try to load from disk if available."""
        self.persist_dir = os.path.abspath(persist_dir)
        self.index_file = os.path.join(self.persist_dir, "index.faiss")
        self.store_file = os.path.join(self.persist_dir, "store.pkl")
        
        # Default empty state
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.metadata_store: List[Dict] = []  # Parallel list to FAISS index
        self.text_store: List[str] = []       # Original chunk text
        self.content_types: List[str] = []
        self.is_initialized = True
        self.bm25 = None
        
        self.load_local()
        
    def _build_bm25(self):
        """Rebuild BM25 index from current text store."""
        if not self.text_store:
            self.bm25 = None
            return
            
        tokenized_corpus = [doc.lower().split() for doc in self.text_store]
        self.bm25 = BM25Okapi(tokenized_corpus)
    def load_local(self):
        """Load FAISS index and metadata from disk if they exist."""
        if os.path.exists(self.index_file) and os.path.exists(self.store_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.store_file, "rb") as f:
                    self.metadata_store, self.text_store, self.content_types = pickle.load(f)
                self._build_bm25()
                print(f"✅ Memuat {self.index.ntotal} dokumen dari {self.persist_dir}")
            except Exception as e:
                print(f"⚠️ Gagal memuat index FAISS: {e}")
                
    def save_local(self):
        """Save FAISS index and metadata to disk."""
        os.makedirs(self.persist_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_file)
        with open(self.store_file, "wb") as f:
            pickle.dump((self.metadata_store, self.text_store, self.content_types), f)
        print(f"💾 Index FAISS berhasil disimpan secara permanen di {self.persist_dir}")

    def add_documents(self, chunks: List[Dict]) -> int:
        """
        Embed and add document chunks to the FAISS index.
        
        Args:
            chunks: List of dicts from chunker.chunk_documents()
                   Each has: {chunk_id, text, metadata}
        
        Returns:
            Number of chunks added successfully.
        """
        if not chunks:
            return 0
        
        embedder = LocalEmbeddings()
        
        # Extract texts for embedding
        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]
        
        # Generate embeddings in batch
        embedding_vectors = embedder.embed_documents(texts)
        
        # Convert to numpy array (FAISS expects float32)
        embedding_array = np.array(embedding_vectors, dtype=np.float32)
        
        # Normalize for Inner Product (Cosine Similarity)
        faiss.normalize_L2(embedding_array)
        
        # Add to FAISS index
        self.index.add(embedding_array)
        
        # Store metadata and text in parallel
        self.metadata_store.extend(metadatas)
        self.text_store.extend(texts)
        self.content_types.extend([chunk.get("metadata", {}).get("content_type", "text") for chunk in chunks])
        
        self._build_bm25()
        
        # Save to disk after adding new documents
        self.save_local()
        
        return len(chunks)
    
    def similarity_search(self, query: str, k: int = None, exclude_content_type: str = None) -> List[Dict]:
        """
        Search for the most relevant chunks to a query.
        
        Only searches text chunks (excludes image chunks from vector search).
        
        Args:
            query: The search query string.
            k: Number of top results to return. Defaults to TOP_K_RESULTS from config.
            exclude_content_type: If set, exclude chunks with this content type (e.g., 'image').
        
        Returns:
            List of dicts with search results:
            [{chunk_text, score, metadata: {filename, page_num, chunk_index, content_type}}, ...]
        """
        if self.index.ntotal == 0:
            return []
        
        if k is None:
            k = min(TOP_K_RESULTS, self.index.ntotal)
        else:
            k = min(k, self.index.ntotal)
        
        embedder = LocalEmbeddings()
        
        # Embed the query
        query_vector = embedder.embed_query(query)
        query_array = np.array([query_vector], dtype=np.float32)
        
        # Normalize for Inner Product (Cosine Similarity)
        faiss.normalize_L2(query_array)
        
        # Search FAISS index
        distances, indices = self.index.search(query_array, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            
            score = float(distances[0][i])
            # Threshold filtering
            if score < MIN_SIMILARITY_THRESHOLD:
                continue
            
            # Skip if content type should be excluded
            if exclude_content_type and self.content_types[idx] == exclude_content_type:
                continue
            
            results.append({
                "chunk_text": self.text_store[idx],
                "score": score,
                "metadata": self.metadata_store[idx]
            })
        
        # Sort by score DESCENDING (higher Cosine Similarity = more relevant)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results[:k]  # Ensure we return exactly k results
    
    def hybrid_search(self, query: str, k: int = None, exclude_content_type: str = None) -> List[Dict]:
        """
        Hybrid search combining FAISS (Dense) and BM25 (Sparse) using Reciprocal Rank Fusion (RRF).
        """
        if self.index.ntotal == 0:
            return []
        
        k = k or min(TOP_K_RESULTS, self.index.ntotal)
        search_k = min(k * 2, self.index.ntotal)  # Retrieve more for better fusion
        
        # 1. FAISS Search
        faiss_results = self.similarity_search(query, k=search_k, exclude_content_type=exclude_content_type)
        
        # 2. BM25 Search
        bm25_results = []
        if self.bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            # Get top indices
            top_bm25_idx = np.argsort(bm25_scores)[::-1][:search_k]
            for idx in top_bm25_idx:
                if bm25_scores[idx] > 0:
                    if exclude_content_type and self.content_types[idx] == exclude_content_type:
                        continue
                    bm25_results.append({
                        "chunk_text": self.text_store[idx],
                        "score": float(bm25_scores[idx]),
                        "metadata": self.metadata_store[idx]
                    })
                    
        # If no BM25 results, return FAISS directly
        if not bm25_results:
            return faiss_results[:k]
            
        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        chunk_map = {}
        rrf_k = 60
        
        for rank, res in enumerate(faiss_results):
            meta = res["metadata"]
            cid = meta.get("chunk_id") or f"{meta.get('filename')}_{meta.get('page_num')}_{meta.get('chunk_index')}"
            chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0) + (1.0 / (rrf_k + rank + 1))
            
        for rank, res in enumerate(bm25_results):
            meta = res["metadata"]
            cid = meta.get("chunk_id") or f"{meta.get('filename')}_{meta.get('page_num')}_{meta.get('chunk_index')}"
            if cid not in chunk_map:
                chunk_map[cid] = res
            rrf_scores[cid] = rrf_scores.get(cid, 0) + (1.0 / (rrf_k + rank + 1))
            
        # Sort by RRF score descending
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build final results list
        final_results = []
        for cid in sorted_cids[:k]:
            res = chunk_map[cid]
            res["score"] = rrf_scores[cid]  # Override score with RRF score
            final_results.append(res)
            
        return final_results

    def multi_query_search(self, queries: List[str], k: int = None, exclude_content_type: str = None,
                           expand_parents: bool = True) -> List[Dict]:
        """
        Search using multiple query variations and merge results.
        
        P2 — Parent-Child Retrieval:
        - Searches ALL chunks (parents + children) for maximum precision.
        - If expand_parents=True, child chunks are expanded to their parent chunks
          before returning, ensuring the LLM gets full context.
        - Merges and deduplicates across all query variations.
        
        Args:
            queries: List of query strings (original + variations)
            k: Number of top results per query. Defaults to TOP_K_PER_QUERY.
            exclude_content_type: If set, exclude chunks with this content type.
            expand_parents: If True (default), expand child results to parent chunks.
        
        Returns:
            List of deduplicated search results, sorted by score.
            If expand_parents, all returned items will be parent-level chunks.
        """
        if self.index.ntotal == 0:
            return []
        
        if k is None:
            from src.config import TOP_K_PER_QUERY
            k = min(TOP_K_PER_QUERY, self.index.ntotal)
        else:
            k = min(k, self.index.ntotal)
        
        all_results = []
        seen_child_keys = set()  # Track children to expand to parents
        
        # Phase 1: Search each query variation using Hybrid Search
        child_parent_map = {}  # parent_id -> parent_chunk_data
        
        for query in queries:
            results = self.hybrid_search(query, k=k, exclude_content_type=exclude_content_type)
            for r in results:
                meta = r.get("metadata", {})
                # Check if this is a child chunk (has parent_id)
                parent_id = meta.get("parent_id")
                if parent_id and expand_parents:
                    # Track as child hit — will expand later
                    seen_child_keys.add(parent_id)
                    if parent_id not in child_parent_map:
                        child_parent_map[parent_id] = {
                            "parent_text": meta.get("parent_text", r["chunk_text"]),
                            "metadata_parent": {
                                "filename": meta["filename"],
                                "page_num": meta["page_num"],
                                "chunk_index": -1,  # Will be replaced
                            }
                        }
                else:
                    # This is a parent or table chunk — add directly
                    key = (meta.get("filename"), meta.get("page_num"), meta.get("chunk_index"))
                    # Only add non-child results (level != "child")
                    if meta.get("level") != "child" and key not in seen_child_keys:
                        seen_child_keys.add(parent_id if parent_id else key)
                        all_results.append(r)
        
        # Phase 2: Expand child hits to parent chunks
        if expand_parents and child_parent_map:
            # Also search directly for parent chunks by parent_id
            # Build results from expanded parents
            expanded_count = 0
            for parent_id, pdata in child_parent_map.items():
                # Check if this parent is already in all_results
                already_included = any(
                    r.get("metadata", {}).get("chunk_id") == parent_id
                    for r in all_results
                )
                # Also check if this parent's text is already in results
                text_already = any(
                    r["chunk_text"] == pdata["parent_text"]
                    for r in all_results
                )
                if not already_included and not text_already:
                    # Find the actual parent chunk in the store
                    for i, meta in enumerate(self.metadata_store):
                        if meta.get("chunk_id") == parent_id and meta.get("level") == "parent":
                            all_results.append({
                                "chunk_text": self.text_store[i],
                                "score": 1.0,  # Expanded parents get top score
                                "metadata": meta
                            })
                            expanded_count += 1
                            break
                    else:
                        # Fallback: use pre-stored parent_text
                        meta_parent = dict(pdata["metadata_parent"])
                        meta_parent["chunk_id"] = parent_id
                        all_results.append({
                            "chunk_text": pdata["parent_text"],
                            "score": 1.0,
                            "metadata": meta_parent
                        })
                        expanded_count += 1
        
        # Remove duplicates by chunk_id
        seen = set()
        deduped = []
        for r in all_results:
            meta = r.get("metadata", {})
            # Use chunk_id if available, else (filename, page, index)
            cid = meta.get("chunk_id") or f"{meta.get('filename')}_{meta.get('page_num')}_{meta.get('chunk_index')}"
            if cid not in seen:
                seen.add(cid)
                deduped.append(r)
        
        # Sort: expanded parents and real matches by score DESCENDING
        deduped.sort(key=lambda x: x["score"], reverse=True)
        
        return deduped
    
    def clear(self):
        """Reset the entire store (for new session / new uploads)."""
        self.index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        self.metadata_store = []
        self.text_store = []
        self.content_types = []
        self.bm25 = None
        
        import shutil
        if os.path.exists(self.persist_dir):
            try:
                shutil.rmtree(self.persist_dir)
                print(f"🗑️ Menghapus direktori {self.persist_dir}")
            except Exception as e:
                print(f"⚠️ Gagal menghapus direktori {self.persist_dir}: {e}")
    
    @property
    def total_documents(self) -> int:
        """Number of chunks currently in the store."""
        return self.index.ntotal