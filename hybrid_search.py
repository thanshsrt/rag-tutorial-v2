from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import numpy as np

class HybridRetriever:
    def __init__(self, vector_db, documents: List[str], metadatas: List[str], k: int = 5):
        self.vector_db = vector_db
        self.k = k
        self.documents = documents
        
        self.metadatas = metadatas or [{} for _ in documents]
        self.metadatas = [m or {} for m in self.metadatas]
        
        # O(1) lookup mapping
        self.doc_to_idx = {content: i for i, content in enumerate(documents)}
        
        # Pre-tokenize once (BM25 is fast, do it all)
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def search(
        self, 
        query: str, 
        alpha: float = 0.5, 
        candidate_count: int = 100
    ) -> List[Dict]:
        """
        Hybrid BM25 + Vector search with candidate capping.
        
        Args:
            query: Search query
            alpha: Weight for vector (0=BM25 only, 1=vector only)
            candidate_count: Max docs to fetch from vector DB (default 100)
        """
        # 1. BM25 scores for ALL docs (fast, CPU-based)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. Vector scores for TOP CANDIDATES ONLY (expensive, limit it)
        # Safety: don't ask for more than we have
        vector_k = min(candidate_count, len(self.documents))
        vector_results = self.vector_db.similarity_search_with_score(
            query, k=vector_k
        )
        
        # Build sparse vector scores array
        vector_scores = np.zeros(len(self.documents))
        for doc, score in vector_results:
            idx = self.doc_to_idx.get(doc.page_content)
            if idx is not None:
                vector_scores[idx] = score
            
        # 3. Normalize both to 0-1
        bm25_norm = self._normalize(bm25_scores)
        vector_norm = self._normalize(vector_scores)
        
        # 4. Fuse: hybrid = (alpha * vector) + ((1-alpha) * bm25)
        hybrid_scores = (alpha * vector_norm) + ((1 - alpha) * bm25_norm)
        
        # 5. Extract top-k
        top_indices = np.argsort(hybrid_scores)[-self.k:][::-1]
        
        return [
            {
                "content": self.documents[idx],
                "hybrid_score": float(hybrid_scores[idx]),
                "bm25_score": float(bm25_norm[idx]),
                "vector_score": float(vector_norm[idx]),
                "source": f"doc_{idx}"
            }
            for idx in top_indices
        ]
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-max normalize to [0, 1]."""
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)