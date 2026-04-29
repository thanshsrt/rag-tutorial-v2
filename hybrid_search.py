from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional
import numpy as np

class HybridRetriever:
    def __init__(self, vector_db, documents: List[str], metadatas: Optional[List[Dict]] = None, k: int = 5):
        self.vector_db = vector_db
        self.k = k
        self.documents = documents
        
        # Handle None and empty list safely
        if metadatas is None:
            self.metadatas = [{} for _ in documents]
        else:
            self.metadatas = [m or {} for m in metadatas]
        
        # Index-based lookup (handles duplicates safely)
        self.doc_to_idx = {i: i for i in range(len(documents))}
        # Content-to-index map for vector result matching
        self.content_to_idx = {}
        for i, content in enumerate(documents):
            if content not in self.content_to_idx:
                self.content_to_idx[content] = i
        
        # Pre-tokenize for BM25
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def search(self, query: str, alpha: float = 0.5, candidate_count: int = 100) -> List[Dict]:
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        vector_k = min(candidate_count, len(self.documents))
        vector_results = self.vector_db.similarity_search_with_score(query, k=vector_k)
        
        vector_scores = np.zeros(len(self.documents))
        for doc, score in vector_results:
            indices = self.content_to_idx.get(doc.page_content, [])
            # Handle both single index and list of indices
            if isinstance(indices, list):
                for idx in indices:
                    vector_scores[idx] = score
            else:
                vector_scores[indices] = score
            
        bm25_norm = self._normalize(bm25_scores)
        vector_norm = self._normalize(vector_scores)
        
        hybrid_scores = (alpha * vector_norm) + ((1 - alpha) * bm25_norm)
        top_indices = np.argsort(hybrid_scores)[-self.k:][::-1]
        
        return [
            {
                "content": self.documents[idx],
                "hybrid_score": float(hybrid_scores[idx]),
                "bm25_score": float(bm25_norm[idx]),
                "vector_score": float(vector_norm[idx]),
                "source": self.metadatas[idx].get("id", f"doc_{idx}")
            }
            for idx in top_indices
        ]
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        s_min, s_max = scores.min(), scores.max()
        if s_max == s_min:
            return np.zeros_like(scores)
        return (scores - s_min) / (s_max - s_min)