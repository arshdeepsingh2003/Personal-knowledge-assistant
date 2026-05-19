import json
import os
from pathlib import Path
from typing import List, Optional
from functools import lru_cache

import numpy as np
from langchain_core.documents import Document

from app.core.config import settings
from app.services.embedder import (
    embed_query,
    embed_chunks,
    get_embedding_model,
    get_embedding_dimension,
)


def _mmr_selection(
    query_vec: np.ndarray,
    doc_vectors: np.ndarray,
    doc_indices: List[int],
    doc_scores: List[float],
    k: int,
    lambda_mult: float,
) -> List[int]:
    """Select diverse documents using Maximal Marginal Relevance.

    MMR selects items that are both relevant to the query (sim(q,d_i))
    AND diverse from already-selected items (1 - max sim(d_i, d_j)).

    λ controls the tradeoff:
      λ = 1     → pure relevance (no diversity)
      λ = 0     → pure diversity (no relevance)
      λ = 0.3   → diversity-heavy (good for multi-section retrieval)
    """
    n = len(doc_indices)
    if n == 0 or k == 0:
        return []

    # Normalise doc vectors for cosine similarity
    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    doc_vectors = doc_vectors / np.clip(norms, 1e-10, None)

    # Similarity of each doc to the query
    sim_to_query = doc_vectors @ query_vec  

    selected_indices = []
    remaining = list(range(n))

    while len(selected_indices) < min(k, n) and remaining:
        best_idx = -1
        best_score = -1.0

        for i in remaining:
            # Relevance term
            mmr_score = lambda_mult * sim_to_query[i]

            # Diversity term: penalise similarity to already-selected docs
            if selected_indices:
                sim_to_selected = max(
                    float(doc_vectors[i] @ doc_vectors[j])
                    for j in selected_indices
                )
                mmr_score -= (1 - lambda_mult) * sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx != -1:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)

    return [doc_indices[i] for i in selected_indices]


def _store_dir() -> Path:
    p = Path(settings.vector_store_path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _faiss_path()     -> str: return str(_store_dir() / "faiss.index")
def _docstore_path()  -> str: return str(_store_dir() / "docstore.json")
def _chroma_dir()     -> str: return str(_store_dir() / "chroma")



# FAISS implementation

class FAISSStore:
    """
    Thin wrapper around a FAISS flat index + a JSON docstore.
    The FAISS index stores vectors only.
    The docstore maps integer IDs → {text, metadata}.
    """

    def __init__(self):
        import faiss
        self.faiss = faiss
        self.dim   = get_embedding_dimension()

        # IndexFlatIP = Inner Product (cosine similarity on normalised vectors)
        self.index     = faiss.IndexFlatIP(self.dim)
        self.docstore: dict[int, dict] = {}

        # Load from disk if a saved index exists
        if os.path.exists(_faiss_path()) and os.path.exists(_docstore_path()):
            self._load()

    #  Write

    def add_chunks(self, chunks: List[Document]) -> int:
        """Embed and index a list of Document chunks. Returns count added."""
        embedded = embed_chunks(chunks)
        if not embedded:
            return 0

        vectors = np.array(
            [e["embedding"] for e in embedded], dtype="float32"
        )
        # Normalise so IndexFlatIP behaves like cosine similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-10, None)

        start_id = self.index.ntotal
        self.index.add(vectors)

        for i, e in enumerate(embedded):
            self.docstore[start_id + i] = {
                "text":     e["text"],
                "metadata": e["metadata"],
            }

        self._save()
        return len(embedded)

    # Search 

    def search(self, query: str, k: int = 5) -> List[dict]:
        """Return top-k chunks most similar to query."""
        return self._search_impl(query, k=k, fetch_k=None, mmr_lambda=None)

    def search_mmr(
        self,
        query:       str,
        k:           int   = 5,
        fetch_k:     int   = 30,
        mmr_lambda:  float = 0.3,
    ) -> List[dict]:
        """Return top-k chunks using MMR diversity selection.

        MMR balances relevance (similarity to query) with diversity
        (dissimilarity among selected chunks) controlled by mmr_lambda.
        Lower λ = more diversity (chunks from different sections).
        """
        return self._search_impl(query, k=k, fetch_k=fetch_k, mmr_lambda=mmr_lambda)

    def _search_impl(
        self,
        query:      str,
        k:          int,
        fetch_k:    Optional[int] = None,
        mmr_lambda: Optional[float] = None,
    ) -> List[dict]:
        """Internal search implementation supporting both plain and MMR search."""
        if self.index.ntotal == 0:
            return []

        q_vec = np.array([embed_query(query)], dtype="float32")
        norm  = np.linalg.norm(q_vec)
        q_vec = q_vec / max(norm, 1e-10)

        use_mmr = mmr_lambda is not None and fetch_k is not None and fetch_k > k
        search_k = fetch_k if use_mmr else k

        scores, indices = self.index.search(q_vec, min(search_k, self.index.ntotal))

        # Gather candidates
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            doc = self.docstore.get(int(idx))
            if doc:
                candidates.append({
                    "index":    int(idx),
                    "text":     doc["text"],
                    "metadata": doc["metadata"],
                    "score":    float(score),
                })

        if use_mmr and len(candidates) > k:
            # Reconstruct vectors for MMR computation
            vecs = self.index.reconstruct_n(0, self.index.ntotal)
            selected_idx_set = set(
                _mmr_selection(
                    query_vec=q_vec[0],
                    doc_vectors=vecs[[c["index"] for c in candidates]],
                    doc_indices=list(range(len(candidates))),
                    doc_scores=[c["score"] for c in candidates],
                    k=k,
                    lambda_mult=mmr_lambda,
                )
            )
            results = [c for i, c in enumerate(candidates) if i in selected_idx_set]
            # Preserve order by MMR score (descending)
            results.sort(key=lambda x: x["score"], reverse=True)
        else:
            results = candidates[:k]

        return results

    # Stats 

    def stats(self) -> dict:
        return {
            "provider":   "faiss",
            "total_docs": self.index.ntotal,
            "dimension":  self.dim,
            "index_path": _faiss_path(),
        }

    def clear(self):
        """Wipe the index and docstore."""
        import faiss
        self.index    = faiss.IndexFlatIP(self.dim)
        self.docstore = {}
        self._save()

    #  Persistence 

    def _save(self):
        self.faiss.write_index(self.index, _faiss_path())
        with open(_docstore_path(), "w") as f:
            json.dump(self.docstore, f)

    def _load(self):
        expected_dim = get_embedding_dimension()
        self.index = self.faiss.read_index(_faiss_path())
        if self.index.d != expected_dim:
            print(
                f"WARNING: Saved index dimension ({self.index.d}) does not match "
                f"current model dimension ({expected_dim}). "
                f"Recreating index."
            )
            self.index = self.faiss.IndexFlatIP(expected_dim)
            self.dim = expected_dim
            self.docstore = {}
            return
        with open(_docstore_path()) as f:
            raw = json.load(f)
            self.docstore = {int(k): v for k, v in raw.items()}



# ChromaDB implementation

class ChromaStore:
    """
    Thin wrapper around ChromaDB's persistent client.
    ChromaDB handles its own storage — no separate docstore needed.
    """

    COLLECTION = "knowledge_copilot"

    def __init__(self):
        import chromadb
        self.client = chromadb.PersistentClient(path=_chroma_dir())
        self.col    = self.client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Document]) -> int:
        embedded = embed_chunks(chunks)
        if not embedded:
            return 0

        start = self.col.count()
        self.col.add(
            ids        = [str(start + i) for i in range(len(embedded))],
            embeddings = [e["embedding"] for e in embedded],
            documents  = [e["text"]      for e in embedded],
            metadatas  = [e["metadata"]  for e in embedded],
        )
        return len(embedded)

    def search(self, query: str, k: int = 5) -> List[dict]:
        if self.col.count() == 0:
            return []

        q_vec   = embed_query(query)
        results = self.col.query(
            query_embeddings=[q_vec],
            n_results=min(k, self.col.count()),
        )

        output = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text":     text,
                "metadata": meta,
                "score":    round(1 - dist, 4),   # convert distance → similarity
            })
        return output

    def stats(self) -> dict:
        return {
            "provider":   "chroma",
            "total_docs": self.col.count(),
            "dimension":  get_embedding_dimension(),
            "store_path": _chroma_dir(),
        }

    def clear(self):
        import chromadb
        self.client.delete_collection(self.COLLECTION)
        self.col = self.client.get_or_create_collection(
            name=self.COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )


# Public factory — the rest of the app uses only this

def get_vector_store():
    """Return the configured vector store. Cached after first call."""
    if settings.vector_store_provider == "chroma":
        return ChromaStore()
    return FAISSStore()