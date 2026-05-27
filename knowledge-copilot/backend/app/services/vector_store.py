import json
import math
import os
import re
import time
from collections import OrderedDict
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


# ── BM25 Index for hybrid search ────────────────────────────────────────────

class BM25Index:
    """In-memory BM25 Okapi index for lexical/semantic hybrid search.

    BM25 captures exact keyword matches (e.g., "GDPR", "ARR", "pricing",
    "subscriptions", "governance") that semantic search may overlook when the
    query phrasing differs from the document's embedding representation.
    """

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[List[str]] = []
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0
        self._idf: dict = {}
        self._vocab: set = set()

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def fit(self, texts: List[str]):
        self._docs = [self._tokenize(t) for t in texts]
        self._doc_len = [len(d) for d in self._docs]
        N = len(self._docs)
        self._avgdl = sum(self._doc_len) / max(N, 1)

        df = {}
        for doc_tokens in self._docs:
            for token in set(doc_tokens):
                df[token] = df.get(token, 0) + 1

        self._idf = {
            term: math.log((N - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in df.items()
        }
        self._vocab = set(self._idf.keys())

    def search(self, query: str, top_k: int = 10) -> List[tuple]:
        """Return list of (docstore_index, bm25_score) sorted by relevance."""
        if not self._docs:
            return []
        query_tokens = self._tokenize(query)

        scores = [0.0] * len(self._docs)
        for qt in query_tokens:
            if qt not in self._idf:
                continue
            idf = self._idf[qt]
            for i in range(len(self._docs)):
                tf = self._docs[i].count(qt)
                if tf == 0:
                    continue
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * self._doc_len[i] / self._avgdl
                )
                scores[i] += idf * numerator / denominator

        indexed = [(i, s) for i, s in enumerate(scores) if s > 0]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:top_k]

    def clear(self):
        self._docs.clear()
        self._doc_len.clear()
        self._avgdl = 0.0
        self._idf.clear()
        self._vocab.clear()


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

        # Choose index type: flat (exact) or IVF (approximate, faster for large)
        self._use_ivf = settings.performance_use_ivf_index
        if self._use_ivf:
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = settings.performance_ivf_nlist
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = min(nlist // 5, 20)
        else:
            self.index = faiss.IndexFlatIP(self.dim)

        self.docstore: dict[int, dict] = {}
        self.bm25 = BM25Index()

        # LRU cache for frequent queries (key: query_hash, value: results)
        self._cache: OrderedDict = OrderedDict()
        self._cache_maxsize = 128
        self._cache_ttl = settings.performance_cache_ttl

        # Load from disk if a saved index exists
        if os.path.exists(_faiss_path()) and os.path.exists(_docstore_path()):
            self._load()
            self._rebuild_bm25()

    #  Write

    def add_chunks(self, chunks: List[Document]) -> int:
        """Embed and index a list of Document chunks. Returns count added."""
        embedded = embed_chunks(chunks)
        if not embedded:
            return 0

        vectors = np.array(
            [e["embedding"] for e in embedded], dtype="float32"
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.clip(norms, 1e-10, None)

        # Train IVF index on first batch if needed
        if self._use_ivf and not self.index.is_trained:
            self.index.train(vectors)

        start_id = self.index.ntotal
        self.index.add(vectors)

        for i, e in enumerate(embedded):
            self.docstore[start_id + i] = {
                "text":     e["text"],
                "metadata": e["metadata"],
            }

        self._save()
        self._rebuild_bm25()

        # Clear cache on index update
        self._cache.clear()
        return len(embedded)

    # ── Cache ───────────────────────────────────────────────────────────────

    def _cache_key(self, query: str, k: int, fetch_k: Optional[int], mmr_lambda: Optional[float]) -> str:
        return f"{query.strip().lower()}:k={k}:fk={fetch_k}:mmr={mmr_lambda}"

    def _cache_get(self, key: str) -> Optional[List[dict]]:
        if key not in self._cache:
            return None
        entry_time, results = self._cache[key]
        if time.monotonic() - entry_time > self._cache_ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return results

    def _cache_set(self, key: str, results: List[dict]):
        self._cache[key] = (time.monotonic(), results)
        while len(self._cache) > self._cache_maxsize:
            self._cache.popitem(last=False)

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

        # Check cache first
        ckey = self._cache_key(query, k, fetch_k, mmr_lambda)
        cached = self._cache_get(ckey)
        if cached is not None:
            return cached

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
            results.sort(key=lambda x: x["score"], reverse=True)
        else:
            results = candidates[:k]

        self._cache_set(ckey, results)
        return results

    # Hybrid search (BM25 + vector)

    def _rebuild_bm25(self):
        texts = [doc["text"] for doc in self.docstore.values()]
        self.bm25.fit(texts)

    def bm25_search(self, query: str, k: int = 10) -> List[dict]:
        raw = self.bm25.search(query, top_k=k)
        results = []
        for idx, score in raw:
            doc = self.docstore.get(idx)
            if doc:
                results.append({
                    "index":    idx,
                    "text":     doc["text"],
                    "metadata": doc["metadata"],
                    "score":    float(score),
                    "_source":  "bm25",
                })
        return results

    def search_hybrid(
        self,
        query:      str,
        k:          int   = 5,
        fetch_k:    int   = 30,
        mmr_lambda: float = 0.5,
        alpha:      float = 0.3,
    ) -> List[dict]:
        """Hybrid search combining BM25 lexical matching with vector similarity.

        alpha controls the blend:
          alpha = 1.0  → pure vector search
          alpha = 0.0  → pure BM25
          alpha = 0.3  → BM25-heavy hybrid (default — boosts lexical recall)
        """
        if self.index.ntotal == 0:
            return []

        ckey = self._cache_key(query, k, fetch_k, mmr_lambda)
        cached = self._cache_get(ckey)
        if cached is not None:
            return cached

        q_vec = np.array([embed_query(query)], dtype="float32")
        norm  = np.linalg.norm(q_vec)
        q_vec = q_vec / max(norm, 1e-10)

        # Step 1 — Vector candidates
        vec_scores, vec_indices = self.index.search(q_vec, min(fetch_k, self.index.ntotal))
        vec_candidates = []
        for score, idx in zip(vec_scores[0], vec_indices[0]):
            if idx == -1:
                continue
            doc = self.docstore.get(int(idx))
            if doc:
                vec_candidates.append({
                    "index":    int(idx),
                    "text":     doc["text"],
                    "metadata": doc["metadata"],
                    "score":    float(score),
                    "_source":  "vector",
                })

        # Step 2 — BM25 candidates
        bm25_raw = self.bm25.search(query, top_k=fetch_k)

        # Step 3 — Merge & normalise scores
        candidates_by_idx: dict = {}
        for r in vec_candidates:
            candidates_by_idx[r["index"]] = r

        for idx, bscore in bm25_raw:
            if idx in candidates_by_idx:
                candidates_by_idx[idx]["bm25_score"] = bscore
            else:
                doc = self.docstore.get(idx)
                if doc:
                    candidates_by_idx[idx] = {
                        "index":       idx,
                        "text":        doc["text"],
                        "metadata":    doc["metadata"],
                        "score":       0.0,
                        "bm25_score":  bscore,
                        "_source":     "bm25",
                    }

        all_candidates = list(candidates_by_idx.values())

        if not all_candidates:
            return []

        # Normalise vector scores to [0, 1]
        vec_scores_list = [r.get("score", 0) for r in all_candidates]
        vmin, vmax = min(vec_scores_list), max(vec_scores_list)
        vrange = max(vmax - vmin, 1e-10)

        # Normalise BM25 scores to [0, 1]
        bm25_scores_list = [r.get("bm25_score", 0) for r in all_candidates]
        bmin, bmax = min(bm25_scores_list), max(bm25_scores_list)
        brange = max(bmax - bmin, 1e-10)

        for r in all_candidates:
            original_vec_score = r.get("score", 0)
            original_bm25_score = r.get("bm25_score", 0)
            vnorm = (original_vec_score - vmin) / vrange
            bnorm = (original_bm25_score - bmin) / brange
            r["score"] = alpha * vnorm + (1 - alpha) * bnorm
            r["vector_score"] = original_vec_score
            r["bm25_score"] = original_bm25_score

        # Sort by combined score
        all_candidates.sort(key=lambda x: x["score"], reverse=True)

        # Step 4 — MMR diversity on top fetch_k candidates
        if mmr_lambda is not None and len(all_candidates) > k:
            doc_vectors = self.index.reconstruct_n(0, self.index.ntotal)
            pool = all_candidates[:fetch_k]
            pool_vecs = np.array([doc_vectors[c["index"]] for c in pool])
            pool_scores = [c["score"] for c in pool]

            selected = _mmr_selection(
                query_vec=q_vec[0],
                doc_vectors=pool_vecs,
                doc_indices=list(range(len(pool))),
                doc_scores=pool_scores,
                k=k,
                lambda_mult=mmr_lambda,
            )
            results = [pool[i] for i in selected]
            results.sort(key=lambda x: x["score"], reverse=True)
            self._cache_set(ckey, results)
            return results

        final = all_candidates[:k]
        self._cache_set(ckey, final)
        return final

    # Stats 

    def stats(self) -> dict:
        return {
            "provider":   "faiss",
            "total_docs": self.index.ntotal,
            "dimension":  self.dim,
            "index_path": _faiss_path(),
        }

    def list_sources(self) -> dict[str, int]:
        """Return mapping of file_name → chunk count for all indexed documents."""
        counts: dict[str, int] = {}
        for doc in self.docstore.values():
            meta = doc.get("metadata", {})
            src = meta.get("file_name", meta.get("source", "__unknown__"))
            counts[src] = counts.get(src, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def get_chunks_by_source(self, source_files: List[str]) -> List[dict]:
        """Return all chunks belonging to the specified source files."""
        source_set = set(source_files)
        results = []
        for doc in self.docstore.values():
            meta = doc.get("metadata", {})
            src = meta.get("file_name", meta.get("source", "__unknown__"))
            if src in source_set:
                results.append({
                    "text":     doc["text"],
                    "metadata": meta,
                    "score":    1.0,
                })
        return results

    def clear(self):
        """Wipe the index, docstore, BM25 index, and cache."""
        import faiss
        if self._use_ivf:
            quantizer = faiss.IndexFlatIP(self.dim)
            nlist = settings.performance_ivf_nlist
            self.index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.nprobe = min(nlist // 5, 20)
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.docstore = {}
        self.bm25.clear()
        self._cache.clear()
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

    def list_sources(self) -> dict[str, int]:
        if self.col.count() == 0:
            return {}
        results = self.col.get()
        counts: dict[str, int] = {}
        for meta in results.get("metadatas", []):
            src = meta.get("file_name", meta.get("source", "__unknown__"))
            counts[src] = counts.get(src, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def get_chunks_by_source(self, source_files: List[str]) -> List[dict]:
        """Return all chunks belonging to the specified source files."""
        if self.col.count() == 0:
            return []
        source_set = set(source_files)
        results = self.col.get()
        output = []
        for text, meta in zip(results.get("documents", []), results.get("metadatas", [])):
            src = meta.get("file_name", meta.get("source", "__unknown__"))
            if src in source_set:
                output.append({
                    "text":     text,
                    "metadata": meta,
                    "score":    1.0,
                })
        return output

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