import json
import logging
import math
import re
import time
import uuid
from collections import OrderedDict
from functools import lru_cache
from typing import List, Optional

import numpy as np
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.core.config import settings
from app.services.embedder import (
    embed_query,
    embed_chunks,
    get_embedding_dimension,
)

logger = logging.getLogger("knowledge_copilot.vector_store")


# ── BM25 Index for hybrid search ────────────────────────────────────────────

class BM25Index:
    """In-memory BM25 Okapi index for lexical/semantic hybrid search."""

    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._docs: List[List[str]] = []
        self._doc_len: List[int] = []
        self._avgdl: float = 0.0
        self._idf: dict = {}
        self._vocab: set = set()
        self._point_ids: List[str] = []  # Qdrant point IDs parallel to _docs

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def fit(self, texts: List[str], point_ids: Optional[List[str]] = None):
        self._docs = [self._tokenize(t) for t in texts]
        self._doc_len = [len(d) for d in self._docs]
        self._point_ids = point_ids or []
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
        self._point_ids.clear()


def _mmr_selection(
    query_vec: np.ndarray,
    doc_vectors: np.ndarray,
    doc_indices: List[int],
    doc_scores: List[float],
    k: int,
    lambda_mult: float,
) -> List[int]:
    """Select diverse documents using Maximal Marginal Relevance."""
    n = len(doc_indices)
    if n == 0 or k == 0:
        return []

    norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    doc_vectors = doc_vectors / np.clip(norms, 1e-10, None)

    sim_to_query = doc_vectors @ query_vec

    selected_indices = []
    remaining = list(range(n))

    while len(selected_indices) < min(k, n) and remaining:
        best_idx = -1
        best_score = -1.0

        for i in remaining:
            mmr_score = lambda_mult * sim_to_query[i]
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


# ── Qdrant implementation ───────────────────────────────────────────────────

_TEXT_KEY = "_text"


class QdrantStore:
    """
    Vector store backed by Qdrant Cloud.
    Stores embeddings + payload (text + metadata) in a Qdrant collection.
    Singleton instance — shared across all requests.
    """

    def __init__(self):
        self._validate_config()

        self.collection_name = settings.qdrant_collection
        self.dim = 384

        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=120,
            prefer_grpc=False,
        )

        self._ensure_collection()

        self.bm25 = BM25Index()
        self._rebuild_bm25()

        self._cache: OrderedDict = OrderedDict()
        self._cache_maxsize = 128
        self._cache_ttl = settings.performance_cache_ttl

    @staticmethod
    def _validate_config():
        missing = []
        if not settings.qdrant_url:
            missing.append("QDRANT_URL")
        if not settings.qdrant_api_key:
            missing.append("QDRANT_API_KEY")
        if not settings.qdrant_collection:
            missing.append("QDRANT_COLLECTION")
        if missing:
            raise RuntimeError(
                f"Qdrant configuration incomplete. Missing: {', '.join(missing)}. "
                "Set these in your .env file or environment."
            )

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d)",
                self.collection_name, self.dim,
            )

        for field in ("conversation_id", "document_id"):
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception:
                logger.debug(
                    "Payload index for '%s' already exists on collection '%s'",
                    field, self.collection_name,
                )

    @staticmethod
    def _normalize(v: np.ndarray) -> List[float]:
        norm = np.linalg.norm(v)
        return (v / max(norm, 1e-10)).tolist()

    def _point_from_chunk(self, chunk_id: str, embedding: List[float], text: str, metadata: dict):
        payload = {_TEXT_KEY: text, **metadata}
        return models.PointStruct(
            id=chunk_id,
            vector=embedding,
            payload=payload,
        )

    def _chunk_from_point(self, point) -> dict:
        payload = dict(point.payload or {})
        text = payload.pop(_TEXT_KEY, "")
        return {
            "id":       str(point.id),
            "text":     text,
            "metadata": payload,
            "score":    point.score if point.score is not None else 0.0,
        }

    # ── Write ────────────────────────────────────────────────────────────────

    def _upsert_with_retry(
        self, points_batch: list, max_retries: int = 3, base_delay: float = 1.0
    ):
        for attempt in range(max_retries + 1):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points_batch,
                    wait=True,
                )
                return
            except Exception as e:
                err_str = str(e)
                is_dim_mismatch = "Vector dimension error" in err_str or "Wrong input" in err_str
                is_transient = any(
                    t in err_str
                    for t in ["timeout", "WriteTimeout", "ConnectionError", "timed out"]
                )

                if is_dim_mismatch:
                    logger.warning(
                        "dimension mismatch, deleting and recreating collection: %s", e
                    )
                    self.client.delete_collection(self.collection_name)
                    self._ensure_collection()
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=points_batch,
                        wait=True,
                    )
                    return

                if is_transient and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        "upsert timeout (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, max_retries, delay, e,
                    )
                    time.sleep(delay)
                    continue

                logger.error(
                    "upsert failed after %d attempts: %s",
                    attempt + 1, e,
                )
                raise

    def add_chunks(
        self,
        chunks: List[Document],
        document_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> int:
        """Embed and index a list of Document chunks into Qdrant. Returns count added.
        
        When document_id, conversation_id, or user_id are provided, they are injected
        into every chunk's metadata for downstream filtering during retrieval.
        """
        for chunk in chunks:
            if document_id:
                chunk.metadata["document_id"] = document_id
            if conversation_id:
                chunk.metadata["conversation_id"] = conversation_id
            if user_id:
                chunk.metadata["user_id"] = user_id
        embedded = embed_chunks(chunks)
        if not embedded:
            return 0

        points = []
        for e in embedded:
            vec = self._normalize(np.array(e["embedding"], dtype="float32"))
            chunk_id = str(uuid.uuid4())
            points.append(self._point_from_chunk(
                chunk_id, vec, e["text"], e["metadata"],
            ))

        self._ensure_collection()

        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self._upsert_with_retry(batch)

        self._rebuild_bm25()
        self._cache.clear()
        return len(points)

    # ── Cache ────────────────────────────────────────────────────────────────

    def _cache_key(self, query: str, k: int, fetch_k: Optional[int], mmr_lambda: Optional[float], filter_condition: Optional[models.Filter] = None) -> str:
        filter_str = ""
        if filter_condition and filter_condition.must:
            parts = []
            for cond in filter_condition.must:
                if hasattr(cond, 'key') and hasattr(cond, 'match'):
                    parts.append(f"{cond.key}={cond.match.value}")
            filter_str = ":".join(parts)
        return f"{query.strip().lower()}:k={k}:fk={fetch_k}:mmr={mmr_lambda}:f={filter_str}"

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

    # ── BM25 ─────────────────────────────────────────────────────────────────

    def _rebuild_bm25(self, texts: Optional[List[str]] = None, point_ids: Optional[List[str]] = None):
        if texts is not None:
            self.bm25.fit(texts, point_ids=point_ids)
            return

        try:
            all_texts = []
            all_point_ids = []
            next_offset = None
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if result is None:
                    break
                page, next_offset = result
                if not page:
                    break
                for point in page:
                    payload = dict(point.payload or {})
                    text = payload.get(_TEXT_KEY, "")
                    if text:
                        all_texts.append(text)
                        all_point_ids.append(str(point.id))
                if next_offset is None:
                    break
            self.bm25.fit(all_texts, point_ids=all_point_ids)
            logger.info(
                "BM25 index rebuilt with %d documents from Qdrant",
                len(all_texts),
            )
        except Exception:
            logger.warning("BM25 index could not be rebuilt from Qdrant. Using empty index.")

    def bm25_search(self, query: str, k: int = 10) -> List[dict]:
        raw = self.bm25.search(query, top_k=k)
        results = []
        for idx, score in raw:
            results.append({
                "index":    idx,
                "text":     "",
                "metadata": {},
                "score":    float(score),
                "_source":  "bm25",
            })
        return results

    # ── Search ───────────────────────────────────────────────────────────────

    def search(self, query: str, k: int = 5, filter_condition: Optional[models.Filter] = None) -> List[dict]:
        """Return top-k chunks most similar to query, with optional Qdrant filter."""
        return self._search_impl(query, k=k, fetch_k=None, mmr_lambda=None, filter_condition=filter_condition)

    def search_mmr(
        self,
        query:       str,
        k:           int   = 5,
        fetch_k:     int   = 30,
        mmr_lambda:  float = 0.3,
        filter_condition: Optional[models.Filter] = None,
    ) -> List[dict]:
        """Return top-k chunks using MMR diversity selection, with optional Qdrant filter."""
        return self._search_impl(query, k=k, fetch_k=fetch_k, mmr_lambda=mmr_lambda, filter_condition=filter_condition)

    def _search_impl(
        self,
        query:      str,
        k:          int,
        fetch_k:    Optional[int] = None,
        mmr_lambda: Optional[float] = None,
        filter_condition: Optional[models.Filter] = None,
    ) -> List[dict]:
        """Internal search implementation supporting both plain and MMR search."""
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total = int(collection_info.points_count or 0)
        except Exception:
            total = 0

        if total == 0:
            return []

        ckey = self._cache_key(query, k, fetch_k, mmr_lambda, filter_condition)
        cached = self._cache_get(ckey)
        if cached is not None:
            return cached

        q_vec = np.array([embed_query(query)], dtype="float32")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-10)

        use_mmr = fetch_k is not None and mmr_lambda is not None and fetch_k > k
        if use_mmr:
            search_k: int = fetch_k  # type: ignore[assignment]
        else:
            search_k: int = k
        effective_k = min(search_k, total)

        resp = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec[0].tolist(),
            limit=effective_k,
            with_payload=True,
            with_vectors=use_mmr,
            query_filter=filter_condition,
        )
        results = resp.points

        candidates = [self._chunk_from_point(r) for r in results]

        if use_mmr and len(candidates) > k:
            vecs_list = []
            for r in results:
                if r.vector is not None:
                    vecs_list.append(r.vector)
            if vecs_list:
                doc_vectors = np.array(vecs_list, dtype="float32")
                selected_idx_set = set(
                    _mmr_selection(
                        query_vec=q_vec[0],
                        doc_vectors=doc_vectors,
                        doc_indices=list(range(len(candidates))),
                        doc_scores=[c["score"] for c in candidates],
                        k=k,
                        lambda_mult=mmr_lambda if mmr_lambda is not None else 0.5,
                    )
                )
                results_mmr = [c for i, c in enumerate(candidates) if i in selected_idx_set]
                results_mmr.sort(key=lambda x: x["score"], reverse=True)
                candidates = results_mmr
            else:
                candidates = candidates[:k]

        self._cache_set(ckey, candidates)
        return candidates[:k]

    # ── Hybrid search (BM25 + vector) ────────────────────────────────────────

    def search_hybrid(
        self,
        query:      str,
        k:          int   = 5,
        fetch_k:    int   = 30,
        mmr_lambda: float = 0.5,
        alpha:      float = 0.3,
        filter_condition: Optional[models.Filter] = None,
    ) -> List[dict]:
        """Hybrid search combining BM25 lexical matching with vector similarity.

        alpha controls the blend:
          alpha = 1.0  → pure vector search
          alpha = 0.0  → pure BM25
          alpha = 0.3  → BM25-heavy hybrid

        When filter_condition is provided, it is applied to the Qdrant query_points
        call so only chunks matching the filter are considered.
        """
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total = int(collection_info.points_count or 0)
        except Exception:
            total = 0

        if total == 0:
            return []

        ckey = self._cache_key(query, k, fetch_k, mmr_lambda, filter_condition)
        cached = self._cache_get(ckey)
        if cached is not None:
            return cached

        q_vec = np.array([embed_query(query)], dtype="float32")
        q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-10)

        # Step 1 — Vector candidates (with optional filter)
        effective_k = min(fetch_k, total)
        vec_resp = self.client.query_points(
            collection_name=self.collection_name,
            query=q_vec[0].tolist(),
            limit=effective_k,
            with_payload=True,
            with_vectors=True,
            query_filter=filter_condition,
        )
        vec_results = vec_resp.points
        vec_candidates = []
        for r in vec_results:
            c = self._chunk_from_point(r)
            c["_source"] = "vector"
            vec_candidates.append(c)

        logger.info(
            "Hybrid search — vector candidates: %d (query_vector dim=%d, effective_k=%d)",
            len(vec_candidates), len(q_vec[0]), effective_k,
        )

        # Step 2 — BM25 candidates
        bm25_raw = self.bm25.search(query, top_k=fetch_k)
        logger.info(
            "Hybrid search — BM25 candidates: %d (BM25 index size=%d)",
            len(bm25_raw), len(self.bm25._docs),
        )

        # Step 3 — Merge and normalise scores
        # Key by Qdrant point ID so BM25 and vector results for the same point merge
        candidates_by_idx: dict = {}
        for i, r in enumerate(vec_candidates):
            r["_vec_idx"] = i
            candidates_by_idx[r["id"]] = r

        for idx, bscore in bm25_raw:
            point_id = self.bm25._point_ids[idx] if idx < len(self.bm25._point_ids) else str(idx)
            if point_id not in candidates_by_idx:
                # The point_id was in BM25 but not in vector results — use BM25 only
                candidates_by_idx[point_id] = {
                    "id":         point_id,
                    "text":       "",
                    "metadata":   {},
                    "score":      0.0,
                    "bm25_score": bscore,
                    "_source":    "bm25",
                    "_bm25_only": True,
                }
            else:
                # Point exists in both BM25 and vector — tag with BM25 score
                candidates_by_idx[point_id]["bm25_score"] = bscore
                candidates_by_idx[point_id]["_source"] = "hybrid"

        all_candidates = list(candidates_by_idx.values())
        logger.info("Hybrid search — merged candidates: %d (before filter)", len(all_candidates))

        if not all_candidates:
            return []

        # Filter out BM25-only entries with no text (not found by vector search)
        before_filter = len(all_candidates)
        all_candidates = [r for r in all_candidates if r.get("text") or not r.get("_bm25_only")]
        logger.info(
            "Hybrid search — after BM25-only filter: %d (removed %d empty)",
            len(all_candidates), before_filter - len(all_candidates),
        )

        if not all_candidates:
            return []

        vec_scores_list = [r.get("score", 0) for r in all_candidates]
        vmin, vmax = min(vec_scores_list), max(vec_scores_list)
        vrange = max(vmax - vmin, 1e-10)

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

        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        logger.info(
            "Hybrid search — after score normalization: %d, top score=%.4f",
            len(all_candidates), all_candidates[0]["score"] if all_candidates else 0,
        )

        # Step 4 — MMR diversity on top fetch_k candidates
        if mmr_lambda is not None and len(all_candidates) > k:
            pool = all_candidates[:fetch_k]
            pool_vecs_list = []
            pool_indices = []
            for pi, c in enumerate(pool):
                matched = False
                for r in vec_results:
                    if str(r.id) == c.get("id") and r.vector is not None:
                        pool_vecs_list.append(r.vector)
                        pool_indices.append(pi)
                        matched = True
                        break
                if not matched:
                    # BM25-only point not in vector results — skip from MMR
                    pool_vecs_list.append(np.zeros(self.dim, dtype="float32"))
                    pool_indices.append(pi)

            if pool_vecs_list and any(np.any(v) for v in pool_vecs_list):
                pool_vecs = np.array(pool_vecs_list, dtype="float32")
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
                logger.info(
                    "Hybrid search — MMR output: %d (pool=%d, k=%d, lambda=%.2f)",
                    len(results), len(pool), k, mmr_lambda,
                )
                self._cache_set(ckey, results)
                return results

        final = all_candidates[:k]
        logger.info(
            "Hybrid search — final top-%d: %d candidates returned",
            k, len(final),
        )
        self._cache_set(ckey, final)
        return final

    # ── Stats ────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        try:
            collection_info = self.client.get_collection(self.collection_name)
            total = collection_info.points_count
        except Exception:
            total = 0
        return {
            "provider":   "qdrant",
            "total_docs": total,
            "dimension":  self.dim,
            "collection": self.collection_name,
        }

    def list_sources(self) -> dict[str, int]:
        """Return mapping of file_name → chunk count for all indexed documents."""
        counts: dict[str, int] = {}
        next_offset = None
        try:
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if result is None:
                    break
                page, next_offset = result
                if not page:
                    break
                for point in page:
                    payload = dict(point.payload or {})
                    src = str(payload.get("file_name", payload.get("source", "__unknown__")))
                    counts[src] = counts.get(src, 0) + 1
                if next_offset is None:
                    break
        except Exception:
            pass
        return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    def get_chunks_by_source(self, source_files: List[str]) -> List[dict]:
        """Return all chunks belonging to the specified source files."""
        if not source_files:
            return []
        source_set = set(source_files)
        results = []
        next_offset = None
        try:
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if result is None:
                    break
                page, next_offset = result
                if not page:
                    break
                for point in page:
                    payload = dict(point.payload or {})
                    src = str(payload.get("file_name", payload.get("source", "__unknown__")))
                    if src in source_set:
                        text = payload.pop(_TEXT_KEY, "")
                        results.append({
                            "text":     text,
                            "metadata": payload,
                            "score":    1.0,
                        })
                if next_offset is None:
                    break
        except Exception:
            pass
        return results

    def get_chunks_by_section_id(self, section_id: str) -> List[dict]:
        """Return all chunks belonging to the specified section_id.

        Uses Qdrant payload filter to scroll chunks matching a section_id.
        Returns chunks ordered by section_chunk_index when available.
        """
        if not section_id:
            return []
        results = []
        next_offset = None
        try:
            while True:
                result = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=1000,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="section_id",
                                match=models.MatchValue(value=section_id),
                            ),
                        ],
                    ),
                )
                if result is None:
                    break
                page, next_offset = result
                if not page:
                    break
                for point in page:
                    payload = dict(point.payload or {})
                    text = payload.pop(_TEXT_KEY, "")
                    results.append({
                        "id":       str(point.id),
                        "text":     text,
                        "metadata": payload,
                        "score":    0.0,
                    })
                if next_offset is None:
                    break
        except Exception:
            logger.warning(f"Failed to scroll chunks for section_id={section_id}")
            return []

        # Sort by section_chunk_index for deterministic ordering
        results.sort(
            key=lambda x: x.get("metadata", {}).get("section_chunk_index", -1)
        )
        logger.info(
            "Retrieved %d chunks for section_id=%s",
            len(results), section_id,
        )
        return results

    def clear(self):
        """Delete and recreate the collection. All data is permanently removed."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self._ensure_collection()
        self.bm25.clear()
        self._cache.clear()


# ── Singleton instance ──────────────────────────────────────────────────────

_store_instance: Optional[QdrantStore] = None


def get_vector_store() -> QdrantStore:
    """Return the singleton QdrantStore instance. Initialized once at startup."""
    global _store_instance
    if _store_instance is None:
        _store_instance = QdrantStore()
    return _store_instance


def reset_vector_store():
    """Reset the singleton (useful for testing)."""
    global _store_instance
    if _store_instance is not None:
        _store_instance = None
