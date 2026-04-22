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
        if self.index.ntotal == 0:
            return []

        q_vec = np.array([embed_query(query)], dtype="float32")
        norm  = np.linalg.norm(q_vec)
        q_vec = q_vec / max(norm, 1e-10)

        scores, indices = self.index.search(q_vec, min(k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:           # FAISS returns -1 for empty slots
                continue
            doc = self.docstore.get(int(idx))
            if doc:
                results.append({
                    "text":     doc["text"],
                    "metadata": doc["metadata"],
                    "score":    float(score),
                })
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