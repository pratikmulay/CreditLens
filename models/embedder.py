"""
models/embedder.py  –  Semantic profile embedder backed by ChromaDB.

Uses 'sentence-transformers/all-MiniLM-L6-v2' to embed queries and
find similar SME profiles already stored in the 'sme_profiles' ChromaDB
collection.

Returned results always include 'risk_label' in their metadata.
"""

from __future__ import annotations
import os
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
COLLECTION_NAME = "sme_profiles"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class ProfileEmbedder:
    """
    Semantic similarity search over the ChromaDB SME profile collection.

    Usage
    -----
    model = ProfileEmbedder()
    results = model.find_similar('small textile firm with moderate revenue', n=3)
    # → [
    #     {"id": "...", "document": "...", "distance": 0.12,
    #      "risk_label": "low", "loan_grade": "B", ...},
    #     ...
    #   ]
    """

    def __init__(self) -> None:
        self._embed_model: SentenceTransformer | None = None
        self._client: chromadb.PersistentClient | None = None
        self._collection = None

    # ── lazy loaders ─────────────────────────────────────────────────────────
    def _load_embed_model(self) -> SentenceTransformer:
        if self._embed_model is None:
            print(f"Loading embedding model '{EMBED_MODEL}' …")
            self._embed_model = SentenceTransformer(EMBED_MODEL)
        return self._embed_model

    def _load_collection(self):
        if self._collection is None:
            self._client = chromadb.PersistentClient(path=CHROMA_DIR)
            self._collection = self._client.get_collection(COLLECTION_NAME)
        return self._collection

    # ── public API ────────────────────────────────────────────────────────────
    def embed(self, text: str) -> list[float]:
        """Return the embedding vector for *text* as a plain Python list."""
        model = self._load_embed_model()
        vector = model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def find_similar(
        self,
        query: str,
        n: int = 5,
        where: dict | None = None,
    ) -> list[dict]:
        """
        Find the *n* most semantically similar SME profiles.

        Parameters
        ----------
        query   – natural-language description of the SME
        n       – number of results to return
        where   – optional ChromaDB metadata filter (e.g. {"risk_label": "high"})

        Returns
        -------
        List of dicts, each with all metadata fields plus:
          id        – ChromaDB document id
          document  – the stored document text
          distance  – cosine distance (lower = more similar)
        """
        collection = self._load_collection()
        query_vec  = self.embed(query)

        kwargs: dict = dict(
            query_embeddings=[query_vec],
            n_results=n,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw = collection.query(**kwargs)

        results: list[dict] = []
        for doc, meta, dist, doc_id in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
            raw["ids"][0],
        ):
            entry = dict(meta)          # copy all metadata (includes risk_label)
            entry["id"]       = doc_id
            entry["document"] = doc
            entry["distance"] = round(float(dist), 4)
            results.append(entry)

        return results

    def find_high_risk_similar(self, query: str, n: int = 5) -> list[dict]:
        """Shortcut: find similar profiles that are labelled high-risk."""
        return self.find_similar(query, n=n, where={"risk_label": "high"})

    def find_low_risk_similar(self, query: str, n: int = 5) -> list[dict]:
        """Shortcut: find similar profiles that are labelled low-risk."""
        return self.find_similar(query, n=n, where={"risk_label": "low"})

    def collection_stats(self) -> dict:
        """Return basic stats about the collection."""
        col = self._load_collection()
        return {"total_documents": col.count(), "collection_name": COLLECTION_NAME}
