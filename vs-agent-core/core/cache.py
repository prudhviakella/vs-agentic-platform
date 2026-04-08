"""
cache.py — SemanticCache
=========================
Simple semantic cache backed by Pinecone.

How it works:
  STORE  — embed the user question, upsert vector to Pinecone with the
           LLM answer stored in metadata. Set expires_at for TTL.
  LOOKUP — embed the incoming question, query Pinecone for the nearest
           vector. If similarity >= threshold and entry is not expired,
           return the cached answer. Agent call is skipped entirely.

One instance per domain (pharma / general) — different namespaces and
thresholds prevent cross-domain cache pollution.
"""

import hashlib
import logging
import time
from typing import Any, Optional

from langchain_openai import OpenAIEmbeddings

log = logging.getLogger(__name__)


class SemanticCache:
    """
    Args:
        index:                Connected Pinecone Index.
        embedder:             OpenAIEmbeddings instance.
        similarity_threshold: Cosine similarity floor for a cache HIT.
                              pharma=0.97 (strict), general=0.88 (relaxed).
        namespace:            Pinecone namespace — one per domain.
    """

    def __init__(
        self,
        index:                Any,
        embedder:             OpenAIEmbeddings,
        similarity_threshold: float = 0.88,
        namespace:            str   = "cache_general",
    ):
        self._index     = index
        self._embedder  = embedder
        self.threshold  = similarity_threshold
        self._namespace = namespace
        log.info(f"[CACHE] ready  namespace={namespace}  threshold={similarity_threshold}")

    # ── Public API ─────────────────────────────────────────────────────────────

    def lookup(self, question: str,user_id:str) -> Optional[str]:
        """
        Check if a similar question is cached.
        Returns the cached answer string on HIT, None on MISS.
        Expired entries are excluded via the expires_at metadata filter.
        """
        try:
            vector  = self._embedder.embed_query(question)
            resp    = self._index.query(
                vector=vector,
                top_k=1,
                namespace=self._namespace,
                include_metadata=True,
                filter={"user_id": {"$eq": user_id},"expires_at": {"$gt": time.time()}},   # skip expired
            )
            matches = resp.get("matches", [])
            if not matches:
                log.info(f"[CACHE] MISS  namespace={self._namespace}")
                return None

            score  = float(matches[0].get("score", 0.0))
            answer = matches[0].get("metadata", {}).get("answer", "")

            if score >= self.threshold and answer:
                log.info(f"[CACHE] HIT  score={score:.3f}  namespace={self._namespace}")
                return answer

            log.info(f"[CACHE] MISS  score={score:.3f} < threshold={self.threshold}")
            return None

        except Exception as exc:
            log.warning(f"[CACHE] lookup failed ({exc}) — treating as MISS")
            return None

    def store(self, question: str, answer: str, user_id:str,ttl: int = 3_600) -> None:
        """
        Cache the question → answer pair with a TTL.
        Vector ID is MD5(namespace + question) so re-asking the same question
        overwrites the old entry instead of creating a duplicate.
        Called from a background thread — response already returned to the user.
        """
        try:
            vector    = self._embedder.embed_query(question)
            vector_id = hashlib.md5(
                f"{self._namespace}{question}".encode()
            ).hexdigest()[:16]

            self._index.upsert(
                vectors=[{
                    "id":     vector_id,
                    "values": vector,
                    "metadata": {
                        "user_id": user_id,
                        "answer":     answer[:8000],       # Pinecone metadata limit
                        "expires_at": time.time() + ttl,    # for TTL filter in lookup
                        "created_at": time.time(),
                    },
                }],
                namespace=self._namespace,
            )
            log.info(f"[CACHE] stored  id={vector_id}  ttl={ttl}s  namespace={self._namespace}")

        except Exception as exc:
            log.warning(f"[CACHE] store failed ({exc})")

    def delete_expired(self) -> int:
        """
        Delete all expired vectors from the Pinecone namespace.

        Pinecone filters only hide expired entries on lookup — they stay in the
        index and consume storage until explicitly deleted. Call this periodically
        (e.g. a scheduled Lambda or a startup hook) to keep the index clean.

        Returns the number of vectors deleted.
        """
        try:
            self._index.delete(
                filter={"expires_at": {"$lt": time.time()}},
                namespace=self._namespace,
            )
            log.info(f"[CACHE] delete_expired complete  namespace={self._namespace}")
            return 0   # Pinecone delete-by-filter returns no count — log only
        except Exception as exc:
            log.warning(f"[CACHE] delete_expired failed ({exc})")
            return 0

    @property
    def namespace(self) -> str:
        return self._namespace