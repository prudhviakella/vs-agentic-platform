"""
cache.py — Semantic Cache  (Pinecone vector similarity cache)
=============================================================
WHY class: Pinecone index client is expensive to re-initialise per lookup.
           Connection state and index reference live here.
WHY NOT shared across agents: cache scope is per-agent — sharing would allow
  a pharma agent's cached clinical answers to be returned for a general agent's
  queries, bypassing domain-specific threshold enforcement entirely.

Cache REPLACES work — HIT means NO episodic search, NO tools, NO LLM call.
Write is always fire-and-forget after response (via daemon thread in
SemanticCacheMiddleware._store_sync).

WHY Pinecone (replacing FAISS):
  FAISS is in-process memory — lost on every restart. For a production
  clinical trial agent where LLM calls are expensive, losing the cache on
  deployment or crash means paying full token cost for all previously cached
  queries. Pinecone persists across restarts, scales to any number of cached
  entries, and handles ANN search in a managed service with no index rebuild
  cost on entry deletion (unlike FAISS IndexFlat which had no delete operation).

  FAISS comparison:
    FAISS    — in-process, sub-ms exact search, lost on restart, manual TTL rebuild
    Pinecone — network call (~50ms), ANN search, persistent, native metadata filter

  The ~50ms latency per cache lookup is acceptable because a HIT saves a full
  LLM call (~2-3s + ~$0.01). The latency cost is paid only on MISS in the
  before_agent hook; on HIT the agent returns immediately with the cached answer.

TTL enforcement:
  Pinecone has no native TTL on vectors. TTL is enforced by storing an
  "expires_at" Unix timestamp in metadata and filtering it in the query:
    filter={"expires_at": {"$gt": time.time()}}
  Expired vectors are not returned by lookup() but remain in the index until
  explicitly deleted. A background cleanup job (not included here) should
  periodically call index.delete(filter={"expires_at": {"$lt": time.time()}})
  to reclaim index space.

Namespace strategy:
  Each agent domain gets its own Pinecone namespace:
    "pharma"  → namespace="cache_pharma"
    "general" → namespace="cache_general"
  This prevents a general-domain cached answer from being returned to a
  pharma-domain query, which would bypass the stricter 0.97 threshold.

Dependencies:
  pip install pinecone-client langchain-openai
"""

import hashlib
import logging
import random
import time
from typing import Any, Optional

from langchain_openai import OpenAIEmbeddings

log = logging.getLogger(__name__)


class SemanticCache:
    """
    Pinecone-backed semantic similarity cache with per-domain threshold and TTL.

    Domain-aware threshold (set at construction by SemanticCacheMiddleware):
      pharma  (high-risk) → 0.97  near-identical questions required for a HIT.
                                   A cached answer about drug dosage must not be
                                   returned for a superficially similar but
                                   clinically distinct question.
      general             → 0.88  semantically similar questions share an answer.
                                   Cost savings are prioritised over strict exactness.

    Args:
        index:                Pinecone Index object (pre-initialised).
        embedder:             OpenAIEmbeddings instance.
        similarity_threshold: Cosine similarity floor for a cache HIT.
        namespace:            Pinecone namespace string — one per domain to prevent
                              cross-domain cache pollution.
    """

    def __init__(
        self,
        index: Any,
        embedder: OpenAIEmbeddings,
        similarity_threshold: float = 0.88,
        namespace: str = "cache_general",
    ):
        self._index     = index
        self._embedder  = embedder
        self.threshold  = similarity_threshold
        self._namespace = namespace
        log.info(
            f"[CACHE] Pinecone cache ready"
            f"  namespace={namespace}  threshold={similarity_threshold}"
        )

    def _embed(self, text: str) -> list[float]:
        """
        Embed *text* using embed_query() (synchronous).

        Fallback on OpenAI failure: returns a deterministic pseudo-random vector
        seeded by MD5(text). This degrades to a guaranteed cache MISS rather
        than blocking the request. The same text always gets the same fallback
        vector, preventing a transient failure from producing a false HIT on a
        subsequent call with a different question.

        WHY 1536 zeros are not used as a fallback:
          A zero vector would produce undefined cosine similarity scores against
          all stored vectors and could return a false HIT. A random-but-deterministic
          vector is guaranteed to produce a MISS against any real embedding.
        """
        try:
            return self._embedder.embed_query(text)
        except Exception:
            # Deterministic fallback — same text always maps to same fake vector.
            seed   = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            rng    = random.Random(seed)
            vector = [rng.gauss(0, 1) for _ in range(1_536)]
            norm   = sum(x * x for x in vector) ** 0.5
            return [x / (norm + 1e-9) for x in vector]

    def lookup(self, question: str) -> Optional[str]:
        """
        Embed *question* and query Pinecone for a cached answer.

        Filters out expired entries via the "expires_at" metadata field before
        scoring. Only returns an answer if the best match's cosine similarity
        meets the domain threshold.

        Returns the cached answer string on HIT, None on MISS.

        Failure handling:
          Any Pinecone error is caught and logged at WARNING. The method returns
          None (MISS) so the agent proceeds normally. Cache availability must not
          be a single point of failure.
        """
        try:
            vector = self._embed(question)
            resp   = self._index.query(
                vector=vector,
                top_k=1,
                namespace=self._namespace,
                include_metadata=True,
                # TTL enforcement — expired entries are excluded from results.
                filter={"expires_at": {"$gt": time.time()}},
            )
            matches = resp.get("matches", [])
            if not matches:
                log.info(f"[CACHE] MISS  no matches  namespace={self._namespace}")
                return None

            best       = matches[0]
            best_score = float(best.get("score", 0.0))
            best_ans   = best.get("metadata", {}).get("answer", "")

            if best_score >= self.threshold and best_ans:
                log.info(
                    f"[CACHE] HIT  score={best_score:.3f}"
                    f"  threshold={self.threshold}  namespace={self._namespace}"
                )
                return best_ans

            log.info(
                f"[CACHE] MISS  score={best_score:.3f}"
                f"  threshold={self.threshold}  namespace={self._namespace}"
            )
            return None

        except Exception as exc:
            log.warning(f"[CACHE] Lookup failed ({exc}) — treating as MISS")
            return None

    def store(self, question: str, answer: str, ttl: int = 3_600) -> None:
        """
        Embed *question* and upsert the (vector, answer) pair into Pinecone.

        Called exclusively from SemanticCacheMiddleware._store_sync() which runs
        in a daemon thread — the response has already been returned to the user.

        Vector ID:
          MD5(namespace + question)[:16] — deterministic so re-asking the same
          question in the same domain overwrites the previous answer rather than
          creating a duplicate entry. The namespace prefix prevents ID collisions
          between pharma and general caches.

        Metadata fields stored:
          answer     — the cached LLM response string (truncated to 8KB for
                       Pinecone's 40KB metadata limit with room for other fields)
          expires_at — Unix timestamp for TTL enforcement in lookup() filter
          created_at — Unix timestamp for operational visibility / debugging

        Failure handling:
          Upsert failures are logged at WARNING and swallowed. A failed write
          means the next identical question will be a MISS — a cost impact, not
          a correctness or safety impact.
        """
        try:
            vector    = self._embed(question)
            vector_id = hashlib.md5(
                f"{self._namespace}{question}".encode()
            ).hexdigest()[:16]

            self._index.upsert(
                vectors=[{
                    "id":     vector_id,
                    "values": vector,
                    "metadata": {
                        "answer":     answer[:8_000],  # Pinecone metadata value limit
                        "expires_at": time.time() + ttl,
                        "created_at": time.time(),
                    },
                }],
                namespace=self._namespace,
            )
            log.info(
                f"[CACHE] Stored  id={vector_id}"
                f"  namespace={self._namespace}  ttl={ttl}s"
            )
        except Exception as exc:
            log.warning(f"[CACHE] Store failed ({exc})")

    @property
    def namespace(self) -> str:
        """The Pinecone namespace this cache instance writes to and reads from."""
        return self._namespace
