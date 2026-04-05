"""
pinecone_store.py — PineconeStore (LangGraph BaseStore → Pinecone adapter)
===========================================================================
Implements LangGraph's BaseStore interface backed by a Pinecone index so
EpisodicMemoryMiddleware and @dynamic_prompt can use Pinecone as their
shared episodic memory store without changing any call sites.

WHY a BaseStore adapter (not calling Pinecone directly from middleware):
  EpisodicMemoryMiddleware calls self._store.put() / self._store.search().
  @dynamic_prompt calls request.runtime.store.search().
  Both call sites expect the LangGraph BaseStore contract. Wrapping Pinecone
  in a BaseStore adapter means zero changes to middleware or prompt code —
  the storage backend is swapped at build_agent() time only.

Namespace mapping:
  LangGraph namespace tuples → Pinecone namespace strings.
  ("episodic", "user_abc") → "episodic__user_abc"
  Double underscore chosen as separator because user IDs are unlikely to
  contain "__" and it is visually distinct from single underscore in logs.

Vector strategy for episodic entries:
  put() embeds the entry's "text" field synchronously so the vector is
  stored at write time. search() with a non-empty query embeds the query
  and performs cosine similarity search. search() with query="" or None
  falls back to a list+fetch pattern (returns the most recent entries by
  metadata timestamp rather than by relevance).

  WHY embed at write time (not at search time):
    Episodic entries are written once and searched many times. Pre-computing
    the vector at write time means each search pays only one embedding call
    (the query), not one per stored entry.

Item ID convention:
  Pinecone vector IDs are constructed as "{namespace_str}__{key}" to ensure
  global uniqueness across namespaces within the same index. Without a
  namespace prefix in the ID, a key collision between two users (both using
  the same MD5 entry_id) would cause silent data corruption.

Metadata schema per vector:
  {
    "namespace":   "episodic__user_abc",   # for filtering in search
    "key":         "3f7a92bc1d4e",         # original entry_id
    "text":        "Q: ...\nA: ...",        # the episodic Q&A text
    "ts":          1714000000.0,            # Unix timestamp for ordering
    "created_at":  "2024-04-25T...",        # ISO for Item construction
    "updated_at":  "2024-04-25T...",        # ISO for Item construction
    ... (any additional value fields)
  }

Dependencies:
  pip install pinecone-client langchain-openai
"""

import logging
import time
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain_openai import OpenAIEmbeddings
from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

log = logging.getLogger(__name__)

# Sentinel used when Pinecone returns no score (list+fetch path).
_NO_SCORE = None


def _ns_to_str(namespace: tuple[str, ...]) -> str:
    """Convert LangGraph namespace tuple to a Pinecone namespace string."""
    return "__".join(namespace)


def _vector_id(namespace_str: str, key: str) -> str:
    """Construct a globally unique Pinecone vector ID."""
    return f"{namespace_str}__{key}"


def _item_from_metadata(meta: dict, score: float | None = None) -> SearchItem:
    """
    Reconstruct a LangGraph SearchItem from Pinecone vector metadata.

    All fields stored at upsert time are present in metadata. The 'value'
    dict is rebuilt by excluding internal bookkeeping keys (namespace, key,
    created_at, updated_at) which belong to the Item envelope, not the value.
    """
    _ENVELOPE_KEYS = {"namespace", "key", "created_at", "updated_at"}
    value = {k: v for k, v in meta.items() if k not in _ENVELOPE_KEYS}
    namespace = tuple(meta["namespace"].split("__"))
    return SearchItem(
        namespace=namespace,
        key=meta["key"],
        value=value,
        created_at=datetime.fromisoformat(meta["created_at"]),
        updated_at=datetime.fromisoformat(meta["updated_at"]),
        score=score,
    )


class PineconeStore(BaseStore):
    """
    LangGraph BaseStore backed by a Pinecone serverless index.

    Used as the shared store for:
      - EpisodicMemoryMiddleware: put() writes Q&A pairs, search() retrieves them
      - @dynamic_prompt:         search() reads episodic context per user turn

    Both consumers receive the same instance (shared via build_agent), so
    writes from the middleware are immediately visible to the prompt layer
    on the next turn.

    Args:
        index:     Pinecone Index object (already initialised and connected).
        embedder:  OpenAIEmbeddings instance for encoding text fields.
                   Must produce vectors of the same dimension as the index.
        top_k:     Maximum results returned by vector search (default 3).
                   Matches the limit=3 used in EpisodicMemoryMiddleware and
                   context_aware_prompt.
    """

    def __init__(self, index: Any, embedder: OpenAIEmbeddings, top_k: int = 3):
        self._index   = index
        self._embedder = embedder
        self._top_k    = top_k

    def _embed(self, text: str) -> list[float]:
        """
        Embed *text* and return a plain float list for Pinecone upsert/query.

        Uses embed_query() (synchronous) because batch() is the sync interface.
        Raises on failure — callers in EpisodicMemoryMiddleware already wrap
        store calls in try/except and degrade gracefully on any exception.
        """
        return self._embedder.embed_query(text)

    # ------------------------------------------------------------------
    # BaseStore abstract interface
    # ------------------------------------------------------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """
        Execute a list of store operations synchronously.

        LangGraph's get(), put(), search(), delete() convenience methods all
        delegate to batch(). Implementing batch() is sufficient to satisfy the
        full BaseStore interface.

        Supported op types:
          GetOp    → fetch by vector ID
          PutOp    → upsert (value non-None) or delete (value is None)
          SearchOp → vector similarity search or recency-ordered list+fetch
          ListNamespacesOp → returns [] (not required for episodic use case)

        Unrecognised op types return None rather than raising so that future
        LangGraph op types added by upstream do not break the adapter.
        """
        results: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._handle_get(op))
            elif isinstance(op, PutOp):
                results.append(self._handle_put(op))
            elif isinstance(op, SearchOp):
                results.append(self._handle_search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append([])  # not needed for episodic use case
            else:
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """
        Async batch — delegates to synchronous batch().

        Pinecone's Python client uses synchronous HTTP calls. A true async
        implementation would require httpx or aiohttp. For the current
        deployment scope (sync middleware hooks) this delegation is correct.
        Replace with an async Pinecone client if the agent migrates to an
        async framework.
        """
        return self.batch(ops)

    # ------------------------------------------------------------------
    # Op handlers
    # ------------------------------------------------------------------

    def _handle_get(self, op: GetOp) -> Item | None:
        """
        Fetch a single item by namespace + key.

        Constructs the vector ID from namespace + key (same convention as
        _handle_put) and fetches it from Pinecone. Returns None if not found
        rather than raising, matching BaseStore.get() semantics.
        """
        ns_str    = _ns_to_str(op.namespace)
        vector_id = _vector_id(ns_str, op.key)
        try:
            resp = self._index.fetch(ids=[vector_id], namespace=ns_str)
            vectors = resp.get("vectors", {})
            if vector_id not in vectors:
                return None
            meta = vectors[vector_id].get("metadata", {})
            return _item_from_metadata(meta)
        except Exception as exc:
            log.warning(f"[PINECONE_STORE] Get failed  id={vector_id}  err={exc}")
            return None

    def _handle_put(self, op: PutOp) -> None:
        """
        Upsert or delete a single item.

        value is None → delete the vector from Pinecone.
        value is dict → embed the "text" field and upsert.

        Metadata schema stores all value fields plus the four envelope fields
        (namespace, key, created_at, updated_at) needed to reconstruct an Item
        on retrieval. created_at and updated_at both use the current timestamp
        since Pinecone has no concept of first-created vs last-updated.

        WHY embed "text" specifically (not all fields):
          The episodic entry value dict is always {"text": "Q:...\nA:...", "ts": float}.
          "text" is the field that carries semantic content worth embedding.
          "ts" is a numeric timestamp with no semantic meaning to embed.
          If future entry schemas add other text fields, extend this logic.
        """
        ns_str    = _ns_to_str(op.namespace)
        vector_id = _vector_id(ns_str, op.key)

        if op.value is None:
            # Delete operation — value=None signals removal per PutOp contract.
            try:
                self._index.delete(ids=[vector_id], namespace=ns_str)
                log.info(f"[PINECONE_STORE] Deleted  id={vector_id}")
            except Exception as exc:
                log.warning(f"[PINECONE_STORE] Delete failed  id={vector_id}  err={exc}")
            return None

        # Upsert — embed the text field to enable semantic search on retrieval.
        text_to_embed = op.value.get("text", op.key)  # fallback to key if no text
        try:
            vector = self._embed(text_to_embed)
        except Exception as exc:
            log.warning(f"[PINECONE_STORE] Embed failed  id={vector_id}  err={exc}")
            return None

        now_iso = datetime.now(timezone.utc).isoformat()
        metadata = {
            "namespace":  ns_str,
            "key":        op.key,
            "created_at": now_iso,
            "updated_at": now_iso,
            **op.value,  # includes "text", "ts", and any other value fields
        }

        try:
            self._index.upsert(
                vectors=[{"id": vector_id, "values": vector, "metadata": metadata}],
                namespace=ns_str,
            )
            log.info(f"[PINECONE_STORE] Upserted  id={vector_id}  ns={ns_str}")
        except Exception as exc:
            log.warning(f"[PINECONE_STORE] Upsert failed  id={vector_id}  err={exc}")
        return None

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        """
        Search within a namespace prefix.

        Two modes depending on whether a query string is provided:

        Mode 1 — Semantic search (query is non-empty string):
          Embed the query and call index.query() with cosine similarity.
          Returns up to op.limit results above no minimum threshold —
          all returned results are assumed relevant since Pinecone's ANN
          index is the relevance judge. Score is passed through to SearchItem.

        Mode 2 — Recency fetch (query is None or empty string):
          Called by EpisodicMemoryMiddleware and @dynamic_prompt with query="".
          Pinecone has no "list all by recency" operation, so this falls back
          to index.list() to get IDs in the namespace, then index.fetch() to
          retrieve their metadata. Results are sorted by the "ts" field
          (Unix timestamp stored at write time) and limited to op.limit.
          No score is meaningful here — SearchItem.score is set to None.

          WHY list() then fetch() rather than query(zero_vector):
            A zero vector has undefined cosine similarity to all stored vectors
            and would return arbitrary results. list() is semantically correct
            for "give me all entries in this namespace".
        """
        ns_str = _ns_to_str(op.namespace_prefix)

        if op.query:
            return self._semantic_search(ns_str, op.query, op.limit)
        else:
            return self._recency_fetch(ns_str, op.limit)

    def _semantic_search(self, ns_str: str, query: str, limit: int) -> list[SearchItem]:
        """Embed query and return top-k similar episodic entries."""
        try:
            vector = self._embed(query)
            resp   = self._index.query(
                vector=vector,
                top_k=limit,
                namespace=ns_str,
                include_metadata=True,
            )
            items = []
            for match in resp.get("matches", []):
                meta  = match.get("metadata", {})
                score = match.get("score")
                items.append(_item_from_metadata(meta, score=score))
            log.info(f"[PINECONE_STORE] Semantic search  ns={ns_str}  hits={len(items)}")
            return items
        except Exception as exc:
            log.warning(f"[PINECONE_STORE] Semantic search failed  ns={ns_str}  err={exc}")
            return []

    def _recency_fetch(self, ns_str: str, limit: int) -> list[SearchItem]:
        """
        List all IDs in the namespace, fetch their metadata, sort by timestamp.
        Used when query="" (EpisodicMemoryMiddleware + @dynamic_prompt path).
        """
        try:
            # index.list() returns a paginator — iterate to collect all IDs.
            # For episodic memory (3–20 entries per user) this is one page.
            all_ids = []
            for id_batch in self._index.list(namespace=ns_str):
                all_ids.extend(id_batch)

            if not all_ids:
                return []

            # Fetch the most recent `limit` IDs by sorting on "ts" metadata.
            # Fetch all first (small dataset), then sort and slice in Python.
            resp    = self._index.fetch(ids=all_ids, namespace=ns_str)
            vectors = resp.get("vectors", {})

            items = []
            for vid, data in vectors.items():
                meta = data.get("metadata", {})
                items.append(_item_from_metadata(meta, score=_NO_SCORE))

            # Sort by ts descending (most recent first), take top limit.
            items.sort(key=lambda i: i.value.get("ts", 0), reverse=True)
            result = items[:limit]
            log.info(f"[PINECONE_STORE] Recency fetch  ns={ns_str}  hits={len(result)}")
            return result
        except Exception as exc:
            log.warning(f"[PINECONE_STORE] Recency fetch failed  ns={ns_str}  err={exc}")
            return []
