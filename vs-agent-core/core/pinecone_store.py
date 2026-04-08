"""
pinecone_store.py — PineconeStore
==================================
Wraps Pinecone so LangGraph's EpisodicMemoryMiddleware and @dynamic_prompt
can use it as a store without knowing anything about Pinecone.

Both of them call store.put() and store.search() — standard LangGraph
BaseStore methods. We just implement those methods using Pinecone underneath.

Namespace:
  LangGraph uses tuples like ("episodic", "user_abc").
  Pinecone uses plain strings.
  We join them with "__"  →  "episodic__user_abc"
"""

import logging
from datetime import datetime, timezone
from typing import Any, Iterable

from langchain_openai import OpenAIEmbeddings
from langgraph.store.base import (
    BaseStore, GetOp, Item, ListNamespacesOp,
    Op, PutOp, Result, SearchItem, SearchOp,
)

log = logging.getLogger(__name__)


def _ns(namespace: tuple) -> str:
    """("episodic", "user_abc")  →  "episodic__user_abc" """
    return "__".join(namespace)


def _vid(namespace_str: str, key: str) -> str:
    """Unique Pinecone vector ID — namespace prefix avoids key collisions."""
    return f"{namespace_str}__{key}"


def _to_item(meta: dict, score: float = None) -> SearchItem:
    """Rebuild a LangGraph SearchItem from Pinecone metadata."""
    skip = {"namespace", "key", "created_at", "updated_at"}
    return SearchItem(
        namespace=tuple(meta["namespace"].split("__")),
        key=meta["key"],
        value={k: v for k, v in meta.items() if k not in skip},
        created_at=datetime.fromisoformat(meta["created_at"]),
        updated_at=datetime.fromisoformat(meta["updated_at"]),
        score=score,
    )


class PineconeStore(BaseStore):
    """
    LangGraph BaseStore backed by Pinecone.

    Used by EpisodicMemoryMiddleware (put + search) and @dynamic_prompt (search).
    Both get the same instance so writes are immediately visible on the next turn.
    """

    def __init__(self, index: Any, embedder: OpenAIEmbeddings, top_k: int = 3):
        self._index    = index
        self._embedder = embedder
        self._top_k    = top_k

    def _embed(self, text: str) -> list[float]:
        return self._embedder.embed_query(text)

    # ── LangGraph calls batch() for every get/put/search/delete ───────────

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        results = []
        for op in ops:
            if isinstance(op, GetOp):
                results.append(self._get(op))
            elif isinstance(op, PutOp):
                results.append(self._put(op))
            elif isinstance(op, SearchOp):
                results.append(self._search(op))
            elif isinstance(op, ListNamespacesOp):
                results.append([])
            else:
                results.append(None)
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        # Pinecone client is sync — delegate to batch()
        return self.batch(ops)

    # ── Handlers ──────────────────────────────────────────────────────────

    def _get(self, op: GetOp) -> Item | None:
        """Fetch one item by namespace + key."""
        ns  = _ns(op.namespace)
        vid = _vid(ns, op.key)
        try:
            resp = self._index.fetch(ids=[vid], namespace=ns)
            vec  = resp.get("vectors", {}).get(vid)
            return _to_item(vec["metadata"]) if vec else None
        except Exception as e:
            log.warning(f"[STORE] get failed  id={vid}  err={e}")
            return None

    def _put(self, op: PutOp) -> None:
        """Upsert (value is dict) or delete (value is None)."""
        ns  = _ns(op.namespace)
        vid = _vid(ns, op.key)

        if op.value is None:
            # Delete
            try:
                self._index.delete(ids=[vid], namespace=ns)
                log.info(f"[STORE] deleted  id={vid}")
            except Exception as e:
                log.warning(f"[STORE] delete failed  id={vid}  err={e}")
            return

        # Upsert — embed the text field, store everything else as metadata
        try:
            vector = self._embed(op.value.get("text", op.key))
        except Exception as e:
            log.warning(f"[STORE] embed failed  id={vid}  err={e}")
            return

        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "namespace":  ns,
            "key":        op.key,
            "created_at": now,
            "updated_at": now,
            **op.value,
        }

        try:
            self._index.upsert(
                vectors=[{"id": vid, "values": vector, "metadata": metadata}],
                namespace=ns,
            )
            log.info(f"[STORE] upserted  id={vid}  ns={ns}")
        except Exception as e:
            log.warning(f"[STORE] upsert failed  id={vid}  err={e}")

    def _search(self, op: SearchOp) -> list[SearchItem]:
        """
        Search within a namespace.
        - query provided  → semantic (vector) search
        - query empty     → return most recent entries
        """
        ns = _ns(op.namespace_prefix)
        if op.query:
            return self._semantic_search(ns, op.query, op.limit)
        else:
            return self._recent(ns, op.limit)

    def _semantic_search(self, ns: str, query: str, limit: int) -> list[SearchItem]:
        """Embed query → find similar episodic entries in Pinecone."""
        try:
            resp = self._index.query(
                vector=self._embed(query),
                top_k=limit,
                namespace=ns,
                include_metadata=True,
            )
            items = [
                _to_item(m["metadata"], score=m.get("score"))
                for m in resp.get("matches", [])
            ]
            log.info(f"[STORE] semantic search  ns={ns}  hits={len(items)}")
            return items
        except Exception as e:
            log.warning(f"[STORE] semantic search failed  ns={ns}  err={e}")
            return []

    def _recent(self, ns: str, limit: int) -> list[SearchItem]:
        """
        Return the most recent entries by timestamp.
        Used when query="" — no vector search needed, just recency.
        """
        try:
            # Collect all IDs in this namespace
            all_ids = [id for batch in self._index.list(namespace=ns) for id in batch]
            if not all_ids:
                return []

            # Fetch metadata for all IDs
            vectors = self._index.fetch(ids=all_ids, namespace=ns).get("vectors", {})

            # Build items, sort by ts (newest first), return top limit
            items = [_to_item(v["metadata"]) for v in vectors.values()]
            items.sort(key=lambda i: i.value.get("ts", 0), reverse=True)

            log.info(f"[STORE] recent fetch  ns={ns}  hits={len(items[:limit])}")
            return items[:limit]
        except Exception as e:
            log.warning(f"[STORE] recent fetch failed  ns={ns}  err={e}")
            return []