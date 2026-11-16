"""A stand-in for the ChromaDB client, used to test ``ChromaStore`` without the library.

It follows the documented behaviour of ChromaDB 1.x closely enough to catch mistakes in the adapter: metadata must be
non-empty dicts of str/int/float/bool (or None), ids are unique, ``query`` returns lists of lists, distances are
``1 - cosine`` for the cosine space, and ``where`` understands equality, ``$in`` and ``$and``. It is NOT a test of
ChromaDB itself - ``scripts/check_stack.py`` does that on the real library.
"""

from __future__ import annotations

import numpy as np


def _valid(value) -> bool:
    return isinstance(value, (str, int, float, bool))


def _matches(metadata: dict, where: dict | None) -> bool:
    if not where:
        return True
    if "$and" in where:
        return all(_matches(metadata, w) for w in where["$and"])
    ((key, cond),) = where.items()
    if key not in metadata:
        return False
    if isinstance(cond, dict):
        ((op, operand),) = cond.items()
        assert op == "$in", f"unsupported operator {op}"
        return metadata[key] in operand
    return metadata[key] == cond


class FakeCollection:
    def __init__(self, name: str, space: str) -> None:
        self.name, self.space = name, space
        self.rows: dict[str, tuple[np.ndarray, str, dict | None]] = {}
        self.add_calls = 0
        self.max_batch = 0

    def add(self, ids, embeddings, documents, metadatas):
        self.add_calls += 1
        self.max_batch = max(self.max_batch, len(ids))
        assert len(ids) == len(embeddings) == len(documents) == len(metadatas)
        for rid, vector, text, meta in zip(ids, embeddings, documents, metadatas, strict=True):
            assert isinstance(rid, str) and rid
            assert isinstance(vector, list) and all(isinstance(x, float) for x in vector), "embeddings must be lists of Python floats"
            if rid in self.rows:
                raise ValueError(f"Expected IDs to be unique, found duplicates of: {rid}")
            if meta is not None:
                if not meta:
                    raise ValueError("Expected metadata to be a non-empty dict")
                assert all(isinstance(k, str) and _valid(v) for k, v in meta.items()), f"bad metadata {meta}"
            self.rows[rid] = (np.array(vector, dtype=np.float32), text, meta)

    def count(self):
        return len(self.rows)

    def query(self, query_embeddings, n_results, where=None, include=None):
        assert len(query_embeddings) == 1 and isinstance(query_embeddings[0], list)
        assert set(include) == {"documents", "metadatas", "distances"}
        assert n_results >= 1
        q = np.array(query_embeddings[0], dtype=np.float32)
        scored = []
        for rid in sorted(self.rows):
            vector, text, meta = self.rows[rid]
            if not _matches(meta or {}, where):
                continue
            if self.space == "cosine":
                distance = 1.0 - float(vector @ q / ((np.linalg.norm(vector) * np.linalg.norm(q)) or 1.0))
            elif self.space == "ip":
                distance = 1.0 - float(vector @ q)
            else:
                distance = float(np.sum((vector - q) ** 2))
            scored.append((distance, rid, text, meta))
        scored.sort(key=lambda row: (row[0], row[1]))
        top = scored[:n_results]
        return {
            "ids": [[r[1] for r in top]],
            "documents": [[r[2] for r in top]],
            "metadatas": [[r[3] for r in top]],
            "distances": [[r[0] for r in top]],
        }

    def get(self, ids=None, limit=None, offset=None, include=None):
        keys = sorted(self.rows) if ids is None else [i for i in ids if i in self.rows]
        if ids is None:
            offset = offset or 0
            keys = keys[offset : offset + (limit if limit is not None else len(keys))]
        return {
            "ids": keys,
            "documents": [self.rows[k][1] for k in keys],
            "metadatas": [self.rows[k][2] for k in keys],
        }


class FakeChromaClient:
    def __init__(self, supports_configuration: bool = True) -> None:
        self.collections: dict[str, FakeCollection] = {}
        self.supports_configuration = supports_configuration
        self.opened_with: list[dict] = []

    def get_or_create_collection(self, name, configuration=None, metadata=None, embedding_function="unset"):
        if configuration is not None and not self.supports_configuration:
            raise TypeError("get_or_create_collection() got an unexpected keyword argument 'configuration'")
        self.opened_with.append({"configuration": configuration, "metadata": metadata, "embedding_function": embedding_function})
        space = ((configuration or {}).get("hnsw") or {}).get("space") or (metadata or {}).get("hnsw:space") or "l2"
        if name not in self.collections:
            self.collections[name] = FakeCollection(name, space)
        return self.collections[name]

    def delete_collection(self, name):
        if name not in self.collections:
            raise ValueError(f"Collection {name} does not exist.")
        del self.collections[name]
