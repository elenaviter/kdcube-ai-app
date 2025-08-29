# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

import inspect
from typing import List, Optional, Dict, Any, Union
from functools import wraps

DEFAULT_DISTANCE_TYPE = "cosine"
OPERATOR_MAP = {
    "l2":     "<->",   # Euclidean (squared L2) distance
    "cosine": "<=>",   # Cosine distance   (0 … 2)
    "ip":     "<#>",   # Negative inner‑product distance
}


def to_pgvector_str(embedding_list: List[float]) -> str:
    """Convert a Python list of floats to a Postgres pgvector literal like '[0.1,0.2,...]'."""
    return "[" + ",".join(str(x) for x in embedding_list) + "]"


def _distance_to_similarity(d: float, metric: str) -> float:
    """
    Convert a pgvector distance to a similarity score in (0, 1].
    The function is robust: if the distance is unexpectedly negative we
    clamp it to zero before applying the formula.
    """
    if metric == "cosine":
        d = max(0.0, d)               # clamp
        return 1.0 - d / 2.0          # 0→1, 2→0
    elif metric == "l2":
        d = max(0.0, d)
        return 1.0 / (1.0 + d)
    else:  # "ip"
        # distance is ‑⟨u,v⟩, so more negative → more similar
        return 1.0 / (1.0 + max(0.0, -d))

def transactional(func):
    """
    Decorator to handle optional connection usage for each DB method.

    - If the caller passes `conn=some_connection`, we reuse that connection
      (so no open/close/commit/rollback is done here).
    - If the caller omits conn, we open a new connection from self.dbmgr.get_connection(),
      commit on success, rollback on exception, and close at the end.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if "conn" not in inspect.signature(func).parameters:
            return func(self, *args, **kwargs)

        existing_conn = kwargs.pop("conn", None)  # Pop it to avoid passing it twice

        if existing_conn is not None:
            # Reuse the caller's transaction
            kwargs["conn"] = existing_conn  # Add it back as a kwarg
            return func(self, *args, **kwargs)
        else:
            # We open our own connection for this call
            conn = self.dbmgr.get_connection()
            try:
                kwargs["conn"] = conn
                result = func(self, *args, **kwargs)
                conn.commit()
                return result
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()
    return wrapper

_BASE_COLS = {"knowledge_elements_versions": (
    # all non‑vector columns of knowledge_element_versions
    "id", "version", "arn", "is_latest", "ref_count", "status",
    "ke_type", "description", "summary", "metadata", "tags",
    "content", "created_at", "updated_at", "lineage",
)}


def _select_list(weight: str, table_alias: str = "KE", table="knowledge_elements_versions") -> str:
    """
    Return the column list for the SELECT clause.
    * heavy  ->  KE.*            (includes embeds)
    * light  ->  only _BASE_COLS (excludes embeds)
    """
    if weight.lower() != "light":
        return f"{table_alias}.*"
    cols = _BASE_COLS.get(table, ())
    return ", ".join(f"{table_alias}.{c}" for c in cols)
