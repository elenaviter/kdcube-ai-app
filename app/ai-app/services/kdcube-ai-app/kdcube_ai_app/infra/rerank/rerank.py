# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Elena Viter

from sentence_transformers import CrossEncoder

# If you need consistent thresholds across queries, collect a small validation set of (query, segment, label)
# and see which raw scores or probabilities best separate “relevant” vs “irrelevant.”

def normalize_scores(scores, normalization_method="sigmoid"):
    # 2) Normalize in one vectorized pass
    if normalization_method == "sigmoid":
        # sigmoid if you want each segment’s relevance on an absolute 0–1 scale;
        norm = 1 / (1 + np.exp(-scores))

    elif normalization_method == "softmax":
        # softmax if you want a probability distribution across your N candidates (e.g. for downstream pooling or ensemble)
        # subtract max for numerical stability
        shifted = scores - scores.max()
        exps = np.exp(shifted)
        norm = exps / exps.sum()

    elif normalization_method == "minmax":
        mn, mx = scores.min(), scores.max()
        if mx > mn:
            norm = (scores - mn) / (mx - mn)
        else:
            norm = np.zeros_like(scores)
    else:
        raise ValueError("unknown method")
    return norm

# Load once at startup (thread‐safe in most server frameworks)
# You can swap in any model from the hub, e.g. 'cross-encoder/ms-marco-MiniLM-L-6-v2'
marco_mini_cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

import numpy as np
from typing import List, Dict, Any

def cross_encoder_rerank(
        query: str,
        candidates: List[Dict[str, Any]],
        column_name: str = "text",
        cross_encoder: str = marco_mini_cross_encoder,
        normalization_method: str = "sigmoid",
        top_k: int = None
) -> List[Dict[str, Any]]:
    """
    Rerank a list of candidate segments using a cross-encoder, then normalize
    the raw logits in-place using one of: "sigmoid", "softmax", or "minmax".

    Args:
      query: the user’s query string
      candidates: list of dicts, each must have at least:
        - 'segment_id'
        - column_name (e.g. 'text') with the actual segment content
      cross_encoder: a sentence-transformers CrossEncoder instance
      normalization_method: "sigmoid", "softmax", or "minmax"
      top_k: if provided, truncate to this many highest‐scoring

    Returns:
      same list of dicts, each annotated with:
        - 'rerank_score': float
      sorted descending by that score.
    """
    # 1) Get raw scores
    pairs = [(query, c[column_name]) for c in candidates]
    scores = cross_encoder.predict(pairs, convert_to_numpy=True)  # shape (N,)

    norm = normalize_scores(scores, normalization_method=normalization_method)

    # 3) Annotate back into the dicts and sort
    for c, s in zip(candidates, norm):
        c["rerank_score"] = float(s)

    candidates.sort(key=lambda c: c["rerank_score"], reverse=True)

    # 4) Truncate if needed
    if top_k is not None:
        return candidates[:top_k]
    return candidates