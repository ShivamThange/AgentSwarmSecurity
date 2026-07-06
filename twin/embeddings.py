"""Local, zero-download, CPU-only embedder.

Section 3.2 / 6.1 mandate that the *default* detection path be pure compute with
no new inference and no data leaving the box. This is a deterministic hashing
embedder: token unigrams/bigrams + character 3-grams hashed into a fixed-width
vector, L2-normalised. Cosine similarity of two texts then measures semantic
overlap cheaply enough to run on everything.

It is intentionally a stand-in for a real local sentence-encoder. The `Embedder`
interface is the seam: drop in `sentence-transformers` (all-MiniLM-L6-v2) or any
on-prem model without touching the detectors. `HashingEmbedder` guarantees the
demo runs offline with zero model downloads, which is itself the on-prem story.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder:
    """Interface. `dim` and `encode(text) -> np.ndarray` (L2-normalised)."""

    dim: int = 512

    def encode(self, text: str) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _hash(self, feature: str) -> int:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.dim

    @lru_cache(maxsize=4096)
    def encode(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = _tokens(text)
        if not toks:
            return vec
        # token unigrams + bigrams
        feats: list[str] = list(toks)
        feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
        # character 3-grams over the joined string (captures morphology / typos)
        joined = " ".join(toks)
        feats += [joined[i : i + 3] for i in range(len(joined) - 2)]
        for f in feats:
            idx = self._hash(f)
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two already-L2-normalised vectors, clamped [0,1]."""
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    val = float(np.dot(a, b))
    # both non-negative feature counts => similarity in [0,1]; clamp for safety
    return max(0.0, min(1.0, val))


def distance(a: np.ndarray, b: np.ndarray) -> float:
    """Semantic distance in [0,1]: 1 - cosine similarity."""
    return 1.0 - cosine(a, b)


# Process-wide default embedder (swap here to go to a real on-prem model).
default_embedder: Embedder = HashingEmbedder()
