from __future__ import annotations

import hashlib
import re
from functools import lru_cache

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())

class Embedder:

    dim: int = 512

    def encode(self, text: str) -> np.ndarray:
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

        feats: list[str] = list(toks)
        feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]

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
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    val = float(np.dot(a, b))

    return max(0.0, min(1.0, val))

def distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - cosine(a, b)

default_embedder: Embedder = HashingEmbedder()
