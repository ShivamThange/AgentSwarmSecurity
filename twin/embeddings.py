from __future__ import annotations

import hashlib
import logging
import re
import threading
from collections import OrderedDict
from typing import Optional

import numpy as np

from .config import Settings

log = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Embedder:
    dim: int = 0
    name: str = "base"

    def encode(self, text: str) -> np.ndarray:
        raise NotImplementedError

    def info(self) -> dict:
        return {"backend": self.name, "dim": self.dim}


class _VectorCache:
    def __init__(self, maxsize: int) -> None:
        self.maxsize = maxsize
        self._data: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[np.ndarray]:
        with self._lock:
            vec = self._data.get(key)
            if vec is not None:
                self._data.move_to_end(key)
            return vec

    def put(self, key: str, vec: np.ndarray) -> None:
        with self._lock:
            self._data[key] = vec
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)


class HashingEmbedder(Embedder):
    name = "hashing-degraded"

    def __init__(self, dim: int = 512, cache_size: int = 8192) -> None:
        self.dim = dim
        self._cache = _VectorCache(cache_size)

    def _hash(self, feature: str) -> int:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(h, "big") % self.dim

    def encode(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = np.zeros(self.dim, dtype=np.float32)
        toks = _tokens(text)
        if toks:
            feats: list[str] = list(toks)
            feats += [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            joined = " ".join(toks)
            feats += [joined[i:i + 3] for i in range(len(joined) - 2)]
            for f in feats:
                vec[self._hash(f)] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
        self._cache.put(text, vec)
        return vec


class SentenceTransformerEmbedder(Embedder):
    name = "sentence-transformers"

    def __init__(self, model_name: str, cache_size: int = 8192,
                 device: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device
        self._cache = _VectorCache(cache_size)
        self._model = None
        self._load_lock = threading.Lock()

    def _get_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from sentence_transformers import SentenceTransformer
                    except ImportError as exc:
                        raise RuntimeError(
                            "embeddings backend 'sentence-transformers' is "
                            "configured but the package is not installed; "
                            "run: pip install -r requirements-ml.txt "
                            "(or set TWIN_EMBEDDINGS_BACKEND=hashing to run "
                            "in degraded lexical mode)"
                        ) from exc
                    log.info("loading embedding model %s", self.model_name)
                    self._model = SentenceTransformer(
                        self.model_name, device=self.device)
                    self.dim = int(
                        self._model.get_sentence_embedding_dimension())
        return self._model

    def encode(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        model = self._get_model()
        vec = model.encode([text], normalize_embeddings=True,
                           show_progress_bar=False)[0]
        vec = np.asarray(vec, dtype=np.float32)
        self._cache.put(text, vec)
        return vec

    def warmup(self) -> None:
        self.encode("warmup")

    def info(self) -> dict:
        return {"backend": self.name, "model": self.model_name,
                "dim": self.dim or None, "loaded": self._model is not None}


def cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    if a.shape != b.shape:
        return 0.0
    val = float(np.dot(a, b))
    return max(0.0, min(1.0, val))


def distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    return 1.0 - cosine(a, b)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embeddings_backend == "sentence-transformers":
        return SentenceTransformerEmbedder(
            settings.embeddings_model,
            cache_size=settings.embeddings_cache_size,
            device=settings.embeddings_device,
        )
    log.warning(
        "embeddings backend is 'hashing' — semantic drift detection is "
        "running in degraded lexical mode; use sentence-transformers in "
        "production")
    return HashingEmbedder(cache_size=settings.embeddings_cache_size)
