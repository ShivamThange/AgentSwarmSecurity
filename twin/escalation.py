from __future__ import annotations

"""Rolling escalation-rate monitor.

The tiered router escalates a span to a paid judge only when a cheap signal
fires. An adversary who can manufacture cheap signals (crafted foreign
entities, spurious tool calls) could drive escalations to exhaust the deep-judge
budget or bury a real incident in noise. This monitor watches the escalation
rate over a sliding window and raises an anomaly when it spikes past configured
bounds — a signal to rate-limit or fall back to the deterministic tier, not to
silently absorb the cost.

It is in-memory and thread-safe; ``now`` is injectable so the logic is testable
without real time.
"""

import threading
import time
from collections import deque
from typing import Optional


class EscalationMonitor:
    def __init__(self, window_seconds: float = 300.0,
                 ratio_threshold: float = 0.5,
                 rate_threshold_per_min: Optional[float] = None,
                 min_samples: int = 20) -> None:
        self.window_seconds = max(1.0, window_seconds)
        self.ratio_threshold = ratio_threshold
        self.rate_threshold_per_min = rate_threshold_per_min
        self.min_samples = max(1, min_samples)
        self._analysed: deque[float] = deque()
        self._escalated: deque[float] = deque()
        self._lock = threading.Lock()
        self._anomaly_count = 0

    def _evict(self, now: float) -> None:
        cutoff = now - self.window_seconds
        for dq in (self._analysed, self._escalated):
            while dq and dq[0] < cutoff:
                dq.popleft()

    def record(self, escalated: bool, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            self._analysed.append(now)
            if escalated:
                self._escalated.append(now)
            self._evict(now)
            snap = self._snapshot_locked(now)
            if snap["anomaly"]:
                self._anomaly_count += 1
                snap["anomalies_seen"] = self._anomaly_count
            return snap

    def snapshot(self, now: Optional[float] = None) -> dict:
        now = time.time() if now is None else now
        with self._lock:
            self._evict(now)
            return self._snapshot_locked(now)

    def _snapshot_locked(self, now: float) -> dict:
        analysed = len(self._analysed)
        escalated = len(self._escalated)
        ratio = (escalated / analysed) if analysed else 0.0
        rate_per_min = escalated / (self.window_seconds / 60.0)
        reasons: list[str] = []
        if analysed >= self.min_samples and ratio > self.ratio_threshold:
            reasons.append(
                f"escalation ratio {ratio:.2f} exceeds "
                f"{self.ratio_threshold:.2f} over the last "
                f"{int(self.window_seconds)}s ({escalated}/{analysed} spans)")
        if (self.rate_threshold_per_min is not None
                and rate_per_min > self.rate_threshold_per_min):
            reasons.append(
                f"escalation rate {rate_per_min:.1f}/min exceeds "
                f"{self.rate_threshold_per_min:.1f}/min")
        anomaly = bool(reasons)
        return {
            "window_seconds": int(self.window_seconds),
            "analysed": analysed,
            "escalated": escalated,
            "ratio": round(ratio, 3),
            "rate_per_min": round(rate_per_min, 2),
            "ratio_threshold": self.ratio_threshold,
            "rate_threshold_per_min": self.rate_threshold_per_min,
            "anomaly": anomaly,
            "reasons": reasons,
            "anomalies_seen": self._anomaly_count,
        }
