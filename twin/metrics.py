from __future__ import annotations

import time

from prometheus_client import (
    CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest,
)

SPANS_INGESTED = Counter(
    "twin_spans_ingested_total",
    "Spans ingested into the twin, by resulting drift status",
    ["status"],
)
DRIFT_FLAGGED = Counter(
    "twin_drift_flagged_total",
    "Spans flagged as drifted, by classified risk type",
    ["risk_type"],
)
TIER_DECISIONS = Counter(
    "twin_detection_tier_total",
    "Detection tier reached per analysed span",
    ["tier"],
)
BLOCKED_ACTIONS = Counter(
    "twin_blocked_actions_total",
    "Sensitive tool calls denied by the inline rail",
)
REMEDIATION_EVENTS = Counter(
    "twin_remediation_events_total",
    "Remediation lifecycle events",
    ["event"],
)
INGEST_ERRORS = Counter(
    "twin_ingest_errors_total",
    "Span ingestion failures",
    ["reason"],
)
HTTP_REQUESTS = Histogram(
    "twin_http_request_duration_seconds",
    "HTTP request latency",
    ["method", "route", "code"],
    buckets=(0.005, 0.025, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
ESCALATION_RATIO = Gauge(
    "twin_escalation_ratio",
    "Fraction of analysed spans escalated to a judge tier in the window",
)
ESCALATION_ANOMALIES = Counter(
    "twin_escalation_anomalies_total",
    "Times the escalation-rate monitor raised an anomaly",
)
READY = Gauge("twin_ready", "1 when the service is ready to serve traffic")


def render() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


class Timer:
    def __init__(self) -> None:
        self.start = time.perf_counter()

    def observe(self, method: str, route: str, code: int) -> None:
        HTTP_REQUESTS.labels(method=method, route=route,
                             code=str(code)).observe(
            time.perf_counter() - self.start)
