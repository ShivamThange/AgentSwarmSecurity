from __future__ import annotations

from twin.escalation import EscalationMonitor


def test_no_anomaly_below_min_samples():
    mon = EscalationMonitor(ratio_threshold=0.1, min_samples=20)
    snap = {}
    for i in range(5):
        snap = mon.record(escalated=True, now=1000.0 + i)
    assert snap["anomaly"] is False  # too few samples to judge


def test_ratio_anomaly_triggers():
    mon = EscalationMonitor(ratio_threshold=0.5, min_samples=10)
    snap = {}
    # 30 spans, 20 escalate -> ratio 0.67 > 0.5
    for i in range(30):
        snap = mon.record(escalated=(i % 3 != 0), now=2000.0 + i)
    assert snap["ratio"] > 0.5
    assert snap["anomaly"] is True
    assert snap["reasons"]
    assert snap["anomalies_seen"] >= 1


def test_low_ratio_no_anomaly():
    mon = EscalationMonitor(ratio_threshold=0.5, min_samples=10)
    snap = {}
    for i in range(30):
        snap = mon.record(escalated=(i % 10 == 0), now=3000.0 + i)
    assert snap["ratio"] < 0.5
    assert snap["anomaly"] is False


def test_window_evicts_old_events():
    mon = EscalationMonitor(window_seconds=100.0, ratio_threshold=0.5,
                            min_samples=5)
    for i in range(20):
        mon.record(escalated=True, now=1000.0 + i)
    # Far in the future: the old window has aged out entirely.
    snap = mon.snapshot(now=1000.0 + 10_000)
    assert snap["analysed"] == 0
    assert snap["escalated"] == 0
    assert snap["anomaly"] is False


def test_absolute_rate_threshold():
    mon = EscalationMonitor(window_seconds=60.0, ratio_threshold=1.1,
                            rate_threshold_per_min=5.0, min_samples=1)
    snap = {}
    for i in range(10):
        snap = mon.record(escalated=True, now=5000.0 + i)
    # 10 escalations in a 60s window = 10/min > 5/min cap
    assert snap["rate_per_min"] == 10.0
    assert snap["anomaly"] is True


def test_snapshot_does_not_inflate_anomaly_count():
    mon = EscalationMonitor(ratio_threshold=0.5, min_samples=10)
    for i in range(30):
        mon.record(escalated=(i % 3 != 0), now=6000.0 + i)
    count_after_record = mon.snapshot(now=6000.0 + 29)["anomalies_seen"]
    # Repeated reads must not bump the counter.
    for _ in range(5):
        again = mon.snapshot(now=6000.0 + 29)["anomalies_seen"]
    assert again == count_after_record
