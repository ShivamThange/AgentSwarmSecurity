from __future__ import annotations

"""Threshold calibration from human-review labels.

Operators label detections as ``true_positive`` (genuinely drift) or
``false_positive`` (wrongly flagged). Treating those as ground truth, this
module sweeps candidate flag thresholds and reports precision/recall/F1 so a
threshold can be chosen from evidence instead of a hardcoded default. It only
*recommends*; applying a threshold stays a human decision (set the value in
``TWIN_FLAG_THRESHOLD`` or a workflow profile).
"""

from typing import Optional

TRUE_POSITIVE = "true_positive"
FALSE_POSITIVE = "false_positive"

MIN_LABELS_FOR_RECOMMENDATION = 10


def _metrics_at(points: list[tuple[float, bool]], threshold: float) -> dict:
    tp = fp = fn = tn = 0
    for score, is_drift in points:
        predicted = score >= threshold
        if is_drift and predicted:
            tp += 1
        elif is_drift and not predicted:
            fn += 1
        elif not is_drift and predicted:
            fp += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "threshold": round(threshold, 3),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 3) if precision is not None else None,
        "recall": round(recall, 3) if recall is not None else None,
        "f1": round(f1, 3),
    }


def sweep(points: list[tuple[float, bool]],
          grid_step: float = 0.05) -> list[dict]:
    steps = max(1, int(round(1.0 / grid_step)))
    thresholds = [round(i * grid_step, 3) for i in range(steps + 1)]
    return [_metrics_at(points, t) for t in thresholds]


def recommend(points: list[tuple[float, bool]],
              current_threshold: float,
              target_precision: Optional[float] = None,
              grid_step: float = 0.05) -> dict:
    """Recommend a flag threshold from labelled ``(score, is_drift)`` points.

    Without ``target_precision`` the threshold maximising F1 is chosen. With a
    ``target_precision`` the lowest threshold meeting it (i.e. best recall for
    the required precision) is chosen. Ties break toward the threshold closest
    to the current one to avoid needless churn.
    """
    n = len(points)
    positives = sum(1 for _s, d in points if d)
    grid = sweep(points, grid_step)
    result = {
        "label_count": n,
        "positives": positives,
        "negatives": n - positives,
        "current_threshold": round(current_threshold, 3),
        "current": _metrics_at(points, current_threshold),
        "sweep": grid,
        "recommended_threshold": None,
        "objective": ("target_precision" if target_precision is not None
                      else "max_f1"),
        "sufficient_data": n >= MIN_LABELS_FOR_RECOMMENDATION,
    }
    if n < MIN_LABELS_FOR_RECOMMENDATION or positives == 0:
        result["note"] = (
            f"need >= {MIN_LABELS_FOR_RECOMMENDATION} labels including at least "
            f"one true_positive before recommending a threshold; have "
            f"{n} label(s), {positives} positive(s)")
        return result

    if target_precision is not None:
        eligible = [row for row in grid
                    if row["precision"] is not None
                    and row["precision"] >= target_precision
                    and row["tp"] > 0]
        if not eligible:
            result["note"] = (f"no threshold reaches precision "
                              f">= {target_precision} on the current labels")
            return result
        best = min(eligible, key=lambda r: (r["threshold"],
                                            abs(r["threshold"]
                                                - current_threshold)))
    else:
        best = max(grid, key=lambda r: (r["f1"],
                                        -abs(r["threshold"]
                                             - current_threshold)))

    result["recommended_threshold"] = best["threshold"]
    result["recommended"] = best
    delta = best["threshold"] - current_threshold
    result["delta_vs_current"] = round(delta, 3)
    return result


def calibrate(labeled_points: list[tuple[float, str, str]],
              base_threshold: float,
              profile_threshold: Optional[dict[str, float]] = None,
              target_precision: Optional[float] = None) -> dict:
    """Group labels by workflow and recommend a threshold for each.

    ``labeled_points`` are ``(score, label, workflow)`` triples.
    ``profile_threshold`` maps a workflow to its currently-configured flag
    threshold; workflows absent from it use ``base_threshold``.
    """
    profile_threshold = profile_threshold or {}
    by_workflow: dict[str, list[tuple[float, bool]]] = {}
    for score, label, workflow in labeled_points:
        if label not in (TRUE_POSITIVE, FALSE_POSITIVE):
            continue
        by_workflow.setdefault(workflow or "", []).append(
            (float(score), label == TRUE_POSITIVE))

    overall = [(s, d) for pts in by_workflow.values() for s, d in pts]
    report = {
        "total_labels": len(overall),
        "overall": recommend(overall, base_threshold,
                             target_precision=target_precision),
        "by_workflow": {},
    }
    for workflow, pts in sorted(by_workflow.items()):
        current = profile_threshold.get(workflow, base_threshold)
        report["by_workflow"][workflow or "(default)"] = recommend(
            pts, current, target_precision=target_precision)
    return report
