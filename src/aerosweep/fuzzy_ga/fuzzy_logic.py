"""Fuzzy logic rules and membership functions for AeroSweep.

The module receives feature rows produced by the ANN/CV pipeline and returns a
priority score for each detected grid. The implementation is intentionally
small and dependency-light so it can be explained easily in a soft-computing
report before GA optimization is added.
"""

from __future__ import annotations

from typing import Any, Dict


DEFAULT_CONFIG: Dict[str, Any] = {
    "variables": {
        "area_density_pct": {
            "low": [0, 0, 5, 15],
            "medium": [10, 25, 40],
            "high": [30, 60, 100, 100],
        },
        "jumlah_instance": {
            "few": [0, 0, 2, 5],
            "medium": [3, 10, 20],
            "many": [15, 35, 100, 100],
        },
        "confidence_score": {
            "low": [0.0, 0.0, 0.30, 0.50],
            "medium": [0.35, 0.60, 0.80],
            "high": [0.70, 0.90, 1.0, 1.0],
        },
    },
    "category_weights": {"none": 0.0, "Asbestos": 1.0},
    "risk_output": {
        "centers": {"low": 20, "medium": 50, "high": 75, "critical": 95},
        "levels": {
            "low": [0, 40],
            "medium": [40, 65],
            "high": [65, 85],
            "critical": [85, 100],
        },
    },
}


def triangular(x: float, points: list[float]) -> float:
    """Return triangular membership degree for [a, b, c]."""
    a, b, c = points
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return _safe_ratio(x - a, b - a)
    return _safe_ratio(c - x, c - b)


def trapezoidal(x: float, points: list[float]) -> float:
    """Return trapezoidal membership degree for [a, b, c, d]."""
    a, b, c, d = points
    if x <= a:
        return 1.0 if a == b and x == a else 0.0
    if x >= d:
        return 1.0 if c == d and x == d else 0.0
    if b <= x <= c:
        return 1.0
    if a < x < b:
        return _safe_ratio(x - a, b - a)
    return _safe_ratio(d - x, d - c)


def membership_degree(x: float, points: list[float]) -> float:
    """Evaluate triangular or trapezoidal membership from point count."""
    if len(points) == 3:
        value = triangular(x, points)
    elif len(points) == 4:
        value = trapezoidal(x, points)
    else:
        raise ValueError(f"Membership points must contain 3 or 4 values: {points}")
    return max(0.0, min(1.0, float(value)))


def fuzzify(inputs: Dict[str, float], config: Dict[str, Any] | None = None) -> Dict[str, Dict[str, float]]:
    """Convert crisp ANN features into fuzzy membership degrees."""
    cfg = _merge_config(config)
    variables = cfg["variables"]

    density = float(inputs.get("area_density_pct", 0.0))
    count = float(inputs.get("jumlah_instance", 0.0))
    confidence = float(inputs.get("confidence_score", 0.0))

    return {
        "density": {
            name: membership_degree(density, points)
            for name, points in variables["area_density_pct"].items()
        },
        "count": {
            name: membership_degree(count, points)
            for name, points in variables["jumlah_instance"].items()
        },
        "confidence": {
            name: membership_degree(confidence, points)
            for name, points in variables["confidence_score"].items()
        },
    }


def evaluate_fuzzy(
    inputs: Dict[str, Any],
    config: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Evaluate one ANN feature row and return fuzzy priority output.

    Expected inputs:
        area_density_pct, jumlah_instance, confidence_score, kategori_dominan

    Returns:
        fuzzy_score, risk_level, rule_strengths, category_weight
    """
    cfg = _merge_config(config)
    memberships = fuzzify(inputs, cfg)
    category = str(inputs.get("kategori_dominan", "none"))
    category_weight = float(cfg.get("category_weights", {}).get(category, 0.60))

    d = memberships["density"]
    n = memberships["count"]
    c = memberships["confidence"]

    rule_strengths = {
        "low": max(
            min(d["low"], n["few"]),
            c["low"],
        ),
        "medium": max(
            min(d["medium"], n["few"], c["medium"]),
            min(d["low"], n["medium"], c["high"]),
            min(d["medium"], n["medium"]),
        ),
        "high": max(
            min(d["high"], n["medium"]),
            min(d["medium"], n["many"]),
            min(d["high"], c["high"]),
            min(n["many"], c["high"]),
        ),
        "critical": max(
            min(d["high"], n["many"], c["high"]),
            min(d["high"], n["many"], category_weight),
            min(d["medium"], n["many"], category_weight),
        ),
    }

    fuzzy_score = _defuzzify(rule_strengths, cfg["risk_output"]["centers"])
    fuzzy_score = _apply_category_adjustment(fuzzy_score, category_weight)
    risk_level = classify_risk(fuzzy_score, cfg)

    return {
        "fuzzy_score": round(fuzzy_score, 2),
        "risk_level": risk_level,
        "category_weight": round(category_weight, 2),
        "rule_low": round(rule_strengths["low"], 4),
        "rule_medium": round(rule_strengths["medium"], 4),
        "rule_high": round(rule_strengths["high"], 4),
        "rule_critical": round(rule_strengths["critical"], 4),
    }


def classify_risk(score: float, config: Dict[str, Any] | None = None) -> str:
    """Map numeric fuzzy score into a linguistic risk label."""
    cfg = _merge_config(config)
    for label, bounds in cfg["risk_output"]["levels"].items():
        low, high = bounds
        if low <= score < high:
            return str(label)
    return "critical" if score >= 100 else "low"


def _defuzzify(rule_strengths: Dict[str, float], centers: Dict[str, float]) -> float:
    """Defuzzify using weighted average of output linguistic centers."""
    numerator = 0.0
    denominator = 0.0
    for label, strength in rule_strengths.items():
        numerator += float(strength) * float(centers[label])
        denominator += float(strength)
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _apply_category_adjustment(score: float, category_weight: float) -> float:
    """Slightly raise risk for hazardous or heavy-material categories."""
    adjusted = score + (category_weight - 0.60) * 15.0
    return max(0.0, min(100.0, adjusted))


def _merge_config(config: Dict[str, Any] | None) -> Dict[str, Any]:
    if config is None:
        return DEFAULT_CONFIG
    merged = dict(DEFAULT_CONFIG)
    merged.update(config)
    return merged


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator
