"""Generate sample ANN-feature CSV with ground-truth risk labels.

Creates a realistic synthetic dataset for development and testing of the
GA optimizer and Streamlit dashboard before the real ANN output is available.

Usage:
    python scripts/generate_sample_data.py [--rows 200] [--output outputs/output_fitur_ann_ke_fuzzy.csv]
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent

CATEGORIES = [
    "Rubble", "Construction and demolition materials", "Asphalt milling",
    "Excavation materials", "Appliances", "Electronic equipment",
    "Furniture", "Metal barrels", "Plastic packaging", "Wood",
    "Pallets", "Scrap", "Plastic", "Vehicles", "Tyres",
    "Paper", "Foundry", "Asbestos", "Textile", "Mixed items",
]


def _assign_risk_gt(area: float, count: int, confidence: float, category: str) -> str:
    """Heuristic ground-truth label based on input features."""
    # Simple scoring heuristic
    score = 0.0
    score += min(area / 100.0, 1.0) * 35
    score += min(count / 50.0, 1.0) * 30
    score += confidence * 20

    # Category hazard bonus
    hazardous = {"Asbestos", "Foundry", "Scrap", "Vehicles", "Metal barrels"}
    moderate = {"Rubble", "Construction and demolition materials", "Asphalt milling"}
    if category in hazardous:
        score += 15
    elif category in moderate:
        score += 8

    # Add small noise
    score += random.gauss(0, 3)
    score = max(0, min(100, score))

    if score < 35:
        return "low"
    elif score < 58:
        return "medium"
    elif score < 78:
        return "high"
    else:
        return "critical"


def generate_sample_data(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic ANN-feature CSV."""
    rng = random.Random(seed)
    np.random.seed(seed)

    rows = []
    grid_idx = 0
    for img_idx in range(1, (n_rows // 10) + 2):
        for r in range(5):
            for c in range(5):
                if grid_idx >= n_rows:
                    break
                grid_id = f"UAV_site{img_idx:02d}_grid_{r}_{c}"

                # ~20% chance of no detection
                has_det = 1 if rng.random() > 0.20 else 0

                if has_det:
                    area = round(max(0, rng.gauss(15, 18)), 4)
                    area = min(area, 95.0)
                    count = max(1, int(rng.gauss(8, 10)))
                    count = min(count, 80)
                    confidence = round(max(0.1, min(1.0, rng.gauss(0.65, 0.20))), 4)
                    category = rng.choice(CATEGORIES)
                    cx = round(rng.uniform(50, 462), 2)
                    cy = round(rng.uniform(50, 462), 2)
                    risk_gt = _assign_risk_gt(area, count, confidence, category)
                else:
                    area = 0.0
                    count = 0
                    confidence = 0.0
                    category = "none"
                    cx = -1.0
                    cy = -1.0
                    risk_gt = "low"

                rows.append({
                    "grid_id": grid_id,
                    "centroid_x": cx,
                    "centroid_y": cy,
                    "area_density_pct": area,
                    "jumlah_instance": count,
                    "kategori_dominan": category,
                    "confidence_score": confidence,
                    "processed_at": "2025-05-25 10:00:00",
                    "has_detection": has_det,
                    "risk_gt": risk_gt,
                })
                grid_idx += 1
            if grid_idx >= n_rows:
                break
        if grid_idx >= n_rows:
            break

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sample ANN-feature CSV.")
    parser.add_argument("--rows", type=int, default=200, help="Number of rows.")
    parser.add_argument(
        "--output",
        default="outputs/output_fitur_ann_ke_fuzzy.csv",
        help="Output CSV path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_sample_data(args.rows, args.seed)
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"[OK] Generated {len(df)} rows -> {output_path}")
    print(f"  has_detection=1: {(df['has_detection']==1).sum()}")
    print(f"  Risk distribution: {df[df['has_detection']==1]['risk_gt'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
