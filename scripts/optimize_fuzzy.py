"""Run fuzzy inference on ANN feature CSV.

This script is the first Fuzzy-only stage. GA optimization can be connected
later through src/aerosweep/fuzzy_ga/ga_optimizer.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aerosweep.fuzzy_ga.fuzzy_logic import evaluate_fuzzy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AeroSweep Fuzzy inference from ANN feature CSV."
    )
    parser.add_argument(
        "--config",
        default="configs/fuzzy.yaml",
        help="Path to fuzzy YAML config.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Override ANN feature CSV path.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Override fuzzy result CSV path.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Include rows with has_detection == 0 in the output.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT_DIR / path


def run_fuzzy_pipeline(
    input_csv: Path,
    output_csv: Path,
    config: dict,
    include_empty: bool = False,
) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    required_columns = {
        "grid_id",
        "area_density_pct",
        "jumlah_instance",
        "kategori_dominan",
        "confidence_score",
        "has_detection",
    }
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Input CSV missing required columns: {missing}")

    if not include_empty and config.get("processing", {}).get("filter_has_detection", True):
        df = df[df["has_detection"] == 1].copy()
    else:
        df = df.copy()

    fuzzy_rows = [
        evaluate_fuzzy(row.to_dict(), config=config)
        for _, row in df.iterrows()
    ]
    fuzzy_df = pd.DataFrame(fuzzy_rows)
    result = pd.concat([df.reset_index(drop=True), fuzzy_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, encoding="utf-8")
    return result


def print_summary(result: pd.DataFrame, output_csv: Path) -> None:
    print("=" * 60)
    print("AeroSweep Fuzzy Inference Finished")
    print("=" * 60)
    print(f"Total rows       : {len(result)}")
    print(f"Output CSV       : {output_csv}")
    if not result.empty:
        print("Risk distribution:")
        for label, count in result["risk_level"].value_counts().items():
            print(f"  {label:<10}: {count}")
        print("\nPreview:")
        columns = [
            "grid_id",
            "area_density_pct",
            "jumlah_instance",
            "confidence_score",
            "kategori_dominan",
            "fuzzy_score",
            "risk_level",
        ]
        print(result[columns].head(10).to_string(index=False))


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    config = load_config(config_path)

    input_csv = resolve_path(args.input_csv or config["input_csv"])
    output_csv = resolve_path(args.output_csv or config["output_csv"])

    result = run_fuzzy_pipeline(
        input_csv=input_csv,
        output_csv=output_csv,
        config=config,
        include_empty=args.include_empty,
    )
    print_summary(result, output_csv)


if __name__ == "__main__":
    main()
