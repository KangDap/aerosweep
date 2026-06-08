"""CLI runner for GA fuzzy parameter optimization.

Usage:
    python scripts/run_ga.py --config configs/fuzzy.yaml --input-csv outputs/output_fitur_ann_ke_fuzzy.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aerosweep.fuzzy_ga.ga_optimizer import optimize_fuzzy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GA optimization on fuzzy parameters.")
    parser.add_argument("--config", default="configs/fuzzy.yaml", help="Path to fuzzy YAML config.")
    parser.add_argument("--input-csv", default="outputs/output_fitur_ann_ke_fuzzy.csv", help="ANN feature CSV path.")
    parser.add_argument("--output-json", default="outputs/ga_best_config.json", help="Output JSON for best config.")
    parser.add_argument("--pop-size", type=int, default=None, help="Override population size.")
    parser.add_argument("--generations", type=int, default=None, help="Override generation count.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    return parser.parse_args()


def resolve_path(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else ROOT_DIR / path


def _progress(gen: int, total: int, best: float, avg: float) -> None:
    bar_len = 30
    filled = int(bar_len * gen / total)
    bar = "#" * filled + "-" * (bar_len - filled)
    print(f"\r  [{bar}] {gen}/{total}  best={best:.4f}  avg={avg:.4f}", end="", flush=True)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config)
    input_csv = resolve_path(args.input_csv)
    output_json = resolve_path(args.output_json)

    ga_overrides = {}
    if args.pop_size is not None:
        ga_overrides["population_size"] = args.pop_size
    if args.generations is not None:
        ga_overrides["generations"] = args.generations
    if args.seed is not None:
        ga_overrides["random_seed"] = args.seed

    print("=" * 60)
    print("AeroSweep -- GA Fuzzy Parameter Optimization")
    print("=" * 60)
    print(f"  Config     : {config_path}")
    print(f"  Input CSV  : {input_csv}")
    print(f"  Output JSON: {output_json}")
    print()

    result = optimize_fuzzy(
        data_path=str(input_csv),
        config_path=str(config_path),
        ga_params=ga_overrides if ga_overrides else None,
        progress_callback=_progress,
    )

    print()  # newline after progress bar
    print()
    print("=" * 60)
    print("Optimization Complete!")
    print("=" * 60)
    print(f"  Best fitness : {result['best_fitness']:.4f}")
    print(f"  Elapsed      : {result['elapsed_seconds']:.1f}s")
    print(f"  Generations  : {result['ga_params']['generations']}")
    print(f"  Population   : {result['ga_params']['population_size']}")
    print()

    # Save best config
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result["best_config"], f, indent=2)
    print(f"  Best config saved to: {output_json}")

    # Save fitness history
    history_path = output_json.parent / "ga_fitness_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(result["fitness_history"], f)
    print(f"  Fitness history saved to: {history_path}")


if __name__ == "__main__":
    main()
