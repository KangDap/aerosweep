"""Genetic Algorithm optimizer for fuzzy membership-function parameters.

Optimises the tuneable points of the triangular / trapezoidal membership
functions and risk-output centres used by :mod:`fuzzy_logic` so that the
resulting fuzzy scores best match a ground-truth risk labelling (supervised
mode) or produce well-separated risk clusters (unsupervised fallback).
"""

from __future__ import annotations

import copy
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from .fuzzy_logic import evaluate_fuzzy, DEFAULT_CONFIG
from .ga_utils import (
    blx_alpha_crossover,
    chromosome_length,
    decode_chromosome,
    encode_config,
    encode_risk_label,
    gaussian_mutation,
    generate_initial_population,
    repair_chromosome,
    tournament_selection,
)

logger = logging.getLogger(__name__)

# ── Default GA hyper-parameters ──────────────────────────────────────────────

_GA_DEFAULTS: Dict[str, Any] = {
    "population_size": 50,
    "generations": 40,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8,
    "tournament_size": 3,
    "elitism_count": 2,
    "blx_alpha": 0.5,
    "mutation_sigma": 0.1,
    "fitness_metric": "mae",  # "mae" | "mse" | "distribution"
    "random_seed": 42,
}


# ── Fitness functions ────────────────────────────────────────────────────────

def _fitness_mae(
    chromosome: np.ndarray,
    df: pd.DataFrame,
    template_config: Dict[str, Any],
    gt_col: str = "risk_gt",
) -> float:
    """Mean Absolute Error between fuzzy_score and encoded ground-truth label."""
    config = decode_chromosome(chromosome, template_config)
    errors: List[float] = []
    for _, row in df.iterrows():
        result = evaluate_fuzzy(row.to_dict(), config=config)
        gt_value = encode_risk_label(str(row[gt_col]))
        errors.append(abs(result["fuzzy_score"] - gt_value))
    return float(np.mean(errors))


def _fitness_mse(
    chromosome: np.ndarray,
    df: pd.DataFrame,
    template_config: Dict[str, Any],
    gt_col: str = "risk_gt",
) -> float:
    """Mean Squared Error between fuzzy_score and encoded ground-truth label."""
    config = decode_chromosome(chromosome, template_config)
    errors: List[float] = []
    for _, row in df.iterrows():
        result = evaluate_fuzzy(row.to_dict(), config=config)
        gt_value = encode_risk_label(str(row[gt_col]))
        errors.append((result["fuzzy_score"] - gt_value) ** 2)
    return float(np.mean(errors))


def _fitness_distribution(
    chromosome: np.ndarray,
    df: pd.DataFrame,
    template_config: Dict[str, Any],
    **_kwargs: Any,
) -> float:
    """Unsupervised fitness: maximise inter-cluster separation.

    Lower return value = better fitness (we negate the separation score).
    Penalises heavily skewed distributions where one risk level dominates.
    """
    config = decode_chromosome(chromosome, template_config)
    scores: List[float] = []
    levels: List[str] = []
    for _, row in df.iterrows():
        result = evaluate_fuzzy(row.to_dict(), config=config)
        scores.append(result["fuzzy_score"])
        levels.append(result["risk_level"])

    if len(scores) < 2:
        return 1e6

    scores_arr = np.array(scores)

    # Inter-cluster variance (higher = better)
    unique_levels = list(set(levels))
    if len(unique_levels) < 2:
        inter_var = 0.0
    else:
        cluster_means = []
        for lv in unique_levels:
            mask = [l == lv for l in levels]
            cluster_scores = scores_arr[mask]
            if len(cluster_scores) > 0:
                cluster_means.append(np.mean(cluster_scores))
        inter_var = float(np.var(cluster_means)) if len(cluster_means) > 1 else 0.0

    # Skewness penalty: penalise if one level has > 70% of data
    counts = {lv: levels.count(lv) for lv in unique_levels}
    max_pct = max(counts.values()) / len(levels)
    skew_penalty = max(0.0, (max_pct - 0.5)) * 200.0

    # Coverage bonus: reward using more distinct levels
    coverage_bonus = len(unique_levels) * 10.0

    # We want to MINIMISE the returned value, so negate the "good" components
    fitness = -inter_var + skew_penalty - coverage_bonus
    return float(fitness)


_FITNESS_FUNCTIONS: Dict[str, Callable] = {
    "mae": _fitness_mae,
    "mse": _fitness_mse,
    "distribution": _fitness_distribution,
}


# ── Main GA loop ─────────────────────────────────────────────────────────────

def optimize_fuzzy(
    data_path: str | Path,
    config_path: str | Path | None = None,
    ga_params: Dict[str, Any] | None = None,
    progress_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> Dict[str, Any]:
    """Run the Genetic Algorithm to tune fuzzy parameters.

    Parameters
    ----------
    data_path:
        Path to the ANN-feature CSV (must contain the fuzzy input columns).
    config_path:
        Path to ``fuzzy.yaml``.  Falls back to ``DEFAULT_CONFIG`` if *None*.
    ga_params:
        Override GA hyper-parameters (merged on top of config-file values).
    progress_callback:
        Optional ``(generation, total_gens, best_fitness, avg_fitness) -> None``
        called after every generation for UI progress reporting.

    Returns
    -------
    dict with keys:
        ``best_config``   – optimised fuzzy config dict
        ``best_fitness``  – final best fitness value
        ``fitness_history`` – list of (best, avg) per generation
        ``best_chromosome`` – raw best chromosome array
        ``elapsed_seconds`` – wall-clock time
        ``default_config``  – the *original* unoptimised config (for comparison)
        ``ga_params``       – the hyper-parameters actually used
    """
    t0 = time.time()

    # ── Load config ──────────────────────────────────────────────────────
    if config_path is not None:
        config_path = Path(config_path)
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                file_config = yaml.safe_load(f) or {}
        else:
            file_config = {}
    else:
        file_config = {}

    template_config: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
    for key in ("variables", "category_weights", "risk_output"):
        if key in file_config:
            template_config[key] = file_config[key]

    default_config_snapshot = copy.deepcopy(template_config)

    # ── Merge GA params ──────────────────────────────────────────────────
    params: Dict[str, Any] = dict(_GA_DEFAULTS)
    if "ga" in file_config and isinstance(file_config["ga"], dict):
        params.update(file_config["ga"])
    if ga_params:
        params.update(ga_params)

    pop_size: int = int(params["population_size"])
    n_gens: int = int(params["generations"])
    mut_rate: float = float(params["mutation_rate"])
    cx_rate: float = float(params["crossover_rate"])
    tourn_size: int = int(params["tournament_size"])
    elite_count: int = int(params["elitism_count"])
    blx_a: float = float(params["blx_alpha"])
    mut_sigma: float = float(params["mutation_sigma"])
    metric: str = str(params["fitness_metric"])
    seed: int = int(params.get("random_seed", 42))

    rng = random.Random(seed)
    np.random.seed(seed)

    # ── Load data ────────────────────────────────────────────────────────
    df = pd.read_csv(data_path)
    if "has_detection" in df.columns:
        df = df[df["has_detection"] == 1].copy()

    # Determine fitness function
    has_gt = "risk_gt" in df.columns
    if metric in ("mae", "mse") and not has_gt:
        logger.warning(
            "Column 'risk_gt' not found — falling back to 'distribution' fitness."
        )
        metric = "distribution"
    fitness_fn = _FITNESS_FUNCTIONS[metric]

    # ── Initialise population ────────────────────────────────────────────
    population = generate_initial_population(template_config, pop_size, rng=rng)
    fitness_history: List[Tuple[float, float]] = []

    # ── Evaluate initial population ──────────────────────────────────────
    fitness_values = np.array([
        fitness_fn(ind, df, template_config) for ind in population
    ])

    for gen in range(n_gens):
        # ── Sort by fitness (ascending = better) ─────────────────────────
        order = np.argsort(fitness_values)
        population = population[order]
        fitness_values = fitness_values[order]

        best_fit = float(fitness_values[0])
        avg_fit = float(np.mean(fitness_values))
        fitness_history.append((best_fit, avg_fit))

        logger.info(
            "Gen %3d/%d  best=%.4f  avg=%.4f", gen + 1, n_gens, best_fit, avg_fit,
        )

        if progress_callback is not None:
            progress_callback(gen + 1, n_gens, best_fit, avg_fit)

        # ── Elitism ──────────────────────────────────────────────────────
        new_pop = [population[i].copy() for i in range(elite_count)]

        # ── Breed next generation ────────────────────────────────────────
        while len(new_pop) < pop_size:
            p1 = tournament_selection(population, fitness_values, tourn_size, rng)
            p2 = tournament_selection(population, fitness_values, tourn_size, rng)

            if rng.random() < cx_rate:
                c1, c2 = blx_alpha_crossover(p1, p2, blx_a, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = gaussian_mutation(c1, mut_rate, mut_sigma, rng)
            c2 = gaussian_mutation(c2, mut_rate, mut_sigma, rng)

            c1 = repair_chromosome(c1, template_config)
            c2 = repair_chromosome(c2, template_config)

            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)

        population = np.array(new_pop[:pop_size])
        fitness_values = np.array([
            fitness_fn(ind, df, template_config) for ind in population
        ])

    # ── Final sort and extract best ──────────────────────────────────────
    best_idx = int(np.argmin(fitness_values))
    best_chromosome = population[best_idx]
    best_config = decode_chromosome(best_chromosome, template_config)
    elapsed = time.time() - t0

    return {
        "best_config": best_config,
        "best_fitness": float(fitness_values[best_idx]),
        "fitness_history": fitness_history,
        "best_chromosome": best_chromosome.tolist(),
        "elapsed_seconds": round(elapsed, 2),
        "default_config": default_config_snapshot,
        "ga_params": params,
    }


def run_fuzzy_with_config(
    df: pd.DataFrame,
    config: Dict[str, Any],
) -> pd.DataFrame:
    """Evaluate fuzzy for every row in *df* using the given *config*.

    Returns a new DataFrame with fuzzy result columns appended.
    """
    if "has_detection" in df.columns:
        df_active = df[df["has_detection"] == 1].copy()
    else:
        df_active = df.copy()

    fuzzy_rows = [
        evaluate_fuzzy(row.to_dict(), config=config)
        for _, row in df_active.iterrows()
    ]
    fuzzy_df = pd.DataFrame(fuzzy_rows)
    return pd.concat([df_active.reset_index(drop=True), fuzzy_df], axis=1)
