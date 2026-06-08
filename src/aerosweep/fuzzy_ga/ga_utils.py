"""Utility functions for the Genetic Algorithm fuzzy parameter optimizer.

Provides chromosome encoding/decoding, genetic operators, and constraint
repair so that membership-function points remain monotonically ordered and
within their valid numerical ranges.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Chromosome layout
# ---------------------------------------------------------------------------
# Each chromosome is a flat numpy array.  The layout maps to the *tuneable*
# points of each membership function (MF) and the four risk-output centres.
#
# For a trapezoidal MF [a, b, c, d]:
#   - If a==0 and/or d==max_val they are *boundary-locked* and NOT in the
#     chromosome.  Only the interior points are encoded.
#   - Triangular [a, b, c] has all three points encoded (unless boundary).
#
# The variable order is:
#   area_density_pct  → low, medium, high
#   jumlah_instance   → few, medium, many
#   confidence_score  → low, medium, high
#   risk_centers      → low, medium, high, critical
# ---------------------------------------------------------------------------

# The specification below describes each gene segment.  Each entry is:
#   (variable_name, term_name, n_points_in_MF, locked_indices, range_min, range_max)
# `locked_indices` lists point positions (0-based) that are boundary-locked.

_GENE_SPEC: List[Tuple[str, str, int, List[int], float, float]] = [
    # area_density_pct  (range 0–100)
    ("area_density_pct", "low",    4, [0, 1], 0.0, 100.0),   # [0,0,c,d]  → encode c,d
    ("area_density_pct", "medium", 3, [],     0.0, 100.0),    # [a,b,c]    → encode a,b,c
    ("area_density_pct", "high",   4, [2, 3], 0.0, 100.0),   # [a,b,100,100] → encode a,b

    # jumlah_instance  (range 0–100)
    ("jumlah_instance",  "few",    4, [0, 1], 0.0, 100.0),
    ("jumlah_instance",  "medium", 3, [],     0.0, 100.0),
    ("jumlah_instance",  "many",   4, [2, 3], 0.0, 100.0),

    # confidence_score  (range 0–1)
    ("confidence_score", "low",    4, [0, 1], 0.0, 1.0),
    ("confidence_score", "medium", 3, [],     0.0, 1.0),
    ("confidence_score", "high",   4, [2, 3], 0.0, 1.0),
]

_RISK_CENTER_NAMES = ["low", "medium", "high", "critical"]
_RISK_CENTER_RANGE = (0.0, 100.0)

# Pre-compute the chromosome length
_CHROM_LEN = sum(n - len(locked) for _, _, n, locked, *_ in _GENE_SPEC) + len(_RISK_CENTER_NAMES)


def chromosome_length() -> int:
    """Return the total number of genes in a chromosome."""
    return _CHROM_LEN


# ---------------------------------------------------------------------------
# Encode / Decode
# ---------------------------------------------------------------------------

def encode_config(config: Dict[str, Any]) -> np.ndarray:
    """Convert a fuzzy config dict into a chromosome (1-D numpy array)."""
    genes: List[float] = []
    variables = config["variables"]

    for var_name, term_name, n_pts, locked, *_ in _GENE_SPEC:
        points = list(variables[var_name][term_name])
        for i in range(n_pts):
            if i not in locked:
                genes.append(float(points[i]))

    centers = config["risk_output"]["centers"]
    for name in _RISK_CENTER_NAMES:
        genes.append(float(centers[name]))

    return np.array(genes, dtype=np.float64)


def decode_chromosome(chromosome: np.ndarray, template_config: Dict[str, Any]) -> Dict[str, Any]:
    """Decode a chromosome back into a full fuzzy config dict.

    ``template_config`` provides the structure and any locked boundary values.
    """
    config = copy.deepcopy(template_config)
    variables = config["variables"]
    idx = 0

    for var_name, term_name, n_pts, locked, *_ in _GENE_SPEC:
        original = list(variables[var_name][term_name])
        new_points: List[float] = []
        for i in range(n_pts):
            if i in locked:
                new_points.append(original[i])
            else:
                new_points.append(float(chromosome[idx]))
                idx += 1
        variables[var_name][term_name] = new_points

    centers = config["risk_output"]["centers"]
    for name in _RISK_CENTER_NAMES:
        centers[name] = float(chromosome[idx])
        idx += 1

    return config


# ---------------------------------------------------------------------------
# Constraint repair
# ---------------------------------------------------------------------------

def repair_chromosome(chromosome: np.ndarray, template_config: Dict[str, Any]) -> np.ndarray:
    """Repair a chromosome so all MF points are monotonically ordered and in range.

    Returns a *new* array (does not mutate in-place).
    """
    chrom = chromosome.copy()
    idx = 0

    for _var, _term, n_pts, locked, rng_min, rng_max in _GENE_SPEC:
        original = list(template_config["variables"][_var][_term])
        # Reconstruct full point list
        full: List[float] = []
        gene_indices: List[int] = []
        for i in range(n_pts):
            if i in locked:
                full.append(original[i])
            else:
                full.append(float(chrom[idx]))
                gene_indices.append(idx)
                idx += 1

        # Clamp to range
        full = [max(rng_min, min(rng_max, v)) for v in full]

        # Enforce monotonic non-decreasing
        for i in range(1, len(full)):
            if full[i] < full[i - 1]:
                full[i] = full[i - 1]

        # Write back to chromosome (only unlocked positions)
        gi = 0
        for i in range(n_pts):
            if i not in locked:
                chrom[gene_indices[gi]] = full[i]
                gi += 1

    # Repair risk centres: must be monotonically increasing, in [0, 100]
    center_start = idx
    for i in range(len(_RISK_CENTER_NAMES)):
        chrom[center_start + i] = max(0.0, min(100.0, chrom[center_start + i]))
    for i in range(1, len(_RISK_CENTER_NAMES)):
        if chrom[center_start + i] < chrom[center_start + i - 1]:
            chrom[center_start + i] = chrom[center_start + i - 1]

    return chrom


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def generate_initial_population(
    config: Dict[str, Any],
    pop_size: int,
    noise_ratio: float = 0.20,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Create an initial population centred on *config* with uniform noise.

    Returns shape ``(pop_size, chromosome_length())``.
    """
    if rng is None:
        rng = random.Random()

    base = encode_config(config)
    n_genes = len(base)
    population = np.empty((pop_size, n_genes), dtype=np.float64)

    # First individual is the exact default (elitism seed)
    population[0] = base.copy()

    idx = 0
    ranges: List[Tuple[float, float]] = []
    for _var, _term, n_pts, locked, rng_min, rng_max in _GENE_SPEC:
        for i in range(n_pts):
            if i not in locked:
                ranges.append((rng_min, rng_max))
                idx += 1
    for _ in _RISK_CENTER_NAMES:
        ranges.append(_RISK_CENTER_RANGE)

    for p in range(1, pop_size):
        child = base.copy()
        for g in range(n_genes):
            span = ranges[g][1] - ranges[g][0]
            delta = rng.uniform(-noise_ratio, noise_ratio) * span
            child[g] += delta
        population[p] = repair_chromosome(child, config)

    return population


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def tournament_selection(
    population: np.ndarray,
    fitness_values: np.ndarray,
    tournament_size: int = 3,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Select one individual via tournament selection (lower fitness = better).

    Returns the chromosome of the winner.
    """
    if rng is None:
        rng = random.Random()

    pop_size = len(population)
    candidates = rng.sample(range(pop_size), min(tournament_size, pop_size))
    best = min(candidates, key=lambda i: fitness_values[i])
    return population[best].copy()


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def blx_alpha_crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    alpha: float = 0.5,
    rng: random.Random | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """BLX-α crossover for real-valued chromosomes.

    For each gene, the child value is sampled uniformly from
    ``[min(p1,p2) - α·d, max(p1,p2) + α·d]`` where ``d = |p1 - p2|``.
    """
    if rng is None:
        rng = random.Random()

    n = len(parent1)
    child1 = np.empty(n, dtype=np.float64)
    child2 = np.empty(n, dtype=np.float64)

    for i in range(n):
        lo = min(parent1[i], parent2[i])
        hi = max(parent1[i], parent2[i])
        d = hi - lo
        low_bound = lo - alpha * d
        high_bound = hi + alpha * d
        child1[i] = rng.uniform(low_bound, high_bound)
        child2[i] = rng.uniform(low_bound, high_bound)

    return child1, child2


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------

def gaussian_mutation(
    chromosome: np.ndarray,
    mutation_rate: float = 0.1,
    sigma: float = 0.1,
    rng: random.Random | None = None,
) -> np.ndarray:
    """Apply Gaussian mutation to each gene with probability *mutation_rate*.

    ``sigma`` is relative to each gene's range.
    """
    if rng is None:
        rng = random.Random()

    chrom = chromosome.copy()
    idx = 0
    ranges: List[Tuple[float, float]] = []
    for _var, _term, n_pts, locked, rng_min, rng_max in _GENE_SPEC:
        for i in range(n_pts):
            if i not in locked:
                ranges.append((rng_min, rng_max))
                idx += 1
    for _ in _RISK_CENTER_NAMES:
        ranges.append(_RISK_CENTER_RANGE)

    for g in range(len(chrom)):
        if rng.random() < mutation_rate:
            span = ranges[g][1] - ranges[g][0]
            chrom[g] += rng.gauss(0, sigma * span)

    return chrom


# ---------------------------------------------------------------------------
# Fitness helpers
# ---------------------------------------------------------------------------

RISK_LABEL_ENCODING: Dict[str, float] = {
    "low": 20.0,
    "medium": 50.0,
    "high": 75.0,
    "critical": 95.0,
}


def encode_risk_label(label: str) -> float:
    """Convert a linguistic risk label to its numeric encoding."""
    return RISK_LABEL_ENCODING.get(str(label).strip().lower(), 50.0)
