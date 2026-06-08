"""Genetic algorithm for fuzzy parameter tuning."""

import copy
import random
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np

from aerosweep.fuzzy_ga.fuzzy_logic import evaluate_fuzzy, DEFAULT_CONFIG


def get_expert_ground_truth(row: pd.Series) -> float:
    """Expert logic to define the ground truth risk score based on features."""
    if row.get("has_detection", 0) == 0:
        return 0.0

    density = float(row.get("area_density_pct", 0.0))
    count = float(row.get("jumlah_instance", 0.0))
    confidence = float(row.get("confidence_score", 0.0))
    category = str(row.get("kategori_dominan", "none"))
    
    # Base logic
    if density > 40 or count >= 10:
        score = 90.0  # Critical base
    elif density > 20 or count >= 5:
        score = 70.0  # High base
    elif density > 5 or count >= 2:
        score = 45.0  # Medium base
    else:
        score = 20.0  # Low base

    # Adjustments
    if confidence > 0.8:
        score += 5.0
    elif confidence < 0.4:
        score -= 10.0

    if category in ["Asbestos", "Hazardous"]:
        score += 15.0
    elif category in ["Vehicles"]:
        score += 10.0

    return max(0.0, min(100.0, score))


def config_to_chromosome(config: Dict[str, Any]) -> np.ndarray:
    """Flatten config membership functions into a 1D array."""
    genes = []
    variables = config["variables"]
    for var_name in ["area_density_pct", "jumlah_instance", "confidence_score"]:
        for mf_name in ["low", "medium", "high", "few", "many"]:
            if mf_name in variables[var_name]:
                genes.extend(variables[var_name][mf_name])
    return np.array(genes, dtype=float)


def chromosome_to_config(chromosome: np.ndarray, base_config: Dict[str, Any]) -> Dict[str, Any]:
    """Decode 1D array back into a config dictionary. Enforces constraints (sorting & clipping)."""
    new_config = copy.deepcopy(base_config)
    variables = new_config["variables"]
    idx = 0
    
    domains = {
        "area_density_pct": (0.0, 100.0),
        "jumlah_instance": (0.0, 100.0),
        "confidence_score": (0.0, 1.0)
    }

    for var_name in ["area_density_pct", "jumlah_instance", "confidence_score"]:
        min_val, max_val = domains[var_name]
        for mf_name in ["low", "medium", "high", "few", "many"]:
            if mf_name in variables[var_name]:
                length = len(variables[var_name][mf_name])
                genes = chromosome[idx:idx+length]
                # Clip to domain
                genes = np.clip(genes, min_val, max_val)
                # Sort to ensure a <= b <= c <= d
                genes = np.sort(genes)
                variables[var_name][mf_name] = genes.tolist()
                idx += length
                
    return new_config


def evaluate_fitness(chromosome: np.ndarray, df: pd.DataFrame, y_true: np.ndarray, base_config: Dict[str, Any]) -> float:
    """Calculate fitness (inverse of Mean Squared Error)."""
    config = chromosome_to_config(chromosome, base_config)
    
    y_pred = []
    for _, row in df.iterrows():
        inputs = {
            "area_density_pct": row.get("area_density_pct", 0.0),
            "jumlah_instance": row.get("jumlah_instance", 0),
            "confidence_score": row.get("confidence_score", 0.0),
            "kategori_dominan": row.get("kategori_dominan", "none")
        }
        res = evaluate_fuzzy(inputs, config)
        y_pred.append(res["fuzzy_score"])
        
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred) ** 2)
    # Fitness is inverse of MSE (add small epsilon to avoid div by zero)
    return 1.0 / (mse + 1e-6)


def evaluate_metrics(config: Dict[str, Any], df: pd.DataFrame) -> Dict[str, float]:
    """Calculate evaluation metrics: MSE, MAE against expert ground truth."""
    df_active = df[df["has_detection"] == 1].copy()
    if len(df_active) == 0:
        return {"MSE": 0.0, "MAE": 0.0}
        
    y_true = np.array([get_expert_ground_truth(row) for _, row in df_active.iterrows()])
    
    y_pred = []
    for _, row in df_active.iterrows():
        inputs = {
            "area_density_pct": row.get("area_density_pct", 0.0),
            "jumlah_instance": row.get("jumlah_instance", 0),
            "confidence_score": row.get("confidence_score", 0.0),
            "kategori_dominan": row.get("kategori_dominan", "none")
        }
        res = evaluate_fuzzy(inputs, config)
        y_pred.append(res["fuzzy_score"])
        
    y_pred = np.array(y_pred)
    
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        "MSE": round(float(mse), 2),
        "MAE": round(float(mae), 2)
    }


def optimize_fuzzy(data_path: str, config_path: str | None = None, pop_size: int = 10, generations: int = 5, progress_callback=None) -> Dict[str, Any]:
    """Run Genetic Algorithm to optimize fuzzy parameters."""
    df = pd.read_csv(data_path)
    # To speed up, we can evaluate only a subset or only positive detections, 
    # but let's evaluate on all items with detection to fine-tune the active boundaries
    df_active = df[df["has_detection"] == 1].copy()
    
    if len(df_active) == 0:
        return DEFAULT_CONFIG
        
    # Cap size for speed in UI
    if len(df_active) > 200:
        df_active = df_active.sample(200, random_state=42)

    # 1. Generate Ground Truth
    y_true = np.array([get_expert_ground_truth(row) for _, row in df_active.iterrows()])
    
    # 2. Initialize Population
    base_chromosome = config_to_chromosome(DEFAULT_CONFIG)
    population = []
    for _ in range(pop_size):
        # Mutate default by some random noise
        noise = np.random.normal(0, 5.0, size=len(base_chromosome))
        # Confidence needs smaller noise
        noise[-11:] = np.random.normal(0, 0.1, size=11) 
        mutated = base_chromosome + noise
        population.append(mutated)
        
    population[0] = base_chromosome # Keep elitism of default
    
    best_config = DEFAULT_CONFIG
    best_fitness = 0
    
    # 3. Evolution Loop
    for gen in range(generations):
        # Evaluate
        fitnesses = [evaluate_fitness(ind, df_active, y_true, DEFAULT_CONFIG) for ind in population]
        
        # Track Best
        max_fit_idx = np.argmax(fitnesses)
        if fitnesses[max_fit_idx] > best_fitness:
            best_fitness = fitnesses[max_fit_idx]
            best_config = chromosome_to_config(population[max_fit_idx], DEFAULT_CONFIG)
            
        if progress_callback:
            progress_callback(gen + 1, best_fitness)
            
        # Selection (Tournament)
        new_population = [population[max_fit_idx]] # Elitism
        while len(new_population) < pop_size:
            i, j = random.sample(range(pop_size), 2)
            parent1 = population[i] if fitnesses[i] > fitnesses[j] else population[j]
            i, j = random.sample(range(pop_size), 2)
            parent2 = population[i] if fitnesses[i] > fitnesses[j] else population[j]
            
            # Crossover (Uniform)
            mask = np.random.rand(len(base_chromosome)) > 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation (Gaussian)
            if random.random() < 0.3:
                noise = np.random.normal(0, 2.0, size=len(base_chromosome))
                noise[-11:] = np.random.normal(0, 0.05, size=11)
                child += noise
                
            new_population.append(child)
            
        population = new_population

    return best_config
