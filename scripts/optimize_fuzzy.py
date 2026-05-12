"""Script entrypoint for GA-Fuzzy optimization."""

from aerosweep.fuzzy_ga.ga_optimizer import optimize_fuzzy


if __name__ == "__main__":
    # Replace with real args or argparse when implemented.
    optimize_fuzzy(data_path="data/tabular", config_path="configs/fuzzy.yaml")
