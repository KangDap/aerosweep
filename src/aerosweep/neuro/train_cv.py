"""Train a CV model for trash detection/segmentation."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CV model")
    parser.add_argument("--data", required=True, help="Path to training data")
    parser.add_argument("--config", default=None, help="Path to config file")
    args = parser.parse_args()

    _ = args  # Placeholder to silence unused warnings.
    raise NotImplementedError("Add training loop and model setup.")


if __name__ == "__main__":
    main()
