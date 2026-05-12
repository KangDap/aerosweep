"""Dataset utilities for CV training."""

from typing import Any, Tuple


def load_sample(sample_path: str) -> Tuple[Any, Any]:
    _ = sample_path
    raise NotImplementedError("Load image and label/mask here.")
