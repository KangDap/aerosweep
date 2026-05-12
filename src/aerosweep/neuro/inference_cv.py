"""Run CV inference for detection/segmentation."""

from typing import Any, Dict


def load_model(weights_path: str) -> Any:
    _ = weights_path
    raise NotImplementedError("Load your trained model here.")


def run_inference(model: Any, image_path: str) -> Dict[str, Any]:
    _ = (model, image_path)
    raise NotImplementedError("Implement inference and post-processing.")
