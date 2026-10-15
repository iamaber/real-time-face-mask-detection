import os
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent
MODEL_PATH_ENV_VAR = "MASK_DETECTOR_MODEL_PATH"


@dataclass
class Config:
    """Single source of truth for training and inference settings."""

    # Data paths from the Kaggle datasets
    data_dir: str = "/kaggle/input/face-mask-12k-images-dataset/Face Mask Dataset"
    test_data_dir: str = "/kaggle/input/face-mask-dataset/data"
    model_save_path: str = str(REPO_ROOT / "models" / "best_mobilenet_mask_detector.pt")

    # Training hyperparameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    epochs: int = 25
    dropout_rate: float = 0.5
    weight_decay: float = 1e-4

    # Early stopping
    patience: int = 7
    min_delta: float = 0.001

    # Model architecture
    intermediate_features: int = 640

    # Reproducibility / loading
    seed: int = 42
    num_workers: int = 4
    pin_memory: bool = torch.cuda.is_available()

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Focal loss parameters
    focal_alpha: float = 1.0
    focal_gamma: float = 2.0


def resolve_model_path() -> str:
    """Path to the checkpoint, overridable via the MASK_DETECTOR_MODEL_PATH env var."""
    return os.environ.get(MODEL_PATH_ENV_VAR, Config().model_save_path)
