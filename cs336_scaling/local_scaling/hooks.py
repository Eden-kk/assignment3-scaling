from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cs336_scaling.model import BasicsTransformerLM

from .config import ExperimentConfig


def estimate_run_flops(config: ExperimentConfig) -> float | None:
    """Return a simple FLOPs estimate for bookkeeping and sweep filtering."""
    if config.model.target_params is None or config.data.train_tokens is None:
        return None
    return 6.0 * float(config.model.target_params) * float(config.data.train_tokens)


def build_model(config: ExperimentConfig) -> torch.nn.Module:
    """Create the model for a single run.

    TODO: Replace this stub with your real model construction logic.
    A common pattern is to map `config.model` into `BasicsTransformerLM`.
    """
    config.model = BasicsTransformerLM()


def build_dataloaders(config: ExperimentConfig) -> tuple[Any, Any]:
    """Create the train and validation dataloaders.

    TODO: Replace this stub with your real dataset/tokenizer/dataloader code.
    """
    raise NotImplementedError("Implement build_dataloaders() in local_scaling/hooks.py")


def build_optimizer(
    model: torch.nn.Module, config: ExperimentConfig
) -> torch.optim.Optimizer:
    """Create the optimizer for a single run.

    TODO: Replace this stub with your optimizer construction code.
    """
    raise NotImplementedError("Implement build_optimizer() in local_scaling/hooks.py")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: ExperimentConfig
) -> Any:
    """Create the scheduler for a single run.

    TODO: Replace this stub with your WSD scheduler construction code.
    """
    raise NotImplementedError("Implement build_scheduler() in local_scaling/hooks.py")


def run_training_loop(
    *,
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
    config: ExperimentConfig,
    run_dir: Path,
    wandb_run: Any | None,
) -> dict[str, Any]:
    """Train one model and return run metrics.

    Expected return shape:
    {
        "final_loss": float,
        "best_val_loss": float,
        "train_steps": int,
        ...
    }

    TODO: Replace this stub with your full local training loop.
    """
    raise NotImplementedError("Implement run_training_loop() in local_scaling/hooks.py")


def fit_scaling_law(records_path: str | Path) -> dict[str, Any]:
    """Fit your local scaling law from saved records.

    TODO: Replace this stub with your actual `L(N, D)` fitting code.
    """
    raise NotImplementedError("Implement fit_scaling_law() in local_scaling/hooks.py")
