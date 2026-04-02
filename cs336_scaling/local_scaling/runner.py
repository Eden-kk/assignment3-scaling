from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .device import select_device
from .hooks import (
    build_dataloaders,
    build_model,
    build_optimizer,
    build_scheduler,
    estimate_run_flops,
    run_training_loop,
)
from .records import RunRecord, append_run_record


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_init_wandb(config: ExperimentConfig) -> Any | None:
    if not config.wandb.enabled:
        return None

    import wandb

    return wandb.init(
        project=config.wandb.project,
        entity=config.wandb.entity,
        group=config.wandb.group,
        job_type=config.wandb.job_type,
        name=config.wandb.run_name,
        mode=config.wandb.mode,
        tags=config.wandb.tags,
        notes=config.wandb.notes,
        config=config.to_dict(),
    )


def _apply_wandb_overrides(config: ExperimentConfig, wandb_run: Any | None) -> ExperimentConfig:
    if wandb_run is None:
        return config

    override_dict = {}
    for key, value in dict(wandb_run.config).items():
        if "." not in key:
            override_dict[key] = value
            continue
        section, field = key.split(".", 1)
        override_dict.setdefault(section, {})
        override_dict[section][field] = value
    return config.with_overrides(override_dict)


def run_local_experiment(config: ExperimentConfig) -> dict[str, Any]:
    run_dir = config.output_path / config.experiment_name / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = _maybe_init_wandb(config)
    config = _apply_wandb_overrides(config, wandb_run)

    _set_seed(config.training.seed)
    device = select_device(config.training.device_preference)

    model = build_model(config).to(device)
    train_loader, val_loader = build_dataloaders(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    metrics = run_training_loop(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config,
        run_dir=run_dir,
        wandb_run=wandb_run,
    )

    estimated_run_flops = estimate_run_flops(config)
    record = RunRecord(
        experiment_name=config.experiment_name,
        run_name=wandb_run.name if wandb_run is not None else run_dir.name,
        compute_budget_flops=config.compute_budget_flops,
        estimated_run_flops=estimated_run_flops,
        final_loss=metrics.get("final_loss"),
        device=str(device),
        status="completed",
        config=config.to_dict(),
        metrics=metrics,
    )
    append_run_record(config.output_path / "run_records.jsonl", record)

    with (run_dir / "config.json").open("w") as handle:
        json.dump(config.to_dict(), handle, indent=2)
    with (run_dir / "metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)

    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.finish()

    return metrics
