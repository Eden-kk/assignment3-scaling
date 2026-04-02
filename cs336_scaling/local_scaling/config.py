from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class ModelConfig:
    target_params: Optional[int] = None
    d_model: Optional[int] = None
    num_layers: Optional[int] = None
    num_heads: Optional[int] = None
    d_ff: Optional[int] = None
    vocab_size: int = 50304
    context_length: int = 1024
    attn_pdrop: Optional[float] = None
    residual_pdrop: Optional[float] = None
    mup_enabled: bool = False
    notes: str = ""


@dataclass
class DataConfig:
    train_tokens: Optional[int] = None
    val_tokens: Optional[int] = None
    train_path: Optional[str] = None
    val_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    micro_batch_size: Optional[int] = None
    num_workers: int = 2
    notes: str = ""


@dataclass
class TrainingConfig:
    seed: int = 1337
    max_steps: Optional[int] = None
    batch_size: int = 128
    gradient_accumulation_steps: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    optimizer_name: str = "adamw"
    scheduler_name: str = "wsd"
    warmup_fraction: float = 0.05
    stable_fraction: float = 0.85
    decay_fraction: float = 0.10
    log_every_steps: int = 10
    eval_every_steps: int = 100
    save_every_steps: int = 0
    precision: str = "fp32"
    device_preference: str = "auto"
    compile_model: bool = False


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "cs336-scaling-local"
    entity: Optional[str] = None
    group: Optional[str] = None
    job_type: str = "train"
    run_name: Optional[str] = None
    mode: str = "online"
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    sweep_id: Optional[str] = None


@dataclass
class ExperimentConfig:
    experiment_name: str
    output_dir: str
    compute_budget_flops: Optional[float] = None
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        return cls(
            experiment_name=raw["experiment_name"],
            output_dir=raw["output_dir"],
            compute_budget_flops=raw.get("compute_budget_flops"),
            model=ModelConfig(**raw.get("model", {})),
            data=DataConfig(**raw.get("data", {})),
            training=TrainingConfig(**raw.get("training", {})),
            wandb=WandbConfig(**raw.get("wandb", {})),
            metadata=raw.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open() as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_overrides(self, overrides: dict[str, Any]) -> "ExperimentConfig":
        raw = self.to_dict()
        for section, values in overrides.items():
            if isinstance(values, dict) and section in raw and isinstance(raw[section], dict):
                raw[section].update(values)
            else:
                raw[section] = values
        return ExperimentConfig.from_dict(raw)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)
