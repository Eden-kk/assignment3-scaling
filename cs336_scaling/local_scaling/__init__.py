from .config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainingConfig,
    WandbConfig,
)


def run_local_experiment(*args, **kwargs):
    from .runner import run_local_experiment as _run_local_experiment

    return _run_local_experiment(*args, **kwargs)

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "WandbConfig",
    "run_local_experiment",
]
