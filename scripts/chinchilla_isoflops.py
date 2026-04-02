from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from dataclasses import dataclass
from math import exp, log
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import linregress


DEFAULT_DATA_PATH = Path("data/isoflops_curves.json")
DEFAULT_OUTPUT_DIR = Path("artifacts/chinchilla_isoflops")
DEFAULT_TARGET_BUDGETS = (1e23, 1e24)


@dataclass(frozen=True)
class IsoFLOPsRun:
    parameters: int
    compute_budget: float
    final_loss: float


@dataclass(frozen=True)
class PowerLawFit:
    coefficient: float
    exponent: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Skeleton for fitting Chinchilla-style IsoFLOPs scaling laws."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to data/isoflops_curves.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for plots and any saved outputs.",
    )
    parser.add_argument(
        "--target-budget",
        type=float,
        nargs="+",
        default=list(DEFAULT_TARGET_BUDGETS),
        help="Compute budgets to evaluate after fitting the scaling laws.",
    )
    return parser.parse_args()


def load_runs(data_path: Path) -> list[IsoFLOPsRun]:
    with data_path.open() as f:
        raw_runs = json.load(f)

    return [
        IsoFLOPsRun(
            parameters=int(run["parameters"]),
            compute_budget=float(run["compute_budget"]),
            final_loss=float(run["final_loss"]),
        )
        for run in raw_runs
    ]


def group_runs_by_budget(
    runs: Iterable[IsoFLOPsRun],
) -> dict[float, list[IsoFLOPsRun]]:
    """Return all runs grouped by compute budget."""
    # group each run into a dictionary keyed by compute_budget.
    # Suggested shape:
    # {
    #     6e18: [IsoFLOPsRun(...), IsoFLOPsRun(...), ...],
    #     1e19: [...],
    # }
    budget_groups = dict[float, list[IsoFLOPsRun]]()
    for run in runs:
        compute_budget = run.compute_budget
        if not compute_budget in budget_groups:
            budget_groups[compute_budget] = list()
        budget_groups[compute_budget].append(run)
    return budget_groups


def select_optimal_runs(
    runs_by_budget: dict[float, list[IsoFLOPsRun]],
) -> list[IsoFLOPsRun]:
    """Pick the minimum-loss run from each IsoFLOPs profile."""
    return [
        min(runs_by_budget[compute_budget], key=lambda run: run.final_loss)
        for compute_budget in sorted(runs_by_budget)
    ]


def extract_optimal_model_points(
    optimal_runs: Iterable[IsoFLOPsRun],
) -> list[tuple[float, float]]:
    """Convert optimal runs into (compute_budget, optimal_parameters) pairs."""
    return [(run.compute_budget, run.parameters) for run in optimal_runs]


def extract_optimal_dataset_points(
    optimal_runs: Iterable[IsoFLOPsRun],
) -> list[tuple[float, float]]:
    """Convert optimal runs into (compute_budget, optimal_dataset_size) pairs."""
    # use D_opt(C_i) = C_i / (6 * N_opt(C_i)).
    return [
        (run.compute_budget, run.compute_budget / (6 * run.parameters))
        for run in optimal_runs
    ]


def fit_power_law(points: Iterable[tuple[float, float]]) -> PowerLawFit:
    """Fit y = k * x^a and return (k, a)."""
    log_x = []
    log_y = []

    for x_value, y_value in points:
        log_x.append(log(x_value))
        log_y.append(log(y_value))

    result = linregress(log_x, log_y)
    return PowerLawFit(coefficient=exp(result.intercept), exponent=result.slope)


def predict_with_power_law(fit: PowerLawFit, x_value: float) -> float:
    return fit.coefficient * (x_value ** fit.exponent)


def plot_scaling_law(
    points: Iterable[tuple[float, float]],
    fit: PowerLawFit,
    target_budgets: Iterable[float],
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    """Save a plot of the observed points and fitted scaling law."""
    point_list = sorted(points)
    x_values = [x_value for x_value, _ in point_list]
    y_values = [y_value for _, y_value in point_list]

    predicted_points = sorted(
        (budget, predict_with_power_law(fit, budget)) for budget in target_budgets
    )
    fit_domain = x_values + [budget for budget, _ in predicted_points]
    log_domain_start = log(min(fit_domain))
    log_domain_end = log(max(fit_domain))
    fit_x_values = [
        exp(log_domain_start + (log_domain_end - log_domain_start) * step / 199)
        for step in range(200)
    ]
    fitted_y_values = [predict_with_power_law(fit, x_value) for x_value in fit_x_values]

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.scatter(x_values, y_values, label="Observed optimum", color="tab:blue")
    axis.plot(fit_x_values, fitted_y_values, label="Power-law fit", color="tab:orange")

    if predicted_points:
        target_x_values = [x_value for x_value, _ in predicted_points]
        target_y_values = [y_value for _, y_value in predicted_points]
        axis.scatter(
            target_x_values,
            target_y_values,
            label="Target-budget prediction",
            color="tab:red",
            marker="x",
            s=80,
        )
        for x_value, y_value in predicted_points:
            axis.annotate(
                f"{x_value:.0e}",
                xy=(x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_title(title)
    axis.set_xlabel("Compute budget (FLOPs)")
    axis.set_ylabel(y_label)
    axis.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
    axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def format_predictions(
    label: str,
    fit: PowerLawFit,
    target_budgets: Iterable[float],
) -> str:
    lines = [f"{label}: y = {fit.coefficient:.6g} * C^{fit.exponent:.6f}"]
    for budget in target_budgets:
        prediction = predict_with_power_law(fit, budget)
        lines.append(f"  C = {budget:.3e} -> predicted {label.lower()} = {prediction:.6g}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.data_path)
    runs_by_budget = group_runs_by_budget(runs)
    optimal_runs = select_optimal_runs(runs_by_budget)

    model_points = extract_optimal_model_points(optimal_runs)
    dataset_points = extract_optimal_dataset_points(optimal_runs)

    model_fit = fit_power_law(model_points)
    dataset_fit = fit_power_law(dataset_points)

    plot_scaling_law(
        points=model_points,
        fit=model_fit,
        target_budgets=args.target_budget,
        output_path=args.output_dir / "optimal_model_size_vs_compute.png",
        title="Compute-Optimal Model Size vs Compute Budget",
        y_label="Optimal model size (parameters)",
    )
    plot_scaling_law(
        points=dataset_points,
        fit=dataset_fit,
        target_budgets=args.target_budget,
        output_path=args.output_dir / "optimal_dataset_size_vs_compute.png",
        title="Compute-Optimal Dataset Size vs Compute Budget",
        y_label="Optimal dataset size (tokens)",
    )

    print(format_predictions("Model size", model_fit, args.target_budget))
    print()
    print(format_predictions("Dataset size", dataset_fit, args.target_budget))


if __name__ == "__main__":
    main()
