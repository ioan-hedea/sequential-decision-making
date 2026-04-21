from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ENV_ORDER = [
    "Tiger",
    "HallwaySearch",
    "RockSample(4,3)",
    "MachineMaintenance",
    "MedicalDiagnosis",
    "InventoryControl",
    "DrivingMerge",
    "DrivingMergeNoisy",
    "RockSample(5,4)",
]

SOLVER_ORDER = [
    "ExactValueIteration",
    "PBVI",
    "POMCP",
    "DESPOT",
    "AdaOPS",
    "BAS",
    "BASDeep",
    "BASDeepRollout",
    "BASRootOnly",
    "BASRolloutOnly",
    "BASPlain",
    "SARSOPJulia",
]
SOLVER_COLORS = {
    "ExactValueIteration": "#1b9e77",
    "PBVI": "#7570b3",
    "POMCP": "#d95f02",
    "DESPOT": "#1f78b4",
    "AdaOPS": "#e7298a",
    "BAS": "#66a61e",
    "BASDeep": "#238b45",
    "BASDeepRollout": "#006d2c",
    "BASRootOnly": "#a6d854",
    "BASRolloutOnly": "#4daf4a",
    "BASPlain": "#b8e186",
    "SARSOPJulia": "#a6761d",
}
SOLVER_MARKERS = {
    "ExactValueIteration": "s",
    "PBVI": "o",
    "POMCP": "^",
    "DESPOT": "D",
    "AdaOPS": "P",
    "BAS": "X",
    "BASDeep": ">",
    "BASDeepRollout": "<",
    "BASRootOnly": "v",
    "BASRolloutOnly": "*",
    "BASPlain": "h",
    "SARSOPJulia": "X",
}


def _style_key(solver_name: str) -> str:
    if solver_name == "BASDeepRolloutLearned":
        return "BASDeepRollout"
    return solver_name


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _error_interval(std: float, episodes: int, error_bar: str) -> float:
    if error_bar == "std" or episodes <= 1:
        return std
    return 1.96 * std / math.sqrt(episodes)


def plot_scaling(rows: list[dict[str, str]], output_path: Path, error_bar: str) -> None:
    grouped: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        if row["status"] != "ok":
            continue
        grouped[row["env"]][row["solver"]].append(
            {
                "budget": float(row["belief_budget"]),
                "return_mean": float(row["discounted_return_mean"]),
                "return_std": float(row["discounted_return_std"]),
                "step_mean_ms": float(row["step_time_mean_ms"]),
                "step_std_ms": float(row["step_time_std_ms"]),
                "episode_mean_sec": float(row.get("episode_compute_mean_sec", "0") or 0.0),
                "episode_std_sec": float(row.get("episode_compute_std_sec", "0") or 0.0),
                "episodes": float(row.get("episodes", "1") or 1.0),
            }
        )

    env_names = [env for env in ENV_ORDER if env in grouped] + [
        env for env in sorted(grouped.keys()) if env not in ENV_ORDER
    ]
    if not env_names:
        raise ValueError("No successful rows found in the CSV.")

    n_envs = len(env_names)
    n_cols = 3 if n_envs > 4 else 2
    n_rows = math.ceil(n_envs / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15.2, 8.8), squeeze=False)
    fig.patch.set_facecolor("white")
    err_label = "95% CI" if error_bar == "ci95" else "Std. Dev."

    ordered_solvers = [solver for solver in SOLVER_ORDER if any(solver in grouped[env] for env in env_names)]
    ordered_solvers.extend(
        solver
        for solver in sorted({solver for env in env_names for solver in grouped[env].keys()})
        if solver not in ordered_solvers
    )

    for index, env_name in enumerate(env_names):
        ax = axes[index // n_cols][index % n_cols]
        ax.set_facecolor("#fcfcfc")

        for solver_name in ordered_solvers:
            points = grouped[env_name].get(solver_name)
            if not points:
                continue
            points = sorted(points, key=lambda p: p["budget"])
            xs = [p["budget"] for p in points]
            episodes = [int(p["episodes"]) for p in points]
            ret_mean = [p["return_mean"] for p in points]
            ret_std = [_error_interval(p["return_std"], n, error_bar) for p, n in zip(points, episodes)]
            ax.errorbar(
                xs,
                ret_mean,
                yerr=ret_std,
                marker=SOLVER_MARKERS.get(_style_key(solver_name), "o"),
                color=SOLVER_COLORS.get(_style_key(solver_name), None),
                linewidth=2.1,
                markersize=6.5,
                capsize=3.5,
                elinewidth=1.1,
                label=solver_name,
            )

        ax.set_title(env_name, fontsize=12, pad=8)
        ax.set_xlabel("Belief Budget", fontsize=10)
        ax.set_ylabel("Discounted Return", fontsize=10)
        ax.set_xticks([64, 128, 256, 512])
        ax.tick_params(axis="both", labelsize=9)
        ax.grid(alpha=0.28, linewidth=0.8)
        ax.axhline(0.0, color="#999999", linewidth=0.8, linestyle=":")
        ax.text(
            0.03,
            0.97,
            err_label,
            transform=ax.transAxes,
            fontsize=8.5,
            ha="left",
            va="top",
            color="#555555",
        )

    total_axes = n_rows * n_cols
    for index in range(n_envs, total_axes):
        axes[index // n_cols][index % n_cols].axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=SOLVER_COLORS.get(_style_key(solver), "#333333"),
            marker=SOLVER_MARKERS.get(_style_key(solver), "o"),
            linewidth=2.1,
            markersize=6.5,
            label=solver,
        )
        for solver in ordered_solvers
    ]
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper center",
        ncol=min(5, len(legend_handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
        fontsize=10,
    )
    fig.suptitle("Return Scaling Across Belief Budgets", fontsize=15, y=1.06)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=240, bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot performance vs belief budget from benchmark_summary.csv")
    parser.add_argument("--csv", type=Path, required=True, help="Path to benchmark_summary.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pomdp_benchmarks") / "results" / "scaling_curve.png",
        help="Output image path",
    )
    parser.add_argument(
        "--error-bar",
        choices=("std", "ci95"),
        default="std",
        help="Uncertainty shown on the plot.",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    plot_scaling(rows, args.out, args.error_bar)
    print(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
