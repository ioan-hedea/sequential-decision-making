from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_scaling(rows: list[dict[str, str]], output_path: Path) -> None:
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
            }
        )

    env_names = sorted(grouped.keys())
    if not env_names:
        raise ValueError("No successful rows found in the CSV.")

    n_envs = len(env_names)
    fig, axes = plt.subplots(n_envs, 3, figsize=(17, 3.6 * n_envs), squeeze=False)

    for row_i, env_name in enumerate(env_names):
        ax_ret = axes[row_i, 0]
        ax_step = axes[row_i, 1]
        ax_ep = axes[row_i, 2]

        for solver_name, points in sorted(grouped[env_name].items()):
            points = sorted(points, key=lambda p: p["budget"])
            xs = [p["budget"] for p in points]

            ret_mean = [p["return_mean"] for p in points]
            ret_std = [p["return_std"] for p in points]
            ax_ret.errorbar(
                xs,
                ret_mean,
                yerr=ret_std,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=solver_name,
            )

            step_mean = [p["step_mean_ms"] for p in points]
            step_std = [p["step_std_ms"] for p in points]
            ax_step.errorbar(
                xs,
                step_mean,
                yerr=step_std,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=solver_name,
            )

            ep_mean = [p["episode_mean_sec"] for p in points]
            ep_std = [p["episode_std_sec"] for p in points]
            ax_ep.errorbar(
                xs,
                ep_mean,
                yerr=ep_std,
                marker="o",
                linewidth=1.8,
                capsize=3,
                label=solver_name,
            )

        ax_ret.set_title(f"{env_name}: Return (mean +- std)")
        ax_ret.set_xlabel("Belief Budget")
        ax_ret.set_ylabel("Discounted Return")
        ax_ret.grid(alpha=0.3)

        ax_step.set_title(f"{env_name}: Step Time (mean +- std)")
        ax_step.set_xlabel("Belief Budget")
        ax_step.set_ylabel("ms / step")
        ax_step.grid(alpha=0.3)

        ax_ep.set_title(f"{env_name}: Episode Compute (mean +- std)")
        ax_ep.set_xlabel("Belief Budget")
        ax_ep.set_ylabel("sec / episode")
        ax_ep.grid(alpha=0.3)

        # One legend per row keeps the figure readable.
        ax_ep.legend(fontsize=8, loc="best")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=160)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot performance vs belief budget from benchmark_summary.csv")
    parser.add_argument("--csv", type=Path, required=True, help="Path to benchmark_summary.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pomdp_benchmarks") / "results" / "scaling_curve.png",
        help="Output image path",
    )
    args = parser.parse_args()

    rows = load_rows(args.csv)
    plot_scaling(rows, args.out)
    print(f"Saved plot: {args.out}")


if __name__ == "__main__":
    main()
