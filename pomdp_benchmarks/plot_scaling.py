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
    grouped: dict[str, dict[str, list[tuple[int, float]]]] = defaultdict(lambda: defaultdict(list))

    for row in rows:
        if row["status"] != "ok":
            continue
        budget = int(row["belief_budget"])
        ret = float(row["discounted_return_mean"])
        grouped[row["env"]][row["solver"]].append((budget, ret))

    env_names = sorted(grouped.keys())
    if not env_names:
        raise ValueError("No successful rows found in the CSV.")

    n_envs = len(env_names)
    fig, axes = plt.subplots(n_envs, 1, figsize=(10, 3.5 * n_envs), squeeze=False)

    for ax, env_name in zip(axes[:, 0], env_names):
        for solver_name, points in sorted(grouped[env_name].items()):
            points = sorted(points, key=lambda x: x[0])
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            ax.plot(xs, ys, marker="o", linewidth=2, label=solver_name)

        ax.set_title(f"Scalability Curve: {env_name}")
        ax.set_xlabel("Belief Budget")
        ax.set_ylabel("Discounted Return (mean)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

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
