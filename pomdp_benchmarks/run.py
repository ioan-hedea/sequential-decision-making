from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .benchmark import BenchmarkConfig, format_results_table, run_benchmark_suite, save_results


def _parse_budgets(value: str) -> tuple[int, ...]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Belief budgets cannot be empty.")
    try:
        budgets = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Belief budgets must be comma-separated integers.") from exc

    if any(b <= 0 for b in budgets):
        raise argparse.ArgumentTypeError("Belief budgets must be positive integers.")
    return budgets


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark POMDP solvers across the tabular benchmark suite.",
    )
    parser.add_argument("--episodes", type=int, default=40, help="Episodes per solver/env setting.")
    parser.add_argument(
        "--belief-budgets",
        type=_parse_budgets,
        default=(64, 128, 256, 512),
        help="Comma-separated belief budgets (particles or PBVI belief points).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Base random seed.")
    parser.add_argument("--include-adaops", action="store_true", help="Include optional AdaOPS solver.")
    parser.add_argument(
        "--include-bas",
        action="store_true",
        help="Include the belief-adaptive compute-allocation solver (BAS).",
    )
    parser.add_argument(
        "--include-bas-standalone",
        action="store_true",
        help="Include a standalone learned BAS policy without online search.",
    )
    parser.add_argument(
        "--bas-ablation",
        choices=("both", "root_only", "rollout_only", "none", "deep", "deep_rollout"),
        default="both",
        help="BAS ablation mode: shallow prior+rollout, root prior only, rollout only, deep prior, deep prior+entropy-shaped rollout, or neither.",
    )
    parser.add_argument(
        "--bas-policy-model",
        choices=("heuristic", "distilled", "learned", "neural"),
        default="heuristic",
        help="Policy prior model for BAS: handcrafted, distilled neural, or learned-from-search checkpoint. `neural` is kept as an alias of `distilled`.",
    )
    parser.add_argument(
        "--bas-model-dir",
        type=str,
        default=None,
        help="Directory containing per-environment learned BAS checkpoints (.npz), used with --bas-policy-model learned.",
    )
    parser.add_argument(
        "--include-sarsop-julia",
        action="store_true",
        help="Include optional Julia bridge solver using SARSOP.jl.",
    )
    parser.add_argument("--julia-bin", type=str, default="julia", help="Julia executable for SARSOP bridge.")
    parser.add_argument(
        "--sarsop-timeout-sec",
        type=float,
        default=120.0,
        help="SARSOP Julia bridge solve timeout in seconds.",
    )
    parser.add_argument(
        "--sarsop-precision",
        type=float,
        default=1e-3,
        help="SARSOP precision epsilon (smaller is more accurate/slower).",
    )
    parser.add_argument("--rocksample-n", type=int, default=4, help="RockSample grid size N.")
    parser.add_argument("--rocksample-k", type=int, default=3, help="RockSample rock count K.")
    parser.add_argument(
        "--include-harder-env",
        action="store_true",
        help="Include harder settings: RockSample(5,4) and DrivingMergeNoisy.",
    )
    parser.add_argument(
        "--include-extra-env",
        action="store_true",
        help="Include additional benchmark worlds: HallwaySearch, MachineMaintenance, and InventoryControl.",
    )
    parser.add_argument(
        "--harder-rocksample-n",
        type=int,
        default=5,
        help="Harder RockSample grid size N (used with --include-harder-env).",
    )
    parser.add_argument(
        "--harder-rocksample-k",
        type=int,
        default=4,
        help="Harder RockSample rock count K (used with --include-harder-env).",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps per episode.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV/JSON results (default: timestamped folder under pomdp_benchmarks/results).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick smoke run: 6 episodes, budgets 64 and 128.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    episodes = args.episodes
    budgets = args.belief_budgets
    if args.quick:
        episodes = 6
        budgets = (64, 128)

    config = BenchmarkConfig(
        episodes=episodes,
        base_seed=args.seed,
        belief_budgets=budgets,
        include_adaops=args.include_adaops,
        include_bas=args.include_bas,
        include_bas_standalone=args.include_bas_standalone,
        bas_ablation=args.bas_ablation,
        bas_policy_model=("distilled" if args.bas_policy_model == "neural" else args.bas_policy_model),
        bas_model_dir=args.bas_model_dir,
        include_sarsop_julia=args.include_sarsop_julia,
        julia_bin=args.julia_bin,
        sarsop_timeout_sec=args.sarsop_timeout_sec,
        sarsop_precision=args.sarsop_precision,
        rocksample_n=args.rocksample_n,
        rocksample_k=args.rocksample_k,
        include_harder_env=args.include_harder_env,
        harder_rocksample_n=args.harder_rocksample_n,
        harder_rocksample_k=args.harder_rocksample_k,
        include_extra_env=args.include_extra_env,
        max_steps_override=args.max_steps,
    )

    results = run_benchmark_suite(config)

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("pomdp_benchmarks") / "results" / stamp

    csv_path, json_path = save_results(results, output_dir)

    print(format_results_table(results))
    print("\nSaved results:")
    print(f"- CSV:  {csv_path}")
    print(f"- JSON: {json_path}")


if __name__ == "__main__":
    main()
