from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .core import EpisodeRecord, TabularPOMDP, bayes_update, js_divergence
from .environments import make_standard_environments
from .seeding import stable_seed_offset
from .solvers import BaseSolver, UnsupportedSolverError, make_solver_suite


@dataclass(frozen=True)
class BenchmarkConfig:
    episodes: int = 40
    base_seed: int = 7
    belief_budgets: tuple[int, ...] = (64, 128, 256, 512)
    include_adaops: bool = False
    include_bas: bool = False
    include_bas_standalone: bool = False
    bas_ablation: str = "both"
    bas_policy_model: str = "heuristic"
    bas_model_dir: str | None = None
    include_sarsop_julia: bool = False
    julia_bin: str = "julia"
    sarsop_timeout_sec: float = 120.0
    sarsop_precision: float = 1e-3
    rocksample_n: int = 4
    rocksample_k: int = 3
    include_harder_env: bool = False
    harder_rocksample_n: int = 5
    harder_rocksample_k: int = 4
    include_extra_env: bool = False
    max_steps_override: int | None = None


@dataclass
class AggregateResult:
    env: str
    solver: str
    belief_budget: int
    episodes: int
    discounted_return_mean: float | None
    discounted_return_std: float | None
    step_time_mean_ms: float | None
    step_time_std_ms: float | None
    episode_compute_mean_sec: float | None
    episode_compute_std_sec: float | None
    belief_divergence_mean: float | None
    belief_divergence_std: float | None
    status: str
    notes: str = ""


@dataclass
class SearchTrainingExample:
    env: str
    belief: np.ndarray
    policy_target: np.ndarray
    raw_action_counts: np.ndarray
    root_action_values: np.ndarray
    root_action_gap: float
    belief_entropy: float
    rollout_depth_mean: float
    rollout_depth_std: float
    value_uncertainty_target: float
    planner_value_target: float
    monte_carlo_value_target: float
    value_target: float
    reward: float
    next_belief: np.ndarray
    done: bool


def run_episode(
    env: TabularPOMDP,
    solver: BaseSolver,
    *,
    belief_budget: int,
    seed: int,
    max_steps: int,
) -> EpisodeRecord:
    env_rng = np.random.default_rng(seed)
    solver_rng = np.random.default_rng(seed + 1_000_003)

    exact_belief = env.initial_belief()
    solver.reset(env, solver_rng, belief_budget, exact_belief.copy())

    true_state = env.sample_initial_state(env_rng)

    discounted_return = 0.0
    step_times: list[float] = []
    divergences: list[float] = []

    for t in range(max_steps):
        start = time.perf_counter()
        action = solver.act(exact_belief.copy())
        step_times.append(time.perf_counter() - start)

        true_state, observation, reward, done = env.step(true_state, action, env_rng)
        discounted_return += (env.gamma ** t) * reward

        exact_belief = bayes_update(env, exact_belief, action, observation)
        solver.observe(action, observation, exact_belief.copy())

        particle_belief = solver.particle_belief()
        if particle_belief is not None:
            divergences.append(js_divergence(exact_belief, particle_belief))

        if done:
            break

    return EpisodeRecord(
        discounted_return=float(discounted_return),
        mean_step_time_sec=float(np.mean(step_times) if step_times else 0.0),
        total_compute_sec=float(np.sum(step_times) if step_times else 0.0),
        mean_belief_divergence=(float(np.mean(divergences)) if divergences else None),
        steps=len(step_times),
    )


def collect_search_training_examples(
    env: TabularPOMDP,
    solver: BaseSolver,
    *,
    belief_budget: int,
    seed: int,
    max_steps: int,
) -> list[SearchTrainingExample]:
    env_rng = np.random.default_rng(seed)
    solver_rng = np.random.default_rng(seed + 1_000_003)

    exact_belief = env.initial_belief()
    solver.reset(env, solver_rng, belief_budget, exact_belief.copy())
    true_state = env.sample_initial_state(env_rng)

    raw_examples: list[
        tuple[
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            np.ndarray,
            bool,
        ]
    ] = []

    for _ in range(max_steps):
        current_belief = exact_belief.copy()
        action = solver.act(exact_belief.copy())
        policy_target = solver.latest_search_policy_target()
        auxiliary = solver.latest_search_auxiliary_targets()
        planner_value_target = solver.latest_search_value_target()

        true_state, observation, reward, done = env.step(true_state, action, env_rng)
        next_belief = bayes_update(env, exact_belief, action, observation)
        if policy_target is not None:
            raw_counts = np.asarray(
                auxiliary.get("raw_action_counts", np.zeros(env.n_actions, dtype=float)) if auxiliary else np.zeros(env.n_actions, dtype=float),
                dtype=float,
            )
            if raw_counts.shape != (env.n_actions,):
                raw_counts = np.zeros(env.n_actions, dtype=float)
            root_action_values = np.asarray(
                auxiliary.get("root_action_values", np.zeros(env.n_actions, dtype=float)) if auxiliary else np.zeros(env.n_actions, dtype=float),
                dtype=float,
            )
            if root_action_values.shape != (env.n_actions,):
                root_action_values = np.zeros(env.n_actions, dtype=float)
            raw_examples.append(
                (
                    current_belief,
                    policy_target.copy(),
                    raw_counts.copy(),
                    root_action_values.copy(),
                    float(auxiliary.get("root_action_gap", 0.0) if auxiliary else 0.0),
                    float(auxiliary.get("belief_entropy", 0.0) if auxiliary else 0.0),
                    float(auxiliary.get("rollout_depth_mean", 0.0) if auxiliary else 0.0),
                    float(auxiliary.get("rollout_depth_std", 0.0) if auxiliary else 0.0),
                    float(auxiliary.get("value_uncertainty_target", 0.0) if auxiliary else 0.0),
                    float(planner_value_target if planner_value_target is not None else 0.0),
                    float(reward),
                    next_belief.copy(),
                    bool(done),
                )
            )

        exact_belief = next_belief
        solver.observe(action, observation, exact_belief.copy())

        if done:
            break

    examples: list[SearchTrainingExample] = []
    discounted = 0.0
    value_targets = [0.0] * len(raw_examples)
    for idx in range(len(raw_examples) - 1, -1, -1):
        reward = raw_examples[idx][10]
        discounted = reward + env.gamma * discounted
        value_targets[idx] = float(discounted)

    for idx, (
        belief,
        policy_target,
        raw_action_counts,
        root_action_values,
        root_action_gap,
        belief_entropy,
        rollout_depth_mean,
        rollout_depth_std,
        value_uncertainty_target,
        planner_value_target,
        reward,
        next_belief,
        done,
    ) in enumerate(raw_examples):
        monte_carlo_value_target = value_targets[idx]
        initial_value_target = (
            planner_value_target if np.isfinite(planner_value_target) else monte_carlo_value_target
        )
        examples.append(
            SearchTrainingExample(
                env=env.name,
                belief=belief,
                policy_target=policy_target,
                raw_action_counts=raw_action_counts,
                root_action_values=root_action_values,
                root_action_gap=root_action_gap,
                belief_entropy=belief_entropy,
                rollout_depth_mean=rollout_depth_mean,
                rollout_depth_std=rollout_depth_std,
                value_uncertainty_target=value_uncertainty_target,
                planner_value_target=planner_value_target,
                monte_carlo_value_target=monte_carlo_value_target,
                value_target=initial_value_target,
                reward=reward,
                next_belief=next_belief,
                done=done,
            )
        )
    return examples


def _aggregate_records(
    env_name: str,
    solver_name: str,
    belief_budget: int,
    records: list[EpisodeRecord],
) -> AggregateResult:
    rewards = np.array([r.discounted_return for r in records], dtype=float)
    times_ms = np.array([1000.0 * r.mean_step_time_sec for r in records], dtype=float)
    total_compute_sec = np.array([r.total_compute_sec for r in records], dtype=float)

    divergence_values = [r.mean_belief_divergence for r in records if r.mean_belief_divergence is not None]
    if divergence_values:
        div_arr = np.array(divergence_values, dtype=float)
        div_mean = float(div_arr.mean())
        div_std = float(div_arr.std())
    else:
        div_mean = None
        div_std = None

    return AggregateResult(
        env=env_name,
        solver=solver_name,
        belief_budget=belief_budget,
        episodes=len(records),
        discounted_return_mean=float(rewards.mean()),
        discounted_return_std=float(rewards.std()),
        step_time_mean_ms=float(times_ms.mean()),
        step_time_std_ms=float(times_ms.std()),
        episode_compute_mean_sec=float(total_compute_sec.mean()),
        episode_compute_std_sec=float(total_compute_sec.std()),
        belief_divergence_mean=div_mean,
        belief_divergence_std=div_std,
        status="ok",
    )


def run_benchmark_suite(config: BenchmarkConfig) -> list[AggregateResult]:
    envs = make_standard_environments(
        rocksample_n=config.rocksample_n,
        rocksample_k=config.rocksample_k,
        include_harder_env=config.include_harder_env,
        harder_rocksample_n=config.harder_rocksample_n,
        harder_rocksample_k=config.harder_rocksample_k,
        include_extra_env=config.include_extra_env,
    )
    solver_specs = make_solver_suite(
        include_adaops=config.include_adaops,
        include_bas=config.include_bas,
        include_bas_standalone=config.include_bas_standalone,
        bas_ablation=config.bas_ablation,
        bas_policy_model=config.bas_policy_model,
        bas_model_dir=config.bas_model_dir,
        include_sarsop_julia=config.include_sarsop_julia,
        julia_bin=config.julia_bin,
        sarsop_timeout_sec=config.sarsop_timeout_sec,
        sarsop_precision=config.sarsop_precision,
    )

    results: list[AggregateResult] = []

    for env_name, env in envs.items():
        max_steps = config.max_steps_override or env.horizon

        for spec in solver_specs:
            # Exact alpha-vector VI is budget-independent. Others scale with belief budget.
            solver_template = spec.factory()
            budgets = (
                [config.belief_budgets[0]]
                if not solver_template.supports_belief_budget
                else list(config.belief_budgets)
            )

            for belief_budget in budgets:
                solver = spec.factory()
                episode_records: list[EpisodeRecord] = []

                try:
                    for episode in range(config.episodes):
                        seed = config.base_seed + 10_000 * episode + 97 * belief_budget
                        seed += stable_seed_offset(env_name, spec.name)
                        episode_records.append(
                            run_episode(
                                env,
                                solver,
                                belief_budget=belief_budget,
                                seed=seed,
                                max_steps=max_steps,
                            )
                        )
                except UnsupportedSolverError as exc:
                    results.append(
                        AggregateResult(
                            env=env_name,
                            solver=spec.name,
                            belief_budget=belief_budget,
                            episodes=0,
                            discounted_return_mean=None,
                            discounted_return_std=None,
                            step_time_mean_ms=None,
                            step_time_std_ms=None,
                            episode_compute_mean_sec=None,
                            episode_compute_std_sec=None,
                            belief_divergence_mean=None,
                            belief_divergence_std=None,
                            status="skipped",
                            notes=str(exc),
                        )
                    )
                    continue

                results.append(
                    _aggregate_records(
                        env_name,
                        spec.name,
                        belief_budget,
                        episode_records,
                    )
                )

    return results


def save_results(results: list[AggregateResult], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "benchmark_summary.csv"
    json_path = output_dir / "benchmark_summary.json"

    rows = [asdict(r) for r in results]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "env",
                "solver",
                "belief_budget",
                "episodes",
                "discounted_return_mean",
                "discounted_return_std",
                "step_time_mean_ms",
                "step_time_std_ms",
                "episode_compute_mean_sec",
                "episode_compute_std_sec",
                "belief_divergence_mean",
                "belief_divergence_std",
                "status",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    return csv_path, json_path


def format_results_table(results: list[AggregateResult]) -> str:
    headers = [
        "Environment",
        "Solver",
        "Budget",
        "Return mean+-std",
        "Step ms+-std",
        "Episode sec+-std",
        "Belief div",
        "Status",
    ]

    lines = [
        f"{headers[0]:<20} {headers[1]:<20} {headers[2]:>7} {headers[3]:>24} {headers[4]:>16} {headers[5]:>18} {headers[6]:>12} {headers[7]:>10}"
    ]
    lines.append("-" * len(lines[0]))

    for r in results:
        if r.status != "ok":
            ret_text = "-"
            time_text = "-"
            ep_text = "-"
            div_text = "-"
        else:
            ret_text = f"{r.discounted_return_mean: .2f} +- {r.discounted_return_std:.2f}"
            time_text = f"{r.step_time_mean_ms:.2f} +- {r.step_time_std_ms:.2f}"
            ep_text = f"{r.episode_compute_mean_sec:.3f} +- {r.episode_compute_std_sec:.3f}"
            div_text = "-" if r.belief_divergence_mean is None else f"{r.belief_divergence_mean:.4f}"

        lines.append(
            f"{r.env:<20} {r.solver:<20} {r.belief_budget:>7} {ret_text:>24} {time_text:>16} {ep_text:>18} {div_text:>12} {r.status:>10}"
        )

    skipped = [r for r in results if r.status != "ok"]
    if skipped:
        lines.append("\nSkipped combinations:")
        for r in skipped:
            lines.append(f"- {r.env} / {r.solver} / budget {r.belief_budget}: {r.notes}")

    return "\n".join(lines)
