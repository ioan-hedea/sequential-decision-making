from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .core import (
    bayes_update,
    belief_from_particles,
    entropy,
    normalize,
    observation_probability,
    sample_categorical,
    TabularPOMDP,
)


class UnsupportedSolverError(RuntimeError):
    pass


class BaseSolver(ABC):
    name: str = "base"
    supports_belief_budget: bool = True

    @abstractmethod
    def reset(
        self,
        env: TabularPOMDP,
        rng: np.random.Generator,
        belief_budget: int,
        exact_belief: np.ndarray,
    ) -> None:
        pass

    @abstractmethod
    def act(self, exact_belief: np.ndarray) -> int:
        pass

    @abstractmethod
    def observe(self, action: int, observation: int, exact_belief: np.ndarray) -> None:
        pass

    def particle_belief(self) -> np.ndarray | None:
        return None


class ExactValueIterationSolver(BaseSolver):
    name = "ExactValueIteration"
    supports_belief_budget = False

    def __init__(
        self,
        horizon: int = 5,
        max_alpha_vectors: int = 192,
        max_states: int = 24,
        max_backup_combinations: int = 1200,
        prune_samples: int = 200,
    ) -> None:
        self.horizon = horizon
        self.max_alpha_vectors = max_alpha_vectors
        self.max_states = max_states
        self.max_backup_combinations = max_backup_combinations
        self.prune_samples = prune_samples

        self.env: TabularPOMDP | None = None
        self.rng: np.random.Generator | None = None
        self.alphas: list[np.ndarray] = []
        self.alpha_actions: list[int] = []
        self._planned_signature: tuple | None = None

    def reset(
        self,
        env: TabularPOMDP,
        rng: np.random.Generator,
        belief_budget: int,
        exact_belief: np.ndarray,
    ) -> None:
        del belief_budget, exact_belief
        if env.n_states > self.max_states:
            raise UnsupportedSolverError(
                f"{self.name} skips {env.name}: {env.n_states} states exceeds max {self.max_states}."
            )

        self.env = env
        self.rng = rng
        signature = (env.name, env.n_states, env.n_actions, env.n_obs, self.horizon)
        if self._planned_signature != signature:
            self.alphas, self.alpha_actions = self._plan_alpha_vectors(env)
            self._planned_signature = signature

    def _plan_alpha_vectors(self, env: TabularPOMDP) -> tuple[list[np.ndarray], list[int]]:
        assert self.rng is not None
        alphas: list[np.ndarray] = [np.zeros(env.n_states, dtype=float)]
        actions: list[int] = [0]

        planning_horizon = min(self.horizon, env.horizon)

        for _ in range(planning_horizon):
            candidates: list[np.ndarray] = []
            candidate_actions: list[int] = []
            n_prev = len(alphas)

            for action in range(env.n_actions):
                n_combos = n_prev ** env.n_obs
                if n_combos <= self.max_backup_combinations:
                    combos: Iterable[tuple[int, ...]] = product(range(n_prev), repeat=env.n_obs)
                else:
                    combos = (
                        tuple(int(self.rng.integers(0, n_prev)) for _ in range(env.n_obs))
                        for _ in range(self.max_backup_combinations)
                    )

                for combo in combos:
                    alpha = env.R[:, action].copy()
                    for obs, alpha_idx in enumerate(combo):
                        trans_obs = env.T[action] * env.O[action, :, obs][np.newaxis, :]
                        alpha += env.gamma * (trans_obs @ alphas[alpha_idx])
                    candidates.append(alpha)
                    candidate_actions.append(action)

            alphas, actions = self._prune(candidates, candidate_actions, env)

        return alphas, actions

    def _prune(
        self,
        alphas: list[np.ndarray],
        actions: list[int],
        env: TabularPOMDP,
    ) -> tuple[list[np.ndarray], list[int]]:
        if len(alphas) <= self.max_alpha_vectors:
            return alphas, actions

        assert self.rng is not None

        # Belief samples for approximate dominance checks.
        belief_samples = [env.initial_belief()]
        for _ in range(self.prune_samples):
            belief_samples.append(self.rng.dirichlet(np.ones(env.n_states)))

        selected: set[int] = set()
        for belief in belief_samples:
            values = [float(belief @ alpha) for alpha in alphas]
            selected.add(int(np.argmax(values)))

        # Keep the best alpha per action to avoid losing action coverage.
        for action in range(env.n_actions):
            idx_for_action = [i for i, a in enumerate(actions) if a == action]
            if not idx_for_action:
                continue
            best = max(idx_for_action, key=lambda i: float(env.initial_belief() @ alphas[i]))
            selected.add(best)

        selected_list = list(selected)
        if len(selected_list) > self.max_alpha_vectors:
            scores = []
            for i in selected_list:
                avg_score = float(np.mean([belief @ alphas[i] for belief in belief_samples]))
                scores.append((avg_score, i))
            scores.sort(reverse=True)
            selected_list = [i for _, i in scores[: self.max_alpha_vectors]]

        return [alphas[i] for i in selected_list], [actions[i] for i in selected_list]

    def act(self, exact_belief: np.ndarray) -> int:
        if not self.alphas:
            return 0
        values = [float(exact_belief @ alpha) for alpha in self.alphas]
        idx = int(np.argmax(values))
        return int(self.alpha_actions[idx])

    def observe(self, action: int, observation: int, exact_belief: np.ndarray) -> None:
        del action, observation, exact_belief
        return


class PBVISolver(BaseSolver):
    name = "PBVI"

    def __init__(
        self,
        n_iterations: int = 10,
        default_belief_points: int = 96,
        max_alpha_vectors: int = 256,
    ) -> None:
        self.n_iterations = n_iterations
        self.default_belief_points = default_belief_points
        self.max_alpha_vectors = max_alpha_vectors

        self.env: TabularPOMDP | None = None
        self.rng: np.random.Generator | None = None
        self.alphas: list[np.ndarray] = []
        self.alpha_actions: list[int] = []
        self.belief_points: list[np.ndarray] = []
        self._planned_signature: tuple | None = None

    def reset(
        self,
        env: TabularPOMDP,
        rng: np.random.Generator,
        belief_budget: int,
        exact_belief: np.ndarray,
    ) -> None:
        del exact_belief
        self.env = env
        self.rng = rng

        n_beliefs = int(belief_budget or self.default_belief_points)
        signature = (env.name, env.n_states, n_beliefs, self.n_iterations)
        if self._planned_signature != signature:
            self.belief_points = self._collect_beliefs(env, n_beliefs)
            self.alphas, self.alpha_actions = self._run_pbvi(env, self.belief_points)
            self._planned_signature = signature

    def _collect_beliefs(self, env: TabularPOMDP, n_beliefs: int) -> list[np.ndarray]:
        assert self.rng is not None
        beliefs = [env.initial_belief()]

        while len(beliefs) < n_beliefs:
            belief = env.initial_belief()
            n_roll_steps = int(self.rng.integers(1, env.horizon + 1))
            for _ in range(n_roll_steps):
                action = int(self.rng.integers(0, env.n_actions))
                obs_probs = np.array(
                    [observation_probability(env, belief, action, obs) for obs in range(env.n_obs)]
                )
                observation = sample_categorical(obs_probs, self.rng)
                belief = bayes_update(env, belief, action, observation)
            beliefs.append(belief)

        return beliefs

    def _run_pbvi(
        self,
        env: TabularPOMDP,
        belief_points: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[int]]:
        # Initialization: greedy immediate-reward alpha per belief.
        alphas: list[np.ndarray] = []
        alpha_actions: list[int] = []

        for belief in belief_points:
            action = int(np.argmax([belief @ env.R[:, a] for a in range(env.n_actions)]))
            alphas.append(env.R[:, action].copy())
            alpha_actions.append(action)

        for _ in range(self.n_iterations):
            candidate_alphas: list[np.ndarray] = []
            candidate_actions: list[int] = []

            for belief in belief_points:
                best_value = -math.inf
                best_alpha = None
                best_action = 0

                for action in range(env.n_actions):
                    alpha = env.R[:, action].copy()
                    for obs in range(env.n_obs):
                        obs_prob = observation_probability(env, belief, action, obs)
                        if obs_prob <= 1e-12:
                            continue
                        next_belief = bayes_update(env, belief, action, obs)
                        best_idx = int(np.argmax([next_belief @ a for a in alphas]))
                        trans_obs = env.T[action] * env.O[action, :, obs][np.newaxis, :]
                        alpha += env.gamma * (trans_obs @ alphas[best_idx])

                    value = float(belief @ alpha)
                    if value > best_value:
                        best_value = value
                        best_alpha = alpha
                        best_action = action

                assert best_alpha is not None
                candidate_alphas.append(best_alpha)
                candidate_actions.append(best_action)

            alphas, alpha_actions = self._prune(candidate_alphas, candidate_actions, belief_points)

        return alphas, alpha_actions

    def _prune(
        self,
        alphas: list[np.ndarray],
        actions: list[int],
        belief_points: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[int]]:
        if len(alphas) <= self.max_alpha_vectors:
            return alphas, actions

        selected: set[int] = set()
        for belief in belief_points:
            values = [float(belief @ alpha) for alpha in alphas]
            selected.add(int(np.argmax(values)))

        selected_list = list(selected)
        if len(selected_list) > self.max_alpha_vectors:
            scores = []
            for i in selected_list:
                avg_score = float(np.mean([belief @ alphas[i] for belief in belief_points]))
                scores.append((avg_score, i))
            scores.sort(reverse=True)
            selected_list = [i for _, i in scores[: self.max_alpha_vectors]]

        return [alphas[i] for i in selected_list], [actions[i] for i in selected_list]

    def act(self, exact_belief: np.ndarray) -> int:
        if not self.alphas:
            return 0
        values = [float(exact_belief @ alpha) for alpha in self.alphas]
        idx = int(np.argmax(values))
        return int(self.alpha_actions[idx])

    def observe(self, action: int, observation: int, exact_belief: np.ndarray) -> None:
        del action, observation, exact_belief
        return


class _TreeNode:
    def __init__(self, n_actions: int) -> None:
        self.visit_count = 0
        self.action_counts = np.zeros(n_actions, dtype=np.int64)
        self.action_values = np.zeros(n_actions, dtype=float)
        self.children: dict[tuple[int, int], _TreeNode] = {}
        self.particles: list[int] = []


class POMCPSolver(BaseSolver):
    name = "POMCP"

    def __init__(
        self,
        n_sims: int = 250,
        max_depth: int = 16,
        rollout_depth: int = 10,
        ucb_c: float = 10.0,
        default_particles: int = 512,
    ) -> None:
        self.n_sims = n_sims
        self.max_depth = max_depth
        self.rollout_depth = rollout_depth
        self.ucb_c = float(ucb_c)
        self.default_particles = default_particles

        self.env: TabularPOMDP | None = None
        self.rng: np.random.Generator | None = None
        self.root: _TreeNode | None = None
        self.particles: list[int] = []
        self.n_particles = default_particles

    def reset(
        self,
        env: TabularPOMDP,
        rng: np.random.Generator,
        belief_budget: int,
        exact_belief: np.ndarray,
    ) -> None:
        self.env = env
        self.rng = rng
        self.n_particles = int(belief_budget or self.default_particles)

        self.particles = self._sample_particles(exact_belief, self.n_particles)
        self.root = _TreeNode(env.n_actions)
        self.root.particles = list(self.particles)

    def _sample_particles(self, belief: np.ndarray, n_particles: int) -> list[int]:
        assert self.rng is not None
        dist = normalize(belief)
        choices = self.rng.choice(len(dist), size=n_particles, replace=True, p=dist)
        return [int(x) for x in choices]

    def _simulation_budget(self, exact_belief: np.ndarray) -> int:
        del exact_belief
        return self.n_sims

    def act(self, exact_belief: np.ndarray) -> int:
        assert self.env is not None
        assert self.rng is not None
        assert self.root is not None

        n_sims = max(1, self._simulation_budget(exact_belief))

        for _ in range(n_sims):
            if self.particles:
                state = int(self.rng.choice(self.particles))
            else:
                state = self.env.sample_initial_state(self.rng)
            self._simulate(state, depth=0, node=self.root)

        return int(np.argmax(self.root.action_values))

    def _simulate(self, state: int, depth: int, node: _TreeNode) -> float:
        assert self.env is not None
        assert self.rng is not None

        if depth >= self.max_depth or self.env.is_terminal(state):
            return 0.0

        action = self._ucb_action(node)
        next_state, observation, reward, done = self.env.step(state, action, self.rng)

        key = (action, observation)
        if done:
            future = 0.0
            child = node.children.get(key)
            if child is None:
                child = _TreeNode(self.env.n_actions)
                node.children[key] = child
        else:
            child = node.children.get(key)
            if child is None:
                child = _TreeNode(self.env.n_actions)
                node.children[key] = child
                future = self._rollout(next_state, depth + 1)
            else:
                future = self._simulate(next_state, depth + 1, child)

        child.particles.append(next_state)

        ret = reward + self.env.gamma * future

        node.visit_count += 1
        node.action_counts[action] += 1
        n = node.action_counts[action]
        node.action_values[action] += (ret - node.action_values[action]) / n

        return ret

    def _rollout(self, state: int, depth: int) -> float:
        assert self.env is not None
        assert self.rng is not None

        total = 0.0
        disc = 1.0

        for _ in range(self.rollout_depth):
            if depth >= self.max_depth or self.env.is_terminal(state):
                break
            action = int(self.rng.integers(0, self.env.n_actions))
            state, _, reward, done = self.env.step(state, action, self.rng)
            total += disc * reward
            disc *= self.env.gamma
            depth += 1
            if done:
                break
        return total

    def _ucb_action(self, node: _TreeNode) -> int:
        assert self.rng is not None
        unvisited = np.where(node.action_counts == 0)[0]
        if len(unvisited) > 0:
            return int(self.rng.choice(unvisited))

        log_n = math.log(max(1, node.visit_count))
        exploration = self.ucb_c * np.sqrt(log_n / node.action_counts)
        scores = node.action_values + exploration
        return int(np.argmax(scores))

    def observe(self, action: int, observation: int, exact_belief: np.ndarray) -> None:
        assert self.env is not None
        assert self.rng is not None
        assert self.root is not None

        new_particles: list[int] = []
        attempts = 0
        max_attempts = 200 * self.n_particles

        while len(new_particles) < self.n_particles and attempts < max_attempts:
            if self.particles:
                state = int(self.rng.choice(self.particles))
            else:
                state = self.env.sample_initial_state(self.rng)

            next_state, obs_sim, _, _ = self.env.step(state, action, self.rng)
            if obs_sim == observation:
                new_particles.append(next_state)
            attempts += 1

        if len(new_particles) == 0:
            new_particles = self._sample_particles(exact_belief, self.n_particles)
        elif len(new_particles) < self.n_particles:
            deficit = self.n_particles - len(new_particles)
            extra = self.rng.choice(new_particles, size=deficit, replace=True)
            new_particles.extend(int(x) for x in extra)

        self.particles = new_particles

        key = (action, observation)
        if key in self.root.children:
            self.root = self.root.children[key]
        else:
            self.root = _TreeNode(self.env.n_actions)
        self.root.particles = list(self.particles)

    def particle_belief(self) -> np.ndarray | None:
        assert self.env is not None
        return belief_from_particles(self.env.n_states, self.particles)


class DESPOTSolver(BaseSolver):
    name = "DESPOT"

    def __init__(
        self,
        n_scenarios: int = 48,
        max_depth: int = 6,
        default_particles: int = 512,
    ) -> None:
        self.n_scenarios = n_scenarios
        self.max_depth = max_depth
        self.default_particles = default_particles

        self.env: TabularPOMDP | None = None
        self.rng: np.random.Generator | None = None
        self.particles: list[int] = []
        self.n_particles = default_particles

    def reset(
        self,
        env: TabularPOMDP,
        rng: np.random.Generator,
        belief_budget: int,
        exact_belief: np.ndarray,
    ) -> None:
        self.env = env
        self.rng = rng
        self.n_particles = int(belief_budget or self.default_particles)
        self.particles = self._sample_particles(exact_belief, self.n_particles)

    def _sample_particles(self, belief: np.ndarray, n_particles: int) -> list[int]:
        assert self.rng is not None
        dist = normalize(belief)
        return [int(x) for x in self.rng.choice(len(dist), size=n_particles, replace=True, p=dist)]

    def act(self, exact_belief: np.ndarray) -> int:
        assert self.env is not None
        assert self.rng is not None

        if not self.particles:
            self.particles = self._sample_particles(exact_belief, self.n_particles)

        scenarios = [
            int(self.rng.choice(self.particles)) for _ in range(max(1, self.n_scenarios))
        ]

        q_values = [self._estimate_q_open_loop(scenarios, first_action=action) for action in range(self.env.n_actions)]
        return int(np.argmax(q_values))

    def _rollout_policy(self, state: int) -> int:
        """Heuristic policy for continuation after root action."""
        assert self.env is not None
        assert self.rng is not None
        greedy = int(np.argmax(self.env.R[state, :]))
        # Keep some exploration to avoid brittle determinization.
        if float(self.rng.random()) < 0.2:
            return int(self.rng.integers(0, self.env.n_actions))
        return greedy

    def _estimate_q_open_loop(self, states: list[int], first_action: int) -> float:
        """DESPOT-style sparse-scenario open-loop evaluation."""
        assert self.env is not None
        assert self.rng is not None

        if not states:
            return 0.0

        total_return = 0.0
        for state in states:
            disc = 1.0
            ret = 0.0
            current = state

            for depth in range(self.max_depth):
                if depth == 0:
                    action = first_action
                else:
                    action = self._rollout_policy(current)

                current, _, reward, done = self.env.step(current, action, self.rng)
                ret += disc * reward
                if done:
                    break
                disc *= self.env.gamma

            total_return += ret

        return total_return / len(states)

    def observe(self, action: int, observation: int, exact_belief: np.ndarray) -> None:
        assert self.env is not None
        assert self.rng is not None

        new_particles: list[int] = []
        attempts = 0
        max_attempts = 200 * self.n_particles

        while len(new_particles) < self.n_particles and attempts < max_attempts:
            if self.particles:
                state = int(self.rng.choice(self.particles))
            else:
                state = self.env.sample_initial_state(self.rng)
            next_state, obs_sim, _, _ = self.env.step(state, action, self.rng)
            if obs_sim == observation:
                new_particles.append(next_state)
            attempts += 1

        if len(new_particles) == 0:
            new_particles = self._sample_particles(exact_belief, self.n_particles)
        elif len(new_particles) < self.n_particles:
            deficit = self.n_particles - len(new_particles)
            extra = self.rng.choice(new_particles, size=deficit, replace=True)
            new_particles.extend(int(x) for x in extra)

        self.particles = new_particles

    def particle_belief(self) -> np.ndarray | None:
        assert self.env is not None
        return belief_from_particles(self.env.n_states, self.particles)


class JuliaSARSOPSolver(BaseSolver):
    """
    Optional bridge to Julia + SARSOP.jl.

    This solver exports the current tabular model to JSON, calls a Julia bridge
    script that computes an alpha-vector policy with SARSOP, and imports
    alpha-vectors back into Python for fast online action selection.
    """

    name = "SARSOPJulia"
    supports_belief_budget = False

    def __init__(
        self,
        julia_bin: str = "julia",
        timeout_sec: float = 120.0,
        precision: float = 1e-3,
        bridge_script: str | None = None,
    ) -> None:
        self.julia_bin = julia_bin
        self.timeout_sec = float(timeout_sec)
        self.precision = float(precision)
        default_script = Path(__file__).resolve().parent / "julia" / "sarsop_bridge.jl"
        self.bridge_script = Path(bridge_script) if bridge_script else default_script

        self.alphas: np.ndarray | None = None
        self.actions: np.ndarray | None = None
        self._planned_signature: tuple | None = None

    def _resolve_julia_executable(self) -> str | None:
        """
        Resolve a usable Julia executable.

        If `self.julia_bin` points to a juliaup launcher, prefer a concrete
        installed Julia binary under ~/.julia/juliaup/julia-*/bin/julia to
        avoid launcher lock/config issues.
        """
        resolved = shutil.which(self.julia_bin)
        if resolved is None:
            return None

        resolved_path = Path(resolved)
        if ".juliaup/bin" in str(resolved_path):
            installs_root = Path.home() / ".julia" / "juliaup"
            candidates = [p for p in installs_root.glob("julia-*/bin/julia") if p.exists()]
            if candidates:
                candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return str(candidates[0])

        return str(resolved_path)

    def reset(
        self,
        env: TabularPOMDP,
        rng: np.random.Generator,
        belief_budget: int,
        exact_belief: np.ndarray,
    ) -> None:
        del rng, belief_budget, exact_belief

        julia_exec = self._resolve_julia_executable()
        if julia_exec is None:
            raise UnsupportedSolverError(
                f"{self.name} unavailable: Julia executable '{self.julia_bin}' not found in PATH."
            )
        if not self.bridge_script.exists():
            raise UnsupportedSolverError(
                f"{self.name} unavailable: bridge script not found at {self.bridge_script}."
            )

        signature = (env.name, env.n_states, env.n_actions, env.n_obs, env.horizon)
        if self._planned_signature == signature and self.alphas is not None and self.actions is not None:
            return

        payload = {
            "name": env.name,
            "gamma": float(env.gamma),
            "transition": env.T.tolist(),
            "observation": env.O.tolist(),
            "reward": env.R.tolist(),
            "initial_belief": env.initial_belief().tolist(),
        }

        with tempfile.TemporaryDirectory(prefix="sarsop_bridge_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_path = tmp_path / "model.json"
            out_path = tmp_path / "policy.json"

            model_path.write_text(json.dumps(payload), encoding="utf-8")

            cmd = [
                julia_exec,
                "--startup-file=no",
                str(self.bridge_script),
                "--model-json",
                str(model_path),
                "--output-json",
                str(out_path),
                "--timeout-sec",
                str(self.timeout_sec),
                "--precision",
                str(self.precision),
            ]

            timeout = max(10.0, self.timeout_sec + 20.0)

            def _run_julia(env_override: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
                run_env = os.environ.copy()
                if env_override:
                    run_env.update(env_override)
                return subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                    env=run_env,
                    timeout=timeout,
                )

            try:
                proc = _run_julia()
            except subprocess.TimeoutExpired as exc:
                raise UnsupportedSolverError(
                    f"{self.name} failed: Julia bridge timed out after {exc.timeout:.1f}s."
                ) from exc

            # Fallback for restricted environments where juliaup cannot lock ~/.julia.
            if proc.returncode != 0 and "Could not create lockfile" in ((proc.stderr or "") + (proc.stdout or "")):
                juliaup_src = Path.home() / ".julia" / "juliaup"
                juliaup_cfg = tmp_path / "juliaup-config"
                if juliaup_src.exists():
                    shutil.copytree(juliaup_src, juliaup_cfg, dirs_exist_ok=True)
                    try:
                        proc = _run_julia({"JULIAUP_CONFIG_DIR": str(juliaup_cfg)})
                    except subprocess.TimeoutExpired as exc:
                        raise UnsupportedSolverError(
                            f"{self.name} failed: Julia bridge timed out after {exc.timeout:.1f}s."
                        ) from exc

            if proc.returncode != 0:
                stderr = (proc.stderr or "").strip()
                stdout = (proc.stdout or "").strip()
                detail = stderr if stderr else stdout
                hint = ""
                lower_detail = detail.lower()
                if "could not create lockfile" in lower_detail:
                    hint = (
                        " Hint: juliaup launcher cannot write ~/.julia in this environment. "
                        "Run with a real Julia binary via --julia-bin, or run outside restricted sandbox."
                    )
                elif "failed to determine the command for the `release` channel" in lower_detail:
                    hint = (
                        " Hint: juliaup release channel looks broken. Try: "
                        "`juliaup add release && juliaup default release`."
                    )
                raise UnsupportedSolverError(
                    f"{self.name} failed (exit {proc.returncode}). {detail}{hint}"
                )

            if not out_path.exists():
                raise UnsupportedSolverError(
                    f"{self.name} failed: Julia bridge did not produce {out_path.name}."
                )

            result = json.loads(out_path.read_text(encoding="utf-8"))

        if "error" in result:
            raise UnsupportedSolverError(f"{self.name} failed: {result['error']}")

        alphas = np.asarray(result.get("alphas", []), dtype=float)
        actions = np.asarray(result.get("actions", []), dtype=int)

        if alphas.ndim != 2 or alphas.shape[1] != env.n_states:
            raise UnsupportedSolverError(
                f"{self.name} failed: malformed alpha-vectors from Julia bridge."
            )
        if len(actions) != alphas.shape[0]:
            raise UnsupportedSolverError(
                f"{self.name} failed: action map length mismatch from Julia bridge."
            )
        if np.any(actions < 0) or np.any(actions >= env.n_actions):
            raise UnsupportedSolverError(
                f"{self.name} failed: action indices out of range in Julia output."
            )

        self.alphas = alphas
        self.actions = actions
        self._planned_signature = signature

    def act(self, exact_belief: np.ndarray) -> int:
        if self.alphas is None or self.actions is None or len(self.actions) == 0:
            return 0
        values = self.alphas @ exact_belief
        idx = int(np.argmax(values))
        return int(self.actions[idx])

    def observe(self, action: int, observation: int, exact_belief: np.ndarray) -> None:
        del action, observation, exact_belief
        return


class AdaOPSSolver(POMCPSolver):
    name = "AdaOPS"

    def __init__(
        self,
        base_n_sims: int = 250,
        min_n_sims: int = 100,
        max_n_sims: int = 1500,
        entropy_scale: float = 0.8,
        **kwargs,
    ) -> None:
        super().__init__(n_sims=base_n_sims, **kwargs)
        self.base_n_sims = base_n_sims
        self.min_n_sims = min_n_sims
        self.max_n_sims = max_n_sims
        self.entropy_scale = entropy_scale

    def _simulation_budget(self, exact_belief: np.ndarray) -> int:
        belief_entropy = entropy(exact_belief)
        adaptive = int(self.base_n_sims * (1.0 + self.entropy_scale * belief_entropy))
        return int(np.clip(adaptive, self.min_n_sims, self.max_n_sims))


@dataclass(frozen=True)
class SolverSpec:
    name: str
    factory: Callable[[], BaseSolver]


def make_solver_suite(
    include_adaops: bool = False,
    include_sarsop_julia: bool = False,
    julia_bin: str = "julia",
    sarsop_timeout_sec: float = 120.0,
    sarsop_precision: float = 1e-3,
) -> list[SolverSpec]:
    specs = [
        SolverSpec("ExactValueIteration", lambda: ExactValueIterationSolver()),
        SolverSpec("PBVI", lambda: PBVISolver()),
        SolverSpec("POMCP", lambda: POMCPSolver()),
        SolverSpec("DESPOT", lambda: DESPOTSolver()),
    ]

    if include_adaops:
        specs.append(SolverSpec("AdaOPS", lambda: AdaOPSSolver()))

    if include_sarsop_julia:
        specs.append(
            SolverSpec(
                "SARSOPJulia",
                lambda: JuliaSARSOPSolver(
                    julia_bin=julia_bin,
                    timeout_sec=sarsop_timeout_sec,
                    precision=sarsop_precision,
                ),
            )
        )

    return specs
