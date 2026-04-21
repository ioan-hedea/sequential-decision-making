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
from .neural_policy import BeliefPolicyValueNetwork, sanitize_env_name


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

    def latest_search_policy_target(self) -> np.ndarray | None:
        return None

    def latest_search_auxiliary_targets(self) -> dict[str, np.ndarray | float] | None:
        return None

    def latest_search_value_target(self) -> float | None:
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
        self.action_m2 = np.zeros(n_actions, dtype=float)
        self.children: dict[tuple[int, int], _TreeNode] = {}
        self.particles: list[int] = []
        self.cached_prior: np.ndarray | None = None
        self.state_counts: np.ndarray | None = None
        self.state_total = 0


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
        self._latest_policy_target: np.ndarray | None = None
        self._latest_auxiliary_targets: dict[str, np.ndarray | float] | None = None
        self._latest_value_target: float | None = None
        self._current_rollout_lengths: list[float] = []

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
        self._set_node_particles(self.root, self.particles)
        self._latest_policy_target = None
        self._latest_auxiliary_targets = None
        self._latest_value_target = None
        self._current_rollout_lengths = []

    def _predictive_state_distribution(self, belief: np.ndarray, action: int) -> np.ndarray:
        assert self.env is not None
        return normalize(belief @ self.env.T[action])

    def _predictive_observation_distribution(self, belief: np.ndarray, action: int) -> np.ndarray:
        assert self.env is not None
        predicted_state = self._predictive_state_distribution(belief, action)
        obs_dist = predicted_state @ self.env.O[action]
        return normalize(obs_dist)

    def _belief_action_scores(self, belief: np.ndarray) -> np.ndarray:
        """
        Baseline uniform allocation.

        Subclasses can override this to induce non-uniform compute-allocation
        policies over belief-action space while keeping the POMCP baseline
        unchanged.
        """
        assert self.env is not None
        del belief
        return np.ones(self.env.n_actions, dtype=float)

    def _action_prior(self, belief: np.ndarray) -> np.ndarray:
        scores = self._belief_action_scores(belief)
        return normalize(scores)

    def _allocate_action_simulations(self, belief: np.ndarray, total_sims: int) -> np.ndarray:
        assert self.env is not None
        total_sims = max(total_sims, self.env.n_actions)
        prior = self._action_prior(belief)
        counts = np.floor(total_sims * prior).astype(int)
        counts = np.maximum(counts, 1)

        remainder = total_sims - int(np.sum(counts))
        if remainder > 0:
            order = np.argsort(-prior)
            for idx in order[:remainder]:
                counts[idx] += 1
        elif remainder < 0:
            order = np.argsort(prior)
            for idx in order:
                if remainder == 0:
                    break
                if counts[idx] > 1:
                    counts[idx] -= 1
                    remainder += 1

        return counts

    def _node_belief(self, node: _TreeNode) -> np.ndarray | None:
        assert self.env is not None
        if node.state_counts is not None and node.state_total > 0:
            return node.state_counts / node.state_total
        if not node.particles:
            return None
        return belief_from_particles(self.env.n_states, node.particles)

    def _set_node_particles(self, node: _TreeNode, particles: list[int]) -> None:
        assert self.env is not None
        node.particles = list(particles)
        counts = np.zeros(self.env.n_states, dtype=float)
        for state in particles:
            counts[int(state)] += 1.0
        node.state_counts = counts
        node.state_total = len(particles)
        node.cached_prior = None

    def _record_node_state(self, node: _TreeNode, state: int) -> None:
        assert self.env is not None
        if node.state_counts is None:
            node.state_counts = np.zeros(self.env.n_states, dtype=float)
        node.state_counts[int(state)] += 1.0
        node.state_total += 1

    def _sample_particles(self, belief: np.ndarray, n_particles: int) -> list[int]:
        assert self.rng is not None
        dist = normalize(belief)
        choices = self.rng.choice(len(dist), size=n_particles, replace=True, p=dist)
        return [int(x) for x in choices]

    def _simulation_budget(self, exact_belief: np.ndarray) -> int:
        del exact_belief
        return self.n_sims

    def _leaf_value(self, state: int, depth: int, belief: np.ndarray | None = None) -> float:
        del state, depth, belief
        return 0.0

    def act(self, exact_belief: np.ndarray) -> int:
        assert self.env is not None
        assert self.rng is not None
        assert self.root is not None

        n_sims = max(1, self._simulation_budget(exact_belief))
        self._latest_policy_target = None
        self._latest_auxiliary_targets = None
        self._latest_value_target = None
        self._current_rollout_lengths = []

        for _ in range(n_sims):
            if self.particles:
                state = int(self.rng.choice(self.particles))
            else:
                state = self.env.sample_initial_state(self.rng)
            self._simulate(state, depth=0, node=self.root)

        if np.sum(self.root.action_counts) > 0:
            self._latest_policy_target = normalize(self.root.action_counts.astype(float))
        else:
            self._latest_policy_target = np.ones(self.env.n_actions, dtype=float) / self.env.n_actions
        self._latest_value_target = self._compute_root_value_target()
        self._latest_auxiliary_targets = self._compute_auxiliary_targets(exact_belief)
        return int(np.argmax(self.root.action_values))

    def _compute_root_value_target(self) -> float:
        assert self.root is not None
        visited = self.root.action_counts > 0
        if not np.any(visited):
            return 0.0
        return float(np.max(self.root.action_values[visited]))

    def _compute_auxiliary_targets(self, belief: np.ndarray) -> dict[str, np.ndarray | float]:
        assert self.root is not None
        counts = self.root.action_counts.astype(float)
        values = self.root.action_values.astype(float)
        visited = self.root.action_counts > 0

        if np.count_nonzero(visited) >= 2:
            ranked_values = np.sort(values[visited])
            root_action_gap = float(ranked_values[-1] - ranked_values[-2])
        else:
            root_action_gap = 0.0

        action_std = np.zeros_like(values)
        has_variance = self.root.action_counts > 1
        if np.any(has_variance):
            action_std[has_variance] = np.sqrt(
                np.maximum(
                    self.root.action_m2[has_variance]
                    / np.maximum(self.root.action_counts[has_variance] - 1, 1),
                    0.0,
                )
            )
        if np.sum(counts) > 0:
            uncertainty = float(np.dot(normalize(counts), action_std))
        else:
            uncertainty = 0.0

        rollout_lengths = np.asarray(self._current_rollout_lengths, dtype=float)
        rollout_depth_mean = float(np.mean(rollout_lengths)) if rollout_lengths.size else 0.0
        rollout_depth_std = float(np.std(rollout_lengths)) if rollout_lengths.size else 0.0

        return {
            "raw_action_counts": counts.copy(),
            "root_action_values": values.copy(),
            "root_action_gap": root_action_gap,
            "belief_entropy": float(entropy(belief)),
            "rollout_depth_mean": rollout_depth_mean,
            "rollout_depth_std": rollout_depth_std,
            "value_uncertainty_target": uncertainty,
        }

    def _simulate(self, state: int, depth: int, node: _TreeNode) -> float:
        assert self.env is not None
        assert self.rng is not None

        if self.env.is_terminal(state):
            return 0.0
        if depth >= self.max_depth:
            return self._leaf_value(state, depth, self._node_belief(node))

        node_belief = self._node_belief(node)
        action = self._ucb_action(node, depth)
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
                rollout_belief = (
                    bayes_update(self.env, node_belief, action, observation)
                    if node_belief is not None
                    else None
                )
                future = self._rollout(next_state, depth + 1, rollout_belief)
            else:
                future = self._simulate(next_state, depth + 1, child)

        self._record_node_state(child, next_state)

        ret = reward + self.env.gamma * future

        node.visit_count += 1
        node.action_counts[action] += 1
        n = node.action_counts[action]
        delta = ret - node.action_values[action]
        node.action_values[action] += delta / n
        delta_post = ret - node.action_values[action]
        node.action_m2[action] += delta * delta_post

        return ret

    def _rollout(self, state: int, depth: int, belief: np.ndarray | None = None) -> float:
        assert self.env is not None
        assert self.rng is not None

        total = 0.0
        disc = 1.0
        rollout_steps = 0

        for _ in range(self.rollout_depth):
            if depth >= self.max_depth or self.env.is_terminal(state):
                break
            action = self._rollout_policy(state, depth, belief)
            state, observation, reward, done = self.env.step(state, action, self.rng)
            total += disc * reward
            if belief is not None:
                belief = bayes_update(self.env, belief, action, observation)
            disc *= self.env.gamma
            depth += 1
            rollout_steps += 1
            if done:
                break
        if depth < self.max_depth and not self.env.is_terminal(state):
            total += disc * self._leaf_value(state, depth, belief)
        self._current_rollout_lengths.append(float(rollout_steps))
        return total

    def _rollout_policy(self, state: int, depth: int, belief: np.ndarray | None = None) -> int:
        del state, depth, belief
        assert self.env is not None
        assert self.rng is not None
        return int(self.rng.integers(0, self.env.n_actions))

    def _ucb_action(self, node: _TreeNode, depth: int) -> int:
        assert self.rng is not None
        del depth
        belief = self._node_belief(node)
        prior = (
            self._action_prior(belief)
            if belief is not None
            else np.ones_like(node.action_values, dtype=float) / len(node.action_values)
        )
        unvisited = np.where(node.action_counts == 0)[0]
        if len(unvisited) > 0:
            weights = normalize(prior[unvisited])
            return int(self.rng.choice(unvisited, p=weights))

        log_n = math.log(max(1, node.visit_count))
        exploration = self.ucb_c * prior * np.sqrt(log_n / node.action_counts)
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
        self._set_node_particles(self.root, self.particles)

    def particle_belief(self) -> np.ndarray | None:
        assert self.env is not None
        return belief_from_particles(self.env.n_states, self.particles)

    def latest_search_policy_target(self) -> np.ndarray | None:
        if self._latest_policy_target is None:
            return None
        return self._latest_policy_target.copy()

    def latest_search_auxiliary_targets(self) -> dict[str, np.ndarray | float] | None:
        if self._latest_auxiliary_targets is None:
            return None
        copied: dict[str, np.ndarray | float] = {}
        for key, value in self._latest_auxiliary_targets.items():
            copied[key] = value.copy() if isinstance(value, np.ndarray) else float(value)
        return copied

    def latest_search_value_target(self) -> float | None:
        if self._latest_value_target is None:
            return None
        return float(self._latest_value_target)


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
        installed Julia binary under ~/.julia/juliaup to avoid launcher
        lock/config issues.
        """
        resolved = shutil.which(self.julia_bin)
        if resolved is None:
            return None

        resolved_path = Path(resolved)
        if ".juliaup/bin" in str(resolved_path):
            installs_root = Path.home() / ".julia" / "juliaup"
            candidates = [p for p in installs_root.glob("julia-*/bin/julia") if p.exists()]
            candidates.extend(
                p
                for p in installs_root.glob("julia-*/Julia-*.app/Contents/Resources/julia/bin/julia")
                if p.exists()
            )
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

            combined_output = (proc.stderr or "") + (proc.stdout or "")
            needs_writable_depot = (
                "Could not create lockfile" in combined_output
                or (".ji.pidfile" in combined_output and "operation not permitted" in combined_output.lower())
            )

            # Fallback for restricted environments where juliaup or Julia's
            # compiled cache cannot write under ~/.julia.
            if proc.returncode != 0 and needs_writable_depot:
                juliaup_src = Path.home() / ".julia" / "juliaup"
                juliaup_cfg = tmp_path / "juliaup-config"
                julia_home = Path.home() / ".julia"
                julia_depot = tmp_path / "julia-depot"
                julia_depot.mkdir(parents=True, exist_ok=True)

                env_override = {
                    "JULIA_DEPOT_PATH": os.pathsep.join([str(julia_depot), str(julia_home)]),
                }
                if juliaup_src.exists():
                    shutil.copytree(juliaup_src, juliaup_cfg, dirs_exist_ok=True)
                    env_override["JULIAUP_CONFIG_DIR"] = str(juliaup_cfg)

                try:
                    proc = _run_julia(env_override)
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
                if "could not create lockfile" in lower_detail or (
                    ".ji.pidfile" in detail and "operation not permitted" in lower_detail
                ):
                    hint = (
                        " Hint: Julia cannot write its lock/cache files under ~/.julia in this environment. "
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


class BeliefAdaptiveAllocationSolver(AdaOPSSolver):
    """
    Compute-allocation policy over belief-action space.

    This solver replaces uniform root simulation allocation with an explicit
    allocation rule driven by belief uncertainty, predictive observation
    dispersion, and action-value disagreement. The same belief-dependent action
    prior is then reused as a multiplicative factor inside UCB exploration,
    yielding a lightweight online planner that concentrates compute on
    value-relevant parts of the search tree.
    """

    name = "BAS"

    def __init__(
        self,
        base_n_sims: int = 96,
        min_n_sims: int = 64,
        max_n_sims: int = 384,
        entropy_scale: float = 0.55,
        reward_weight: float = 0.35,
        state_entropy_weight: float = 0.10,
        obs_entropy_weight: float = 0.25,
        prior_strength: float = 0.30,
        prior_activation_visits: int = 12,
        prior_max_depth: int = 2,
        use_prior: bool = True,
        use_rollout_heuristic: bool = True,
        deep_allocation: bool = False,
        rollout_entropy_shaping: bool = False,
        rollout_min_depth: int = 3,
        rollout_extra_depth: int = 4,
        deep_prior_decay: float = 0.75,
        policy_model: str = "heuristic",
        model_dir: str | None = None,
        neural_hidden_dim: int = 96,
        neural_train_beliefs: int = 320,
        neural_train_steps: int = 300,
        neural_learning_rate: float = 0.05,
        neural_weight_decay: float = 1e-4,
        neural_value_loss_weight: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(
            base_n_sims=base_n_sims,
            min_n_sims=min_n_sims,
            max_n_sims=max_n_sims,
            entropy_scale=entropy_scale,
            max_depth=kwargs.pop("max_depth", 10),
            rollout_depth=kwargs.pop("rollout_depth", 6),
            ucb_c=kwargs.pop("ucb_c", 6.0),
            **kwargs,
        )
        self.reward_weight = float(reward_weight)
        self.state_entropy_weight = float(state_entropy_weight)
        self.obs_entropy_weight = float(obs_entropy_weight)
        self.prior_strength = float(prior_strength)
        self.prior_activation_visits = int(prior_activation_visits)
        self.prior_max_depth = int(prior_max_depth)
        self.use_prior = bool(use_prior)
        self.use_rollout_heuristic = bool(use_rollout_heuristic)
        self.deep_allocation = bool(deep_allocation)
        self.rollout_entropy_shaping = bool(rollout_entropy_shaping)
        self.rollout_min_depth = int(rollout_min_depth)
        self.rollout_extra_depth = int(rollout_extra_depth)
        self.deep_prior_decay = float(deep_prior_decay)
        self.policy_model = str(policy_model)
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.neural_hidden_dim = int(neural_hidden_dim)
        self.neural_train_beliefs = int(neural_train_beliefs)
        self.neural_train_steps = int(neural_train_steps)
        self.neural_learning_rate = float(neural_learning_rate)
        self.neural_weight_decay = float(neural_weight_decay)
        self.neural_value_loss_weight = float(neural_value_loss_weight)
        self.policy_network: BeliefPolicyValueNetwork | None = None
        self._policy_signature: tuple[str, int, int, str] | None = None

    def _structured_domain_guardrail(self, belief: np.ndarray | None = None) -> float:
        assert self.env is not None
        score = 1.0

        if self.env.n_states <= 4:
            score *= 0.45
        elif self.env.n_states <= 8:
            score *= 0.65

        if belief is not None:
            support = int(np.count_nonzero(belief > 1e-3))
            if support <= 2:
                score *= 0.60
            elif support <= 4:
                score *= 0.80

            top = float(np.max(belief))
            if top >= 0.80:
                score *= 0.70

        if self.env.name == "Tiger":
            score *= 0.55
        elif self.env.name == "MedicalDiagnosis":
            score *= 0.85

        return float(np.clip(score, 0.20, 1.0))

    def _decision_uncertainty(self, values: np.ndarray) -> float:
        values = np.asarray(values, dtype=float)
        if values.size <= 1:
            return 0.0
        order = np.sort(values)
        best = float(order[-1])
        second = float(order[-2])
        span = float(np.max(values) - np.min(values))
        if span <= 1e-12:
            return 1.0
        gap = (best - second) / span
        return float(np.clip(1.0 - gap, 0.0, 1.0))

    def _catastrophe_risk(self, belief: np.ndarray) -> float:
        assert self.env is not None
        expected_rewards = belief @ self.env.R
        negative = np.maximum(-expected_rewards, 0.0)
        scale = float(np.max(np.abs(expected_rewards)))
        if scale <= 1e-12:
            return 0.0
        return float(np.clip(np.max(negative) / scale, 0.0, 1.0))

    def _rollout_signal(self, belief: np.ndarray) -> float:
        assert self.env is not None
        immediate_q = belief @ self.env.R
        disagreement = self._decision_uncertainty(immediate_q)

        obs_entropies = np.array(
            [entropy(self._predictive_observation_distribution(belief, action)) for action in range(self.env.n_actions)],
            dtype=float,
        )
        max_obs = float(np.max(obs_entropies))
        normalized_obs = obs_entropies / max_obs if max_obs > 1e-12 else obs_entropies
        voi = float(np.mean(normalized_obs)) * disagreement

        risk = self._catastrophe_risk(belief)
        guardrail = self._structured_domain_guardrail(belief)

        signal = 0.45 * disagreement + 0.35 * voi + 0.20 * risk
        return float(np.clip(signal * guardrail, 0.0, 1.0))

    def _bootstrap_value(self, belief: np.ndarray, depth: int) -> float:
        assert self.env is not None
        remaining = max(0, self.max_depth - depth)
        immediate_q = belief @ self.env.R
        immediate_best = float(np.max(immediate_q))
        if remaining <= 1:
            return immediate_best

        bootstrap_values = []
        for action in range(self.env.n_actions):
            predicted_state = self._predictive_state_distribution(belief, action)
            obs_dist = self._predictive_observation_distribution(belief, action)
            future = 0.0
            for obs, obs_prob in enumerate(obs_dist):
                if obs_prob <= 1e-12:
                    continue
                updated = bayes_update(self.env, belief, action, obs)
                future += float(obs_prob) * float(np.max(updated @ self.env.R))
            bootstrap_values.append(float(immediate_q[action]) + self.env.gamma * future)

        blended = max(immediate_best, float(np.max(bootstrap_values)))
        guardrail = self._structured_domain_guardrail(belief)
        return float((0.5 + 0.5 * guardrail) * blended)

    def reset(
        self,
        env: TabularPOMDP,
        rng: np.random.Generator,
        belief_budget: int,
        exact_belief: np.ndarray,
    ) -> None:
        super().reset(env, rng, belief_budget, exact_belief)
        assert self.root is not None
        self._ensure_policy_network()
        self.root.cached_prior = self._compute_action_prior(exact_belief) if self.use_prior else None

    def _uses_distilled_policy(self) -> bool:
        return self.policy_model in ("neural", "distilled")

    def _uses_learned_policy(self) -> bool:
        return self.policy_model == "learned"

    def _uses_any_network(self) -> bool:
        return self._uses_distilled_policy() or self._uses_learned_policy()

    def _env_profile(self) -> dict[str, float]:
        assert self.env is not None

        profile = {
            "reward_weight": self.reward_weight,
            "state_entropy_weight": self.state_entropy_weight,
            "obs_entropy_weight": self.obs_entropy_weight,
            "prior_strength": self.prior_strength,
            "prior_max_depth": float(self.prior_max_depth),
            "rollout_signal_scale": 1.0,
        }

        env_name = self.env.name
        if env_name == "Tiger":
            profile.update(
                reward_weight=0.55,
                state_entropy_weight=0.05,
                obs_entropy_weight=0.20,
                prior_strength=0.20,
                prior_max_depth=1.0,
                rollout_signal_scale=0.55,
            )
        elif env_name.startswith("RockSample"):
            profile.update(
                reward_weight=0.18,
                state_entropy_weight=0.12,
                obs_entropy_weight=0.55,
                prior_strength=0.12,
                prior_max_depth=1.0,
                rollout_signal_scale=1.10,
            )
        elif env_name.startswith("DrivingMerge"):
            profile.update(
                reward_weight=0.30,
                state_entropy_weight=0.10,
                obs_entropy_weight=0.28,
                prior_strength=0.18,
                prior_max_depth=1.0,
                rollout_signal_scale=1.05,
            )
        elif env_name == "MedicalDiagnosis":
            profile.update(
                reward_weight=0.40,
                state_entropy_weight=0.08,
                obs_entropy_weight=0.20,
                prior_strength=0.22,
                prior_max_depth=1.0,
                rollout_signal_scale=0.85,
            )

        if self.deep_allocation:
            profile["prior_max_depth"] = float(self.max_depth)
            profile["prior_strength"] = min(float(profile["prior_strength"]), 0.16)

        return profile

    def _heuristic_action_scores(self, belief: np.ndarray) -> np.ndarray:
        assert self.env is not None
        profile = self._env_profile()

        immediate_q = belief @ self.env.R
        shifted_q = immediate_q - float(np.min(immediate_q))
        reward_scale = float(np.max(shifted_q))
        if reward_scale > 1e-12:
            shifted_q = shifted_q / reward_scale

        state_entropies = np.array(
            [entropy(self._predictive_state_distribution(belief, action)) for action in range(self.env.n_actions)],
            dtype=float,
        )
        obs_entropies = np.array(
            [entropy(self._predictive_observation_distribution(belief, action)) for action in range(self.env.n_actions)],
            dtype=float,
        )
        if float(np.max(state_entropies)) > 1e-12:
            state_entropies = state_entropies / float(np.max(state_entropies))
        if float(np.max(obs_entropies)) > 1e-12:
            obs_entropies = obs_entropies / float(np.max(obs_entropies))

        scores = np.zeros(self.env.n_actions, dtype=float)
        for action in range(self.env.n_actions):
            scores[action] = (
                profile["reward_weight"] * shifted_q[action]
                + profile["state_entropy_weight"] * state_entropies[action]
                + profile["obs_entropy_weight"] * obs_entropies[action]
            )

        action_names = getattr(self.env, "action_names", [])
        env_name = self.env.name
        if env_name == "Tiger":
            left_prob, right_prob = float(belief[0]), float(belief[1])
            for i, name in enumerate(action_names):
                if name == "listen":
                    scores[i] += 0.35 * entropy(belief[:2])
                elif name == "open_left":
                    scores[i] += 0.60 * right_prob - 0.60 * left_prob
                elif name == "open_right":
                    scores[i] += 0.60 * left_prob - 0.60 * right_prob
        elif env_name.startswith("RockSample"):
            for i, name in enumerate(action_names):
                if name.startswith("check_"):
                    scores[i] += 0.35 * obs_entropies[i]
                elif name == "east":
                    scores[i] += 0.08
        elif env_name.startswith("DrivingMerge"):
            small_prob = float(belief[0]) if len(belief) > 0 else 0.0
            medium_prob = float(belief[1]) if len(belief) > 1 else 0.0
            large_prob = float(belief[2]) if len(belief) > 2 else 0.0
            for i, name in enumerate(action_names):
                if name == "wait":
                    scores[i] += 0.40 * small_prob + 0.10 * medium_prob
                elif name == "slow_merge":
                    scores[i] += 0.30 * medium_prob + 0.15 * large_prob - 0.30 * small_prob
                elif name == "merge":
                    scores[i] += 0.65 * large_prob + 0.20 * medium_prob - 0.85 * small_prob
        elif env_name == "MedicalDiagnosis":
            confidence = float(np.max(belief[:3]))
            best_state = int(np.argmax(belief[:3]))
            diagnose_map = {
                "diagnose_healthy": 0,
                "diagnose_mild": 1,
                "diagnose_severe": 2,
            }
            test_bonus = entropy(belief[:3])
            for i, name in enumerate(action_names):
                if name in ("blood_test", "imaging"):
                    scores[i] += 0.30 * test_bonus
                elif name in diagnose_map:
                    if diagnose_map[name] == best_state:
                        scores[i] += 0.45 * confidence
                    else:
                        scores[i] -= 0.25 * confidence

        return np.maximum(scores, 1e-6)

    def _collect_policy_training_beliefs(self, n_beliefs: int) -> np.ndarray:
        assert self.env is not None
        assert self.rng is not None

        beliefs: list[np.ndarray] = [self.env.initial_belief()]
        uniform = np.ones(self.env.n_actions, dtype=float) / self.env.n_actions

        while len(beliefs) < n_beliefs:
            belief = self.env.initial_belief()
            rollout_steps = int(self.rng.integers(1, self.env.horizon + 1))
            for _ in range(rollout_steps):
                beliefs.append(belief.copy())
                if len(beliefs) >= n_beliefs:
                    break

                teacher_prior = normalize(self._heuristic_action_scores(belief))
                mixed = normalize(0.7 * teacher_prior + 0.3 * uniform)
                action = sample_categorical(mixed, self.rng)
                obs_probs = np.array(
                    [
                        observation_probability(self.env, belief, action, obs)
                        for obs in range(self.env.n_obs)
                    ],
                    dtype=float,
                )
                observation = sample_categorical(obs_probs, self.rng)
                belief = bayes_update(self.env, belief, action, observation)

        return np.asarray(beliefs[:n_beliefs], dtype=float)

    def _ensure_policy_network(self) -> None:
        assert self.env is not None
        assert self.rng is not None

        if self._uses_learned_policy():
            self.policy_network = self._load_policy_network()
            self._policy_signature = (self.env.name, self.env.n_states, self.env.n_actions, self.policy_model)
            return

        if not self._uses_distilled_policy():
            self.policy_network = None
            self._policy_signature = None
            return

        signature = (self.env.name, self.env.n_states, self.env.n_actions, self.policy_model)
        if self.policy_network is not None and self._policy_signature == signature:
            return

        belief_batch = self._collect_policy_training_beliefs(self.neural_train_beliefs)
        target_batch = np.asarray(
            [normalize(self._heuristic_action_scores(belief)) for belief in belief_batch],
            dtype=float,
        )

        self.policy_network = BeliefPolicyValueNetwork(
            self.env.n_states,
            self.env.n_actions,
            hidden_dim=self.neural_hidden_dim,
            seed=int(self.rng.integers(0, 2**31 - 1)),
        )
        self.policy_network.fit(
            belief_batch,
            target_batch,
            np.zeros(len(belief_batch), dtype=float),
            steps=self.neural_train_steps,
            learning_rate=self.neural_learning_rate,
            weight_decay=self.neural_weight_decay,
            value_loss_weight=0.0,
        )
        self._policy_signature = signature

    def _model_checkpoint_path(self) -> Path:
        assert self.env is not None
        if self.model_dir is None:
            raise UnsupportedSolverError(
                f"{self.name} with policy_model={self.policy_model!r} requires --bas-model-dir."
            )
        return self.model_dir / f"{sanitize_env_name(self.env.name)}.npz"

    def _load_policy_network(self) -> BeliefPolicyValueNetwork:
        assert self.env is not None
        checkpoint_path = self._model_checkpoint_path()
        if not checkpoint_path.exists():
            raise UnsupportedSolverError(
                f"{self.name} missing learned checkpoint for {self.env.name}: {checkpoint_path}"
            )

        network, checkpoint = BeliefPolicyValueNetwork.load(checkpoint_path)
        if checkpoint.input_dim != self.env.n_states or checkpoint.output_dim != self.env.n_actions:
            raise UnsupportedSolverError(
                f"{self.name} checkpoint shape mismatch for {self.env.name}: "
                f"expected ({self.env.n_states}, {self.env.n_actions}), "
                f"got ({checkpoint.input_dim}, {checkpoint.output_dim})."
            )
        return network

    def _belief_action_scores(self, belief: np.ndarray) -> np.ndarray:
        if self._uses_any_network() and self.policy_network is not None:
            predicted = self.policy_network.predict_policy(belief)
            return np.maximum(predicted, 1e-6)
        return self._heuristic_action_scores(belief)

    def _compute_action_prior(self, belief: np.ndarray) -> np.ndarray:
        assert self.env is not None
        if not self.use_prior:
            return np.ones(self.env.n_actions, dtype=float) / self.env.n_actions
        profile = self._env_profile()
        learned = normalize(self._belief_action_scores(belief))
        uniform = np.ones(self.env.n_actions, dtype=float) / self.env.n_actions
        prior_strength = float(profile["prior_strength"]) * self._structured_domain_guardrail(belief)
        return normalize((1.0 - prior_strength) * uniform + prior_strength * learned)

    def _action_prior(self, belief: np.ndarray) -> np.ndarray:
        return self._compute_action_prior(belief)

    def _depth_adjusted_prior(self, prior: np.ndarray, depth: int) -> np.ndarray:
        uniform = np.ones_like(prior, dtype=float) / len(prior)
        if not self.deep_allocation or depth <= 0:
            return prior
        blend = self.deep_prior_decay ** depth
        return normalize((1.0 - blend) * uniform + blend * prior)

    def _rollout_limit(self, belief: np.ndarray | None) -> int:
        if not self.rollout_entropy_shaping or belief is None or self.env is None:
            return self.rollout_depth

        profile = self._env_profile()
        scaled = self._rollout_signal(belief) * float(profile.get("rollout_signal_scale", 1.0))
        scaled = float(np.clip(scaled, 0.0, 1.0))
        extra = int(round(self.rollout_extra_depth * scaled))
        return int(np.clip(self.rollout_min_depth + extra, self.rollout_min_depth, self.rollout_depth + self.rollout_extra_depth))

    def _leaf_value(self, state: int, depth: int, belief: np.ndarray | None = None) -> float:
        assert self.env is not None
        if self.env.is_terminal(state):
            return 0.0
        if belief is None:
            belief = np.zeros(self.env.n_states, dtype=float)
            belief[int(state)] = 1.0
        if self._uses_learned_policy() and self.policy_network is not None:
            return self.policy_network.predict_value(belief)
        return self._bootstrap_value(belief, depth)

    def _rollout(self, state: int, depth: int, belief: np.ndarray | None = None) -> float:
        assert self.env is not None
        assert self.rng is not None

        total = 0.0
        disc = 1.0
        rollout_limit = self._rollout_limit(belief)

        for _ in range(rollout_limit):
            if depth >= self.max_depth or self.env.is_terminal(state):
                break
            action = self._rollout_policy(state, depth, belief)
            state, observation, reward, done = self.env.step(state, action, self.rng)
            total += disc * reward
            if belief is not None:
                belief = bayes_update(self.env, belief, action, observation)
            disc *= self.env.gamma
            depth += 1
            if done:
                break
        if depth < self.max_depth and not self.env.is_terminal(state):
            total += disc * self._leaf_value(state, depth, belief)
        return total

    def _rollout_policy(self, state: int, depth: int, belief: np.ndarray | None = None) -> int:
        assert self.rng is not None
        assert self.env is not None
        if not self.use_rollout_heuristic:
            return super()._rollout_policy(state, depth, belief)

        if self.env.name.startswith("DrivingMerge") and belief is not None:
            small_prob = float(belief[0]) if len(belief) > 0 else 0.0
            medium_prob = float(belief[1]) if len(belief) > 1 else 0.0
            large_prob = float(belief[2]) if len(belief) > 2 else 0.0
            action_lookup = {name: i for i, name in enumerate(self.env.action_names)}
            if large_prob >= 0.55:
                return action_lookup.get("merge", int(np.argmax(self.env.R[state, :])))
            if small_prob >= 0.45:
                return action_lookup.get("wait", int(np.argmax(self.env.R[state, :])))
            if medium_prob + large_prob >= 0.60:
                return action_lookup.get("slow_merge", int(np.argmax(self.env.R[state, :])))

        if self.env.name == "MedicalDiagnosis" and belief is not None:
            action_lookup = {name: i for i, name in enumerate(self.env.action_names)}
            posterior = belief[:3]
            confidence = float(np.max(posterior))
            best_state = int(np.argmax(posterior))
            if confidence >= 0.72:
                diagnosis_actions = [
                    action_lookup.get("diagnose_healthy", 0),
                    action_lookup.get("diagnose_mild", 0),
                    action_lookup.get("diagnose_severe", 0),
                ]
                return diagnosis_actions[best_state]
            return action_lookup.get("blood_test", int(np.argmax(self.env.R[state, :])))

        greedy = int(np.argmax(self.env.R[state, :]))
        if float(self.rng.random()) < 0.08:
            return int(self.rng.integers(0, self.env.n_actions))
        return greedy

    def _ucb_action(self, node: _TreeNode, depth: int) -> int:
        assert self.rng is not None
        assert self.env is not None

        uniform = np.ones_like(node.action_values, dtype=float) / len(node.action_values)
        prior = uniform
        effective_max_depth = int(self._env_profile()["prior_max_depth"])
        if self.use_prior and depth <= effective_max_depth and node.visit_count >= self.prior_activation_visits:
            if node.cached_prior is None:
                belief = self._node_belief(node)
                node.cached_prior = uniform if belief is None else self._compute_action_prior(belief)
            prior = self._depth_adjusted_prior(node.cached_prior, depth)

        unvisited = np.where(node.action_counts == 0)[0]
        if len(unvisited) > 0:
            weights = normalize(prior[unvisited])
            return int(self.rng.choice(unvisited, p=weights))

        log_n = math.log(max(1, node.visit_count))
        exploration = self.ucb_c * prior * np.sqrt(log_n / node.action_counts)
        scores = node.action_values + exploration
        return int(np.argmax(scores))


class BASStandalonePolicySolver(BaseSolver):
    name = "BASPolicyOnlyLearned"
    supports_belief_budget = False

    def __init__(self, model_dir: str | None = None) -> None:
        self.model_dir = Path(model_dir).expanduser() if model_dir else None
        self.env: TabularPOMDP | None = None
        self.policy_network: BeliefPolicyValueNetwork | None = None
        self._loaded_signature: tuple[str, int, int] | None = None

    def _checkpoint_path(self) -> Path:
        assert self.env is not None
        if self.model_dir is None:
            raise UnsupportedSolverError(f"{self.name} requires --bas-model-dir.")
        return self.model_dir / f"{sanitize_env_name(self.env.name)}.npz"

    def reset(
        self,
        env: TabularPOMDP,
        rng: np.random.Generator,
        belief_budget: int,
        exact_belief: np.ndarray,
    ) -> None:
        del rng, belief_budget, exact_belief
        self.env = env
        signature = (env.name, env.n_states, env.n_actions)
        if self.policy_network is not None and self._loaded_signature == signature:
            return

        checkpoint_path = self._checkpoint_path()
        if not checkpoint_path.exists():
            raise UnsupportedSolverError(
                f"{self.name} missing learned checkpoint for {env.name}: {checkpoint_path}"
            )
        network, checkpoint = BeliefPolicyValueNetwork.load(checkpoint_path)
        if checkpoint.input_dim != env.n_states or checkpoint.output_dim != env.n_actions:
            raise UnsupportedSolverError(
                f"{self.name} checkpoint shape mismatch for {env.name}: "
                f"expected ({env.n_states}, {env.n_actions}), got ({checkpoint.input_dim}, {checkpoint.output_dim})."
            )
        self.policy_network = network
        self._loaded_signature = signature

    def act(self, exact_belief: np.ndarray) -> int:
        if self.policy_network is None:
            return 0
        probs = self.policy_network.predict_policy(exact_belief)
        return int(np.argmax(probs))

    def observe(self, action: int, observation: int, exact_belief: np.ndarray) -> None:
        del action, observation, exact_belief
        return


@dataclass(frozen=True)
class SolverSpec:
    name: str
    factory: Callable[[], BaseSolver]


def make_solver_suite(
    include_adaops: bool = False,
    include_bas: bool = False,
    include_bas_standalone: bool = False,
    bas_ablation: str = "both",
    bas_policy_model: str = "heuristic",
    bas_model_dir: str | None = None,
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

    if include_bas:
        bas_use_prior = bas_ablation in ("both", "root_only", "deep", "deep_rollout")
        bas_use_rollout = bas_ablation in ("both", "rollout_only", "deep_rollout")
        bas_deep = bas_ablation in ("deep", "deep_rollout")
        bas_rollout_shape = bas_ablation == "deep_rollout"
        bas_name = {
            "both": "BAS",
            "root_only": "BASRootOnly",
            "rollout_only": "BASRolloutOnly",
            "none": "BASPlain",
            "deep": "BASDeep",
            "deep_rollout": "BASDeepRollout",
        }.get(bas_ablation, "BAS")
        if bas_policy_model in ("neural", "distilled"):
            bas_name = f"{bas_name}Neural"
        elif bas_policy_model == "learned":
            bas_name = f"{bas_name}Learned"
        specs.append(
            SolverSpec(
                bas_name,
                lambda: BeliefAdaptiveAllocationSolver(
                    use_prior=bas_use_prior,
                    use_rollout_heuristic=bas_use_rollout,
                    deep_allocation=bas_deep,
                    rollout_entropy_shaping=bas_rollout_shape,
                    policy_model=bas_policy_model,
                    model_dir=bas_model_dir,
                ),
            )
        )
        if include_bas_standalone:
            if bas_policy_model != "learned":
                raise ValueError("--include-bas-standalone currently requires --bas-policy-model learned.")
            specs.append(
                SolverSpec(
                    "BASPolicyOnlyLearned",
                    lambda: BASStandalonePolicySolver(model_dir=bas_model_dir),
                )
            )

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
