from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def normalize(dist: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return a normalized copy of dist with numerical safeguards."""
    clipped = np.clip(np.asarray(dist, dtype=float), eps, None)
    total = clipped.sum()
    if total <= 0.0:
        return np.ones_like(clipped) / len(clipped)
    return clipped / total


def sample_categorical(probabilities: np.ndarray, rng: np.random.Generator) -> int:
    probs = normalize(probabilities)
    return int(rng.choice(len(probs), p=probs))


def entropy(dist: np.ndarray, eps: float = 1e-12) -> float:
    p = normalize(dist, eps)
    return float(-np.sum(p * np.log(p + eps)))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """Jensen-Shannon divergence between two discrete distributions."""
    p_n = normalize(p, eps)
    q_n = normalize(q, eps)
    m = 0.5 * (p_n + q_n)

    kl_pm = np.sum(p_n * (np.log(p_n + eps) - np.log(m + eps)))
    kl_qm = np.sum(q_n * (np.log(q_n + eps) - np.log(m + eps)))
    return float(0.5 * (kl_pm + kl_qm))


@dataclass(frozen=True)
class EpisodeRecord:
    discounted_return: float
    mean_step_time_sec: float
    total_compute_sec: float
    mean_belief_divergence: float | None
    steps: int


class TabularPOMDP:
    """
    Generic tabular POMDP model with discrete state/action/observation spaces.

    Arrays:
      T[a, s, s']      = P(s' | s, a)
      O[a, s', o]      = P(o | s', a)
      R[s, a]          = E[r | s, a] (used by tabular planners)
      R_sas[a, s, s']  = sampled reward for transitions (optional)
    """

    def __init__(
        self,
        *,
        name: str,
        gamma: float,
        transition: np.ndarray,
        observation: np.ndarray,
        reward: np.ndarray,
        initial_belief: np.ndarray,
        action_names: list[str],
        observation_names: list[str],
        state_names: list[str],
        terminal_states: Iterable[int] = (),
        reward_tensor: np.ndarray | None = None,
        horizon: int = 15,
    ) -> None:
        self.name = name
        self.gamma = float(gamma)
        self.T = np.asarray(transition, dtype=float)
        self.O = np.asarray(observation, dtype=float)
        self.R = np.asarray(reward, dtype=float)
        self._initial_belief = normalize(np.asarray(initial_belief, dtype=float))
        self.action_names = list(action_names)
        self.observation_names = list(observation_names)
        self.state_names = list(state_names)
        self.horizon = int(horizon)

        self.n_actions, self.n_states, n_states_t = self.T.shape
        if n_states_t != self.n_states:
            raise ValueError("Transition matrix must be square in state dims.")
        if self.O.shape != (self.n_actions, self.n_states, len(self.observation_names)):
            raise ValueError("Observation tensor shape mismatch.")
        if self.R.shape != (self.n_states, self.n_actions):
            raise ValueError("Reward matrix shape mismatch.")

        self.n_obs = self.O.shape[2]
        self.terminal_mask = np.zeros(self.n_states, dtype=bool)
        for s in terminal_states:
            self.terminal_mask[int(s)] = True

        if reward_tensor is not None:
            reward_tensor = np.asarray(reward_tensor, dtype=float)
            if reward_tensor.shape != (self.n_actions, self.n_states, self.n_states):
                raise ValueError("Reward tensor shape must be [A,S,S].")
        self.reward_tensor = reward_tensor

        # Normalize model for safety.
        self.T = np.apply_along_axis(normalize, 2, self.T)
        self.O = np.apply_along_axis(normalize, 2, self.O)

    def initial_belief(self) -> np.ndarray:
        return self._initial_belief.copy()

    def sample_initial_state(self, rng: np.random.Generator) -> int:
        return sample_categorical(self._initial_belief, rng)

    def is_terminal(self, state: int) -> bool:
        return bool(self.terminal_mask[state])

    def step(self, state: int, action: int, rng: np.random.Generator) -> tuple[int, int, float, bool]:
        next_state = sample_categorical(self.T[action, state, :], rng)
        obs = sample_categorical(self.O[action, next_state, :], rng)

        if self.reward_tensor is not None:
            reward = float(self.reward_tensor[action, state, next_state])
        else:
            reward = float(self.R[state, action])

        done = self.is_terminal(next_state)
        return next_state, obs, reward, done


def observation_probability(env: TabularPOMDP, belief: np.ndarray, action: int, obs: int) -> float:
    predicted = belief @ env.T[action]
    return float(np.dot(predicted, env.O[action, :, obs]))


def bayes_update(env: TabularPOMDP, belief: np.ndarray, action: int, obs: int) -> np.ndarray:
    predicted = belief @ env.T[action]
    updated = env.O[action, :, obs] * predicted
    total = updated.sum()
    if total <= 0.0:
        return predicted if predicted.sum() > 0.0 else env.initial_belief()
    return updated / total


def belief_from_particles(n_states: int, particles: list[int]) -> np.ndarray:
    dist = np.zeros(n_states, dtype=float)
    if not particles:
        return normalize(dist)
    for s in particles:
        dist[s] += 1.0
    return dist / len(particles)
