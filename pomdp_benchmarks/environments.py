from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from .core import TabularPOMDP


def make_tiger_pomdp() -> TabularPOMDP:
    states = ["tiger_left", "tiger_right"]
    actions = ["listen", "open_left", "open_right"]
    observations = ["hear_left", "hear_right"]

    n_states = 2
    n_actions = 3
    n_obs = 2

    T = np.zeros((n_actions, n_states, n_states), dtype=float)
    O = np.zeros((n_actions, n_states, n_obs), dtype=float)

    # listen: state stays
    T[0, 0, 0] = 1.0
    T[0, 1, 1] = 1.0

    # open actions: tiger resets uniformly
    for a in (1, 2):
        T[a, :, 0] = 0.5
        T[a, :, 1] = 0.5

    # listen observations
    O[0, 0, 0] = 0.85
    O[0, 0, 1] = 0.15
    O[0, 1, 0] = 0.15
    O[0, 1, 1] = 0.85

    # open observations are uninformative
    for a in (1, 2):
        O[a, :, :] = 0.5

    R = np.array(
        [
            [-1.0, -100.0, 10.0],
            [-1.0, 10.0, -100.0],
        ],
        dtype=float,
    )

    return TabularPOMDP(
        name="Tiger",
        gamma=0.95,
        transition=T,
        observation=O,
        reward=R,
        initial_belief=np.array([0.5, 0.5], dtype=float),
        action_names=actions,
        observation_names=observations,
        state_names=states,
        horizon=15,
    )


def _default_rock_positions(n: int, k: int) -> list[tuple[int, int]]:
    rng = np.random.default_rng(17)
    positions: list[tuple[int, int]] = []
    used = set([(n // 2, 0)])
    while len(positions) < k:
        cell = (int(rng.integers(0, n)), int(rng.integers(0, n)))
        if cell not in used:
            used.add(cell)
            positions.append(cell)
    return positions


def make_rocksample_pomdp(n: int = 4, k: int = 3) -> TabularPOMDP:
    rock_positions = _default_rock_positions(n, k)

    action_names = ["north", "east", "south", "west", "sample"] + [
        f"check_{i}" for i in range(k)
    ]
    obs_names = ["none", "good", "bad"]

    move_north = 0
    move_east = 1
    move_south = 2
    move_west = 3
    sample = 4
    check_offset = 5

    states: list[tuple[int, int, int]] = []
    for row in range(n):
        for col in range(n):
            for mask in range(1 << k):
                states.append((row, col, mask))

    terminal_state = len(states)
    state_names = [f"r{r}_c{c}_mask{m:0{k}b}" for (r, c, m) in states] + ["terminal"]

    def state_idx(row: int, col: int, mask: int) -> int:
        return ((row * n) + col) * (1 << k) + mask

    n_states = len(states) + 1
    n_actions = 5 + k
    n_obs = len(obs_names)

    T = np.zeros((n_actions, n_states, n_states), dtype=float)
    O = np.zeros((n_actions, n_states, n_obs), dtype=float)
    R = np.zeros((n_states, n_actions), dtype=float)

    d_half = n / 2.0

    for s_idx, (row, col, mask) in enumerate(states):
        # Terminal behavior
        for a in range(n_actions):
            T[a, terminal_state, terminal_state] = 1.0
            O[a, terminal_state, 0] = 1.0
            R[terminal_state, a] = 0.0

        # Movement
        north_row = max(0, row - 1)
        south_row = min(n - 1, row + 1)
        west_col = max(0, col - 1)

        T[move_north, s_idx, state_idx(north_row, col, mask)] = 1.0
        O[move_north, :, 0] = np.maximum(O[move_north, :, 0], 0.0)

        T[move_south, s_idx, state_idx(south_row, col, mask)] = 1.0
        T[move_west, s_idx, state_idx(row, west_col, mask)] = 1.0

        if col == n - 1:
            T[move_east, s_idx, terminal_state] = 1.0
            R[s_idx, move_east] = 10.0
        else:
            T[move_east, s_idx, state_idx(row, col + 1, mask)] = 1.0

        for a in (move_north, move_east, move_south, move_west):
            O[a, :, 0] = 1.0

        # Sample
        rock_here = None
        for i, (rr, cc) in enumerate(rock_positions):
            if rr == row and cc == col:
                rock_here = i
                break

        if rock_here is not None and (mask & (1 << rock_here)):
            next_mask = mask & ~(1 << rock_here)
            T[sample, s_idx, state_idx(row, col, next_mask)] = 1.0
            R[s_idx, sample] = 10.0
        else:
            T[sample, s_idx, state_idx(row, col, mask)] = 1.0
            R[s_idx, sample] = -10.0
        O[sample, :, 0] = 1.0

        # Check actions
        for rock_i in range(k):
            a = check_offset + rock_i
            rr, cc = rock_positions[rock_i]
            dist = math.sqrt((row - rr) ** 2 + (col - cc) ** 2)
            accuracy = (1.0 + 2.0 ** (-dist / d_half)) / 2.0

            is_good = bool(mask & (1 << rock_i))
            T[a, s_idx, s_idx] = 1.0
            if is_good:
                O[a, s_idx, 1] = accuracy
                O[a, s_idx, 2] = 1.0 - accuracy
            else:
                O[a, s_idx, 1] = 1.0 - accuracy
                O[a, s_idx, 2] = accuracy

    # Ensure each action has terminal self-loop fully defined
    for a in range(n_actions):
        O[a, terminal_state, :] = 0.0
        O[a, terminal_state, 0] = 1.0

    initial = np.zeros(n_states, dtype=float)
    start_row, start_col = n // 2, 0
    for mask in range(1 << k):
        initial[state_idx(start_row, start_col, mask)] = 1.0 / (1 << k)

    return TabularPOMDP(
        name=f"RockSample({n},{k})",
        gamma=0.95,
        transition=T,
        observation=O,
        reward=R,
        initial_belief=initial,
        action_names=action_names,
        observation_names=obs_names,
        state_names=state_names,
        terminal_states=[terminal_state],
        horizon=max(20, 4 * n + 6),
    )


def make_driving_merge_pomdp(sensor_noise: float = 0.0, name: str = "DrivingMerge") -> TabularPOMDP:
    # Hidden gap-size state + absorbing outcomes.
    SMALL, MEDIUM, LARGE, MERGED, COLLISION = range(5)
    WAIT, SLOW_MERGE, MERGE = range(3)

    states = ["gap_small", "gap_medium", "gap_large", "merged", "collision"]
    actions = ["wait", "slow_merge", "merge"]
    observations = ["sensor_small", "sensor_medium", "sensor_large", "terminal"]

    n_states = len(states)
    n_actions = len(actions)
    n_obs = len(observations)

    T = np.zeros((n_actions, n_states, n_states), dtype=float)
    O = np.zeros((n_actions, n_states, n_obs), dtype=float)
    R_sas = np.zeros((n_actions, n_states, n_states), dtype=float)

    # WAIT dynamics
    T[WAIT, SMALL, :] = [0.55, 0.35, 0.10, 0.0, 0.0]
    T[WAIT, MEDIUM, :] = [0.20, 0.50, 0.30, 0.0, 0.0]
    T[WAIT, LARGE, :] = [0.10, 0.35, 0.55, 0.0, 0.0]

    # SLOW_MERGE dynamics
    T[SLOW_MERGE, SMALL, :] = [0.20, 0.35, 0.00, 0.10, 0.35]
    T[SLOW_MERGE, MEDIUM, :] = [0.05, 0.20, 0.00, 0.65, 0.10]
    T[SLOW_MERGE, LARGE, :] = [0.01, 0.04, 0.05, 0.86, 0.04]

    # MERGE dynamics
    T[MERGE, SMALL, :] = [0.10, 0.00, 0.00, 0.20, 0.70]
    T[MERGE, MEDIUM, :] = [0.03, 0.05, 0.00, 0.75, 0.17]
    T[MERGE, LARGE, :] = [0.01, 0.02, 0.01, 0.95, 0.01]

    # Absorbing terminal outcomes
    for a in range(n_actions):
        T[a, MERGED, MERGED] = 1.0
        T[a, COLLISION, COLLISION] = 1.0

    sensor_model = {
        SMALL: [0.80, 0.18, 0.02, 0.0],
        MEDIUM: [0.15, 0.70, 0.15, 0.0],
        LARGE: [0.05, 0.20, 0.75, 0.0],
        MERGED: [0.0, 0.0, 0.0, 1.0],
        COLLISION: [0.0, 0.0, 0.0, 1.0],
    }

    if sensor_noise > 0.0:
        noise = float(np.clip(sensor_noise, 0.0, 0.49))
        uniform_non_terminal = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0], dtype=float)
        for s in (SMALL, MEDIUM, LARGE):
            base = np.asarray(sensor_model[s], dtype=float)
            sensor_model[s] = ((1.0 - noise) * base + noise * uniform_non_terminal).tolist()

    for a in range(n_actions):
        for sp in range(n_states):
            O[a, sp, :] = sensor_model[sp]

    # Transition-dependent reward captures collision vs successful merge.
    action_cost = {WAIT: -2.0, SLOW_MERGE: -4.0, MERGE: -1.0}

    for a in range(n_actions):
        for s in range(n_states):
            for sp in range(n_states):
                reward = action_cost[a]
                if sp == MERGED:
                    reward += 45.0
                elif sp == COLLISION:
                    reward -= 140.0
                if s in (MERGED, COLLISION):
                    reward = 0.0
                R_sas[a, s, sp] = reward

    R = np.einsum("ass,ass->sa", T, R_sas)

    return TabularPOMDP(
        name=name,
        gamma=0.97,
        transition=T,
        observation=O,
        reward=R,
        reward_tensor=R_sas,
        initial_belief=np.array([0.45, 0.35, 0.20, 0.0, 0.0], dtype=float),
        action_names=actions,
        observation_names=observations,
        state_names=states,
        terminal_states=[MERGED, COLLISION],
        horizon=18,
    )


def make_medical_diagnosis_pomdp() -> TabularPOMDP:
    HEALTHY, MILD, SEVERE, TERMINAL = range(4)
    BLOOD_TEST, IMAGING, WAIT, DIAG_H, DIAG_M, DIAG_S = range(6)

    states = ["healthy", "mild", "severe", "terminal"]
    actions = [
        "blood_test",
        "imaging",
        "wait",
        "diagnose_healthy",
        "diagnose_mild",
        "diagnose_severe",
    ]
    observations = ["normal", "abnormal", "highly_abnormal", "terminal"]

    n_states = len(states)
    n_actions = len(actions)
    n_obs = len(observations)

    T = np.zeros((n_actions, n_states, n_states), dtype=float)
    O = np.zeros((n_actions, n_states, n_obs), dtype=float)
    R = np.zeros((n_states, n_actions), dtype=float)

    progression = np.array(
        [
            [0.90, 0.08, 0.02, 0.0],
            [0.05, 0.80, 0.15, 0.0],
            [0.01, 0.09, 0.90, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    for a in (BLOOD_TEST, IMAGING, WAIT):
        T[a, :, :] = progression

    for a in (DIAG_H, DIAG_M, DIAG_S):
        T[a, :, :] = 0.0
        T[a, HEALTHY, TERMINAL] = 1.0
        T[a, MILD, TERMINAL] = 1.0
        T[a, SEVERE, TERMINAL] = 1.0
        T[a, TERMINAL, TERMINAL] = 1.0

    obs_wait = np.array(
        [
            [0.65, 0.28, 0.07, 0.0],
            [0.25, 0.55, 0.20, 0.0],
            [0.08, 0.28, 0.64, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    obs_blood = np.array(
        [
            [0.80, 0.16, 0.04, 0.0],
            [0.16, 0.70, 0.14, 0.0],
            [0.05, 0.25, 0.70, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    obs_imaging = np.array(
        [
            [0.90, 0.08, 0.02, 0.0],
            [0.06, 0.82, 0.12, 0.0],
            [0.02, 0.10, 0.88, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    O[BLOOD_TEST, :, :] = obs_blood
    O[IMAGING, :, :] = obs_imaging
    O[WAIT, :, :] = obs_wait

    for a in (DIAG_H, DIAG_M, DIAG_S):
        O[a, :, :] = 0.0
        O[a, :, 3] = 1.0

    R[HEALTHY, BLOOD_TEST] = -2.0
    R[MILD, BLOOD_TEST] = -2.0
    R[SEVERE, BLOOD_TEST] = -2.0

    R[HEALTHY, IMAGING] = -6.0
    R[MILD, IMAGING] = -6.0
    R[SEVERE, IMAGING] = -6.0

    R[HEALTHY, WAIT] = 0.0
    R[MILD, WAIT] = -3.0
    R[SEVERE, WAIT] = -12.0

    # Diagnosis rewards
    R[HEALTHY, DIAG_H] = +35.0
    R[MILD, DIAG_H] = -25.0
    R[SEVERE, DIAG_H] = -80.0

    R[HEALTHY, DIAG_M] = -15.0
    R[MILD, DIAG_M] = +30.0
    R[SEVERE, DIAG_M] = -45.0

    R[HEALTHY, DIAG_S] = -35.0
    R[MILD, DIAG_S] = -20.0
    R[SEVERE, DIAG_S] = +45.0

    return TabularPOMDP(
        name="MedicalDiagnosis",
        gamma=0.95,
        transition=T,
        observation=O,
        reward=R,
        initial_belief=np.array([0.55, 0.30, 0.15, 0.0], dtype=float),
        action_names=actions,
        observation_names=observations,
        state_names=states,
        terminal_states=[TERMINAL],
        horizon=12,
    )


def make_hallway_search_pomdp() -> TabularPOMDP:
    action_names = ["move_left", "move_right", "scan", "claim"]
    obs_names = ["goal_left", "goal_here", "goal_right", "quiet", "terminal"]

    positions = 3
    terminal_state = positions * positions

    def state_idx(agent_pos: int, goal_pos: int) -> int:
        return agent_pos * positions + goal_pos

    state_names = [f"agent{agent}_goal{goal}" for agent in range(positions) for goal in range(positions)] + [
        "terminal"
    ]

    n_states = terminal_state + 1
    n_actions = len(action_names)
    n_obs = len(obs_names)

    MOVE_LEFT, MOVE_RIGHT, SCAN, CLAIM = range(n_actions)

    T = np.zeros((n_actions, n_states, n_states), dtype=float)
    O = np.zeros((n_actions, n_states, n_obs), dtype=float)
    R = np.zeros((n_states, n_actions), dtype=float)

    for agent in range(positions):
        for goal in range(positions):
            s = state_idx(agent, goal)

            left_pos = max(0, agent - 1)
            right_pos = min(positions - 1, agent + 1)

            T[MOVE_LEFT, s, state_idx(left_pos, goal)] = 1.0
            T[MOVE_RIGHT, s, state_idx(right_pos, goal)] = 1.0
            T[SCAN, s, s] = 1.0
            T[CLAIM, s, terminal_state] = 1.0

            O[MOVE_LEFT, :, 3] = 1.0
            O[MOVE_RIGHT, :, 3] = 1.0

            relation = 1 if goal == agent else (0 if goal < agent else 2)
            O[SCAN, s, relation] = 0.82
            O[SCAN, s, (relation + 1) % 3] += 0.09
            O[SCAN, s, (relation + 2) % 3] += 0.09

            O[CLAIM, s, 4] = 1.0

            R[s, MOVE_LEFT] = -1.0
            R[s, MOVE_RIGHT] = -1.0
            R[s, SCAN] = -2.0
            R[s, CLAIM] = 25.0 if agent == goal else -20.0

    for a in range(n_actions):
        T[a, terminal_state, terminal_state] = 1.0
        O[a, terminal_state, :] = 0.0
        O[a, terminal_state, 4] = 1.0

    initial = np.zeros(n_states, dtype=float)
    for goal in range(positions):
        initial[state_idx(1, goal)] = 1.0 / positions

    return TabularPOMDP(
        name="HallwaySearch",
        gamma=0.96,
        transition=T,
        observation=O,
        reward=R,
        initial_belief=initial,
        action_names=action_names,
        observation_names=obs_names,
        state_names=state_names,
        terminal_states=[terminal_state],
        horizon=10,
    )


def make_machine_maintenance_pomdp() -> TabularPOMDP:
    GOOD, WORN, CRITICAL, FAILED = range(4)
    OBSERVE, CONTINUE, SERVICE, REPLACE = range(4)

    states = ["good", "worn", "critical", "failed"]
    actions = ["observe", "continue", "service", "replace"]
    observations = ["healthy", "warning", "alarm", "failed"]

    n_states = len(states)
    n_actions = len(actions)
    n_obs = len(observations)

    T = np.zeros((n_actions, n_states, n_states), dtype=float)
    O = np.zeros((n_actions, n_states, n_obs), dtype=float)
    reward_tensor = np.zeros((n_actions, n_states, n_states), dtype=float)

    T[OBSERVE, :, :] = np.eye(n_states)
    T[CONTINUE, GOOD, :] = [0.78, 0.18, 0.04, 0.0]
    T[CONTINUE, WORN, :] = [0.02, 0.68, 0.22, 0.08]
    T[CONTINUE, CRITICAL, :] = [0.0, 0.08, 0.50, 0.42]
    T[CONTINUE, FAILED, FAILED] = 1.0

    T[SERVICE, GOOD, :] = [0.92, 0.08, 0.0, 0.0]
    T[SERVICE, WORN, :] = [0.70, 0.24, 0.06, 0.0]
    T[SERVICE, CRITICAL, :] = [0.38, 0.44, 0.14, 0.04]
    T[SERVICE, FAILED, FAILED] = 1.0

    for s in (GOOD, WORN, CRITICAL):
        T[REPLACE, s, GOOD] = 1.0
    T[REPLACE, FAILED, FAILED] = 1.0

    obs_observe = np.array(
        [
            [0.86, 0.12, 0.02, 0.0],
            [0.16, 0.70, 0.14, 0.0],
            [0.04, 0.21, 0.75, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    obs_default = np.array(
        [
            [0.74, 0.21, 0.05, 0.0],
            [0.15, 0.63, 0.22, 0.0],
            [0.05, 0.25, 0.70, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    O[OBSERVE, :, :] = obs_observe
    O[CONTINUE, :, :] = obs_default
    O[SERVICE, :, :] = obs_default
    O[REPLACE, :, :] = obs_default

    for s in range(n_states):
        for sp in range(n_states):
            reward_tensor[OBSERVE, s, sp] = -1.0 if s != FAILED else 0.0
            if s == FAILED:
                reward_tensor[CONTINUE, s, sp] = 0.0
                reward_tensor[SERVICE, s, sp] = 0.0
                reward_tensor[REPLACE, s, sp] = 0.0
                continue

            running_reward = {GOOD: 8.0, WORN: 5.0, CRITICAL: 1.5}[s]
            reward_tensor[CONTINUE, s, sp] = running_reward - (35.0 if sp == FAILED else 0.0)
            reward_tensor[SERVICE, s, sp] = -6.0 - (18.0 if sp == FAILED else 0.0)
            reward_tensor[REPLACE, s, sp] = -11.0

    reward = np.einsum("ass,ass->sa", T, reward_tensor)

    return TabularPOMDP(
        name="MachineMaintenance",
        gamma=0.97,
        transition=T,
        observation=O,
        reward=reward,
        reward_tensor=reward_tensor,
        initial_belief=np.array([0.55, 0.30, 0.15, 0.0], dtype=float),
        action_names=actions,
        observation_names=observations,
        state_names=states,
        terminal_states=[FAILED],
        horizon=14,
    )


def make_inventory_control_pomdp() -> TabularPOMDP:
    LOW, HIGH = range(2)
    max_inventory = 3
    SENSE, ORDER0, ORDER1, ORDER2 = range(4)

    action_names = ["sense", "order_0", "order_1", "order_2"]
    observation_names = ["low_signal", "high_signal", "none"]

    states: list[tuple[int, int]] = []
    for inventory in range(max_inventory + 1):
        for regime in (LOW, HIGH):
            states.append((inventory, regime))

    def state_idx(inventory: int, regime: int) -> int:
        return inventory * 2 + regime

    state_names = [f"inv{inventory}_{'low' if regime == LOW else 'high'}" for inventory, regime in states]
    n_states = len(states)
    n_actions = len(action_names)
    n_obs = len(observation_names)

    T = np.zeros((n_actions, n_states, n_states), dtype=float)
    O = np.zeros((n_actions, n_states, n_obs), dtype=float)
    reward_tensor = np.zeros((n_actions, n_states, n_states), dtype=float)

    regime_transition = np.array([[0.82, 0.18], [0.28, 0.72]], dtype=float)
    demand_probs = {
        LOW: {0: 0.25, 1: 0.60, 2: 0.15},
        HIGH: {0: 0.05, 1: 0.35, 2: 0.60},
    }

    for inventory, regime in states:
        s = state_idx(inventory, regime)
        for action in range(n_actions):
            order_qty = 0 if action == SENSE else action - 1
            post_order = min(max_inventory, inventory + order_qty)

            for next_regime in (LOW, HIGH):
                for demand, demand_prob in demand_probs[regime].items():
                    next_inventory = max(0, post_order - demand)
                    sp = state_idx(next_inventory, next_regime)
                    prob = regime_transition[regime, next_regime] * demand_prob
                    T[action, s, sp] += prob

                    sales = min(post_order, demand)
                    stockout = max(0, demand - post_order)
                    holding = next_inventory
                    order_cost = 0.0 if action == SENSE else 2.0 * order_qty
                    sense_cost = 1.0 if action == SENSE else 0.0
                    reward_tensor[action, s, sp] += prob * (
                        6.0 * sales - 4.5 * stockout - 0.8 * holding - order_cost - sense_cost
                    )

            if action == SENSE:
                O[action, :, :] = np.maximum(O[action, :, :], 0.0)
                O[action, s, 0 if regime == LOW else 1] = 0.85
                O[action, s, 1 if regime == LOW else 0] = 0.15
            else:
                O[action, s, 2] = 1.0

    reward = np.einsum("ass,ass->sa", T, reward_tensor)

    return TabularPOMDP(
        name="InventoryControl",
        gamma=0.96,
        transition=T,
        observation=O,
        reward=reward,
        reward_tensor=reward_tensor,
        initial_belief=np.array([0.0, 0.0, 0.0, 0.0, 0.35, 0.65, 0.0, 0.0], dtype=float),
        action_names=action_names,
        observation_names=observation_names,
        state_names=state_names,
        horizon=16,
    )


def make_standard_environments(
    rocksample_n: int = 4,
    rocksample_k: int = 3,
    include_harder_env: bool = False,
    harder_rocksample_n: int = 5,
    harder_rocksample_k: int = 4,
    include_extra_env: bool = False,
) -> Dict[str, TabularPOMDP]:
    envs = {
        "Tiger": make_tiger_pomdp(),
        f"RockSample({rocksample_n},{rocksample_k})": make_rocksample_pomdp(
            n=rocksample_n,
            k=rocksample_k,
        ),
        "DrivingMerge": make_driving_merge_pomdp(),
        "MedicalDiagnosis": make_medical_diagnosis_pomdp(),
    }
    if include_extra_env:
        envs["HallwaySearch"] = make_hallway_search_pomdp()
        envs["MachineMaintenance"] = make_machine_maintenance_pomdp()
        envs["InventoryControl"] = make_inventory_control_pomdp()
    if include_harder_env:
        envs[f"RockSample({harder_rocksample_n},{harder_rocksample_k})"] = make_rocksample_pomdp(
            n=harder_rocksample_n,
            k=harder_rocksample_k,
        )
        envs["DrivingMergeNoisy"] = make_driving_merge_pomdp(
            sensor_noise=0.30,
            name="DrivingMergeNoisy",
        )
    return envs
