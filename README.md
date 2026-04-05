# Sequential Decision-Making

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Focus](https://img.shields.io/badge/focus-RL%20%7C%20POMDP%20%7C%20MPC-0A7E8C)
![Status](https://img.shields.io/badge/status-active-success)

A practical collection of sequential decision-making implementations and visualizations across planning, reinforcement learning, uncertainty-aware control, and verification.

## What This Repository Covers

- Dynamic Programming for MDPs (value iteration, policy iteration, Bellman contraction)
- POMDP belief-space reasoning (belief updates, alpha-vectors, PBVI)
- Monte Carlo Tree Search (MCTS/UCT) and POMCP
- Model-based RL and Bayesian RL intuition
- Safe and robust RL concepts (CVaR, robust/constrained MDPs, SPIBB)
- Inverse RL and neural-policy verification
- Model Predictive Control (Python + MATLAB assignments)

## Repository Map

| Folder | Focus | Main Entry Point |
|---|---|---|
| `mdp_dp/` | Value iteration, policy iteration, contraction | `python3 mdp_dp/mdp_visualization.py` |
| `pomdp/` | Tiger POMDP, PBVI, abstract POMDP scenarios | `python3 pomdp/pomdp_visualization.py` |
| `pomcp/` | Generic POMCP + Tiger/RockSample examples | `python3 pomcp/pomcp_visualization.py` |
| `pomdp_benchmarks/` | Reproducible solver benchmarking across Tiger/RockSample/custom POMDPs | `python3 -m pomdp_benchmarks.run --quick` |
| `mcts/` | UCT phases and exploration behavior | `python3 mcts/mcts_visualization.py` |
| `model_based_rl/` | Dyna-Q and model-learning effects | `python3 model_based_rl/dyna_q_visualization.py` |
| `bayesian_rl/` | Thompson sampling and Bayesian updates | `python3 bayesian_rl/bayesian_rl_visualization.py` |
| `safe_rl/` | Risk-sensitive and constrained RL visuals | `python3 safe_rl/safe_rl_visualization.py` |
| `verification/` | ReLU encoding, reachability, BMC demos | `python3 verification/verification_visualization.py` |
| `depp_rl/` | Assignment-style tabular/deep Q-learning code | `python3 depp_rl/q_learning_main.py` |
| `irl/` | Linear-programming IRL + extra verification/control subprojects | `python3 irl/main.py` |
| `mpc/` | MPC modules, report, generated figures | `python3 mpc/Module_one.py` |

## Quick Start

### 1. Create an Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy
```

### 2. Optional Extras

```bash
pip install cvxopt gym torch
```

- `cvxopt` is needed for `irl/main.py`
- `gym` and `torch` are used in `depp_rl/`

### 3. Run Core Visualizations

```bash
python3 mdp_dp/mdp_visualization.py
python3 pomdp/pomdp_visualization.py
python3 pomcp/pomcp_visualization.py
python3 -m pomdp_benchmarks.run --quick
python3 mcts/mcts_visualization.py
python3 model_based_rl/dyna_q_visualization.py
python3 bayesian_rl/bayesian_rl_visualization.py
python3 safe_rl/safe_rl_visualization.py
python3 verification/verification_visualization.py
```

All visualization scripts save figures directly in their own module directory.

## POMDP Benchmark Suite

Run the full benchmark harness:

```bash
python3 -m pomdp_benchmarks.run --episodes 40 --belief-budgets 64,128,256,512
```

Included environments:
- `Tiger`
- `RockSample(n,k)` (default `4,3`)
- `DrivingMerge` (custom highway merge under uncertainty)
- `MedicalDiagnosis` (tests with costs + stop-and-diagnose)

Included solvers:
- Exact alpha-vector value iteration (finite-horizon tabular baseline)
- PBVI
- POMCP
- DESPOT-style sparse-scenario online planner
- Optional `AdaOPS`-style adaptive POMCP (`--include-adaops`)

Outputs are saved as:
- `benchmark_summary.csv`
- `benchmark_summary.json`

Plot scalability curves from the CSV:

```bash
python3 -m pomdp_benchmarks.plot_scaling --csv /path/to/benchmark_summary.csv
```

## Gallery

| MDP DP | POMDP Belief Updates | MCTS Phases |
|---|---|---|
| ![MDP DP](mdp_dp/vi_vs_pi.png) | ![POMDP Belief](pomdp/belief_updates.png) | ![MCTS](mcts/mcts_phases.png) |

| POMCP Online Planning | Safe RL (CVaR) | Verification Reachability |
|---|---|---|
| ![POMCP](pomcp/pomcp_online_planning.png) | ![CVaR](safe_rl/cvar_visualization.png) | ![Verification](verification/reachability.png) |

## Notes by Subproject

- `depp_rl/`: educational assignment scaffolding (some scripts target older Gym APIs).
- `irl/`: includes adapted inverse RL code with attribution details in `irl/README.md`, plus separate `SMT Solver/`, `NN Controllers/`, and `Neural Certificates/` tracks.
- `mpc/`: includes Python and MATLAB implementations (`Module_one/two/three`) and a report in `mpc/mpc_report.pdf`.

## Optional `uv` Usage

```bash
uv run --with numpy --with matplotlib python3 mdp_dp/mdp_visualization.py
uv run --with numpy --with matplotlib --with scipy python3 mpc/Module_two.py
uv run --with numpy --with matplotlib --with cvxopt python3 irl/main.py
```

## Attribution and Licensing

- `irl/` includes adapted inverse reinforcement learning code and its own attribution/license notes.
- Check local README/LICENSE files inside subfolders for module-specific licensing details.
