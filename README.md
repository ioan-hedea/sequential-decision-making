# Sequential Decision-Making

Hands-on implementations and visualizations for core sequential decision-making topics:

- Dynamic Programming for MDPs
- POMDP belief-space planning
- Monte Carlo Tree Search (MCTS) and POMCP
- Model-based, Bayesian, and safe RL concepts
- Inverse RL and verification
- Model Predictive Control (MPC)

The repository is organized by topic, with most modules providing runnable scripts that generate figures and demonstrations.

## Repository Structure

- `mdp_dp/` — Value Iteration, Policy Iteration, Bellman contraction demos
- `pomdp/` — Tiger POMDP and abstract POMDP examples with belief updates/PBVI
- `pomcp/` — Generic POMCP implementation + Tiger and RockSample examples
- `mcts/` — MCTS/UCT visualizations and behavior analyses
- `model_based_rl/` — Dyna-Q visualizations and model-learning intuition
- `bayesian_rl/` — Thompson sampling, conjugate priors, posterior shrinkage
- `safe_rl/` — CVaR, robust MDPs, constrained MDPs, SPIBB-style concepts
- `verification/` — ReLU/SMT encoding, reachability, bounded model checking visuals
- `depp_rl/` — Q-learning and deep Q-learning assignment code
- `irl/` — Linear-programming inverse RL + additional verification/control subfolders
- `mpc/` — Python and MATLAB MPC modules, report, and generated figures

## Quick Start

### 1. Prerequisites

- Python 3.10+ recommended
- Optional: `uv` (for lightweight, reproducible runs)
- For MPC Python modules: `scipy`
- For IRL LP solver: `cvxopt`
- For `depp_rl/` deep RL scripts: `gym` and `torch`
- For Julia-based subfolders under `irl/`: Julia (see local READMEs)

### 2. Install Common Python Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy
```

Optional extras:

```bash
pip install cvxopt gym torch
```

### 3. Run Main Visualization Modules

From repository root:

```bash
python mdp_dp/mdp_visualization.py
python pomdp/pomdp_visualization.py
python pomdp/abstract_pomdps.py
python pomcp/pomcp_visualization.py
python mcts/mcts_visualization.py
python model_based_rl/dyna_q_visualization.py
python bayesian_rl/bayesian_rl_visualization.py
python safe_rl/safe_rl_visualization.py
python verification/verification_visualization.py
```

Each script runs multiple demos and saves figures into its module directory.

## Module Entry Points

- `mdp_dp/mdp_visualization.py`
- `pomdp/pomdp_visualization.py`
- `pomdp/abstract_pomdps.py`
- `pomcp/pomcp_visualization.py`
- `pomcp/tiger_problem.py`
- `pomcp/rocksample.py`
- `mcts/mcts_visualization.py`
- `model_based_rl/dyna_q_visualization.py`
- `bayesian_rl/bayesian_rl_visualization.py`
- `safe_rl/safe_rl_visualization.py`
- `verification/verification_visualization.py`
- `irl/main.py`
- `mpc/Module_one.py`
- `mpc/Module_two.py`
- `mpc/Module_three.py`

## Notes By Folder

### `depp_rl/`

Contains assignment-style tabular and deep Q-learning code:

- `q_learning_main.py`
- `deep_q_learning_main.py`

Some scripts assume older Gym APIs and local assignment scaffolding.

### `irl/`

Includes:

- Linear IRL implementation (`main.py`, `linear_irl.py`, `gridworld.py`)
- Additional subprojects:
  - `SMT Solver/`
  - `NN Controllers/`
  - `Neural Certificates/`

Refer to the local READMEs in each subfolder for environment-specific setup.

### `mpc/`

Contains Python and MATLAB versions of assignment modules plus a report:

- Python: `Module_one.py`, `Module_two.py`, `Module_three.py`
- MATLAB: `Module_one.m`, `Module_two.m`, `Module_three.m`
- Outputs: `mpc/figures/`

## Suggested `uv` Usage (Optional)

If you use `uv`, you can run scripts without manually managing a virtualenv:

```bash
uv run --with numpy --with matplotlib python mdp_dp/mdp_visualization.py
uv run --with numpy --with matplotlib --with scipy python mpc/Module_two.py
uv run --with numpy --with matplotlib --with cvxopt python irl/main.py
```

## License and Attribution

- `irl/` contains attribution details for adapted inverse RL code in its local [`irl/README.md`](irl/README.md).
