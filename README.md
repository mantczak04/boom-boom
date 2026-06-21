# prob-minesweeper

Probabilistic Minesweeper environment for reinforcement learning. Each cell has a mine
probability `p ∈ [0, 1]`. Hidden mine outcomes are sampled at episode start from
`Bernoulli(p_mine)`; the agent does not observe those outcomes until revealing cells.
Built as a [Gymnasium](https://gymnasium.farama.org/) environment (`ProbMinesweeper-v0`).

## Requirements

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

Clone the repo and install dependencies from the project root:

```bash
git clone <repo-url>
cd prob-minesweeper
uv sync --dev
```

`uv sync --dev` creates a virtual environment (`.venv`), installs the package in editable
mode, and pulls in dev tools including pytest.

### Alternative: pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run

All commands below assume you are in the project root. With uv, prefix commands with
`uv run`; with an activated venv, omit it.

### Streamlit application

The application combines manual play, agent demonstrations, benchmarks, and a concise
description of the reinforcement-learning model:

```bash
uv run streamlit run app.py
```

The sidebar configures the board and probability distribution. In the Game tab, use
either the cell buttons or the Random, Min-risk, and (when trained) DQN agent controls.
The Benchmark tab evaluates available agents over reproducible episodes.

### DQN agent

DQN support is optional. Install the RL dependencies, train the default 5x5 model,
evaluate it, and start the application with:

```bash
uv sync --dev --extra rl
uv run python experiments/train_dqn.py --timesteps 50000
uv run python experiments/evaluate_dqn.py --episodes 100
uv run streamlit run app.py
```

Training saves `models/dqn_prob_minesweeper.zip`. Use matching `--width`, `--height`,
and `--distribution` options when training and evaluating a non-default model. For a
quick end-to-end check, reduce `--timesteps` to `1000`.

If no trained model exists, the Streamlit app still works with RandomAgent and
MinRiskAgent. DQN predictions that target an already revealed cell are replaced by a
valid Min-risk action during evaluation and in the application.

### Interactive play

Play manually in the terminal (ASCII board):

```bash
uv run prob-minesweeper play
```

Common options:

```bash
uv run prob-minesweeper play \
  --width 9 \
  --height 9 \
  --distribution correlated \
  --seed 42
```

At the prompt, reveal a cell using either:

- **Flat index** — `0` … `(height × width - 1)`, row-major (e.g. on 9×9, cell `(row=1, col=2)` → `11`)
- **Row and column** — `1 2` (0-based)
- **Quit** — `q`, `quit`, or `exit`

Distributions: `correlated` (default), `uniform`, `constant`.

### Random-agent benchmark

Run a random valid-action agent over many episodes and print aggregate stats:

```bash
uv run prob-minesweeper benchmark --episodes 1000
```

Example with options:

```bash
uv run prob-minesweeper benchmark \
  --episodes 1000 \
  --width 9 \
  --height 9 \
  --distribution constant \
  --p 0.2 \
  --seed 42
```

### Use as a Gymnasium environment

After install, import the package to register the env, then use `gym.make`:

```python
import gymnasium as gym
import prob_minesweeper  # registers ProbMinesweeper-v0

env = gym.make("ProbMinesweeper-v0", width=9, height=9, render_mode="human")
obs, info = env.reset(seed=42)

action = info["action_mask"].argmax()  # example: first valid action
obs, reward, terminated, truncated, info = env.step(action)

env.close()
```

Key constructor kwargs: `width`, `height`, `distribution`, `distribution_kwargs`,
`obs_mode` (`"state"` or `"state+prob"`), `render_mode` (`None`, `"human"`, `"rgb_array"`),
`seed`.

## Test

Run the full suite:

```bash
uv run pytest
```

Useful variants:

```bash
# Quiet summary
uv run pytest -q

# Single file
uv run pytest tests/test_env.py

# Single test by name
uv run pytest tests/test_env.py::test_env_checker -v

# Stop on first failure
uv run pytest -x
```

The suite covers the environment, board, agents, evaluation, CLI, and frontend imports.

| File | Covers |
|------|--------|
| `tests/test_env.py` | Gymnasium API (`env_checker`), obs/action spaces, masks, rewards, termination, seeds |
| `tests/test_board.py` | Board logic, reveals, win/loss |
| `tests/test_distributions.py` | Mine probability field generators |
| `tests/test_rewards.py` | Reward configs and factories |
| `tests/test_rendering.py` | ASCII and RGB rendering |
| `tests/test_cli.py` | CLI parsing, benchmark, entry point |

Gymnasium compliance is checked via `gymnasium.utils.env_checker.check_env` in
`tests/test_env.py`.

## Project layout

```
prob_minesweeper/   # installable package
tests/              # pytest suite
pyproject.toml      # metadata and dependencies
uv.lock             # locked dependency versions (uv)
```

## Agents

- **Random** samples uniformly from actions allowed by `action_mask`.
- **Min-risk** reveals the valid cell with the smallest known mine probability.
- **DQN** loads a trained Stable-Baselines3 action-value model and safely handles
  invalid predictions using `action_mask`.

Random and Min-risk are reusable baselines. DQN is a learned policy compared against
both reference strategies.

## Benchmarking

Run the CLI random baseline with `prob-minesweeper benchmark`, or compare available
agents in the Streamlit Benchmark tab. A fixed seed makes generated episode boards
repeatable.

## Mathematical model

The environment is an MDP: the state contains revealed-cell state and optionally the
probability field, an action reveals one cell, and transitions depend on hidden mine
outcomes sampled at reset. The risk-adjusted reward encourages safe progress and adds
a win bonus. The objective is to maximize expected discounted return.

## Project scope for grade 4.0

The project includes a Gymnasium backend, interactive Streamlit frontend, reusable
agents, reproducible evaluation tooling, automated tests, and Polish report/defense
material. Report screenshots should be added to `report/screenshots/` before submission.

## Dependencies

Runtime: `gymnasium`, `numpy`, `scipy`, `streamlit`. Dev: `pytest` (via
`[project.optional-dependencies] dev`). Optional RL dependencies, including
`stable-baselines3`, are installed through the `rl` extra.
