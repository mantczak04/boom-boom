# prob-minesweeper

Probabilistic Minesweeper environment for reinforcement learning. Each cell has a mine
probability `p ∈ [0, 1]`; whether a mine is present is sampled when the cell is revealed.
Built as a [Gymnasium](https://gymnasium.farama.org/) environment (`ProbMinesweeper-v0`).

## Requirements

- Python **3.11+**
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Setup

Clone the repo and install dependencies from the project root:

```bash
git clone <repo-url>
cd boom-boom
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

The suite currently has **79 tests** across:

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

## Dependencies

Runtime: `gymnasium`, `numpy`, `scipy`. Dev: `pytest` (via `[project.optional-dependencies] dev`).
