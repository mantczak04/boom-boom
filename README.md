# prob-minesweeper

Probabilistic Minesweeper implemented as a Gymnasium reinforcement-learning
environment, with a Streamlit application, baseline agents, Stable-Baselines3
DQN, sb3-contrib MaskablePPO, and an adapted Rainbow DQN implementation.

In the main hidden-risk variant, every cell has a mine probability `p_mine`, but
the agent does not observe that probability field. At reset, the true mine layout
is sampled from Bernoulli distributions. Safe reveals expose clues, mine reveals
end the episode, and the objective is to reveal all safe cells.

## Quick Start

Requirements:

- Python 3.11+
- `uv` recommended, or a standard virtual environment with pip

Install the core project and development dependencies:

```bash
uv sync --dev
```

Install optional reinforcement-learning dependencies:

```bash
uv sync --dev --extra rl
```

Run the Streamlit app:

```bash
uv run streamlit run app.py
```

Run tests:

```bash
uv run pytest -q
```

Useful Makefile shortcuts:

```bash
make install
make test
make app
make benchmark
```

## What This Project Contains

```text
prob_minesweeper/        Core package: board, environment, rewards, agents, CLI
prob_minesweeper/agents/ Reusable Random, Min-risk, DQN, MaskablePPO agents
experiments/             SB3 DQN and MaskablePPO train/evaluate/compare scripts
rainbow/                 Adapted Rainbow DQN implementation
app.py                   Streamlit game, benchmark, and model explanation UI
agents/greedy.py         Standalone greedy p_mine baseline CLI
scripts/train.sh         Rainbow training helper
tests/                   Pytest suite
report/                  Polish report and defense notes
models/                  Saved SB3/sb3-contrib `.zip` models
results/                 Saved Rainbow model and metric artifacts
```

The package registers `ProbMinesweeper-v0` with Gymnasium when
`prob_minesweeper` is imported.

## Environment Model

### Board

The board has height `H` and width `W`. Each cell `(r, c)` receives a probability
`p_mine[r, c]` in `[0, 1]`. At episode reset, a hidden mine outcome is sampled:

```text
M[r, c] ~ Bernoulli(p_mine[r, c])
```

The sampled mine layout remains fixed during the episode. The agent only discovers
whether a selected cell contains a mine when it reveals that cell.

An episode ends in one of three ways:

- win: every non-mine cell has been revealed; mines may remain hidden
- loss: a revealed cell contains a mine
- truncation: the step limit is reached, default `2 * H * W`

### Actions

The action space is discrete:

```text
A = {0, 1, ..., H * W - 1}
```

Actions are flat row-major cell indices:

```text
action = row * width + col
```

The environment returns `info["action_mask"]`, a boolean vector of shape
`(H * W,)`. Revealed cells are marked invalid. MaskablePPO uses this mask directly;
other agents should also use it to avoid no-op revealed-cell actions.

### Observations

Gymnasium observations have shape `(H, W, C)` and dtype `float32`.

| Mode | Channels | Meaning |
|------|----------|---------|
| `state` | 3 | revealed flag, revealed clue value, reserved flag channel |
| `state+prob` | 4 | the three state channels plus hidden-cell `p_mine` |

The reserved flag channel is currently always `0.0`. In `state+prob`, the
probability channel is present only for unrevealed cells; revealed cells store
`0.0` in that channel.

### Clues

Two clue modes are implemented:

| Mode | Safe revealed-cell value |
|------|--------------------------|
| `actual_count` | Number of sampled neighbouring mines in the eight-cell neighbourhood |
| `prob_sum` | Rounded sum of neighbouring `p_mine` values |

`actual_count` is closest to classic Minesweeper after the hidden mine layout has
been sampled. `prob_sum` is the original probabilistic clue analogue.

### Initial Reveal

`initial_reveal` can be:

| Mode | Behavior |
|------|----------|
| `none` | No automatic reveal at reset |
| `safe_2x2` | Reveal a random safe 2x2 block at reset |

The `safe_2x2` opening gives four visible clues before the first agent action.
Those automatic reveals give no reward and consume no steps. One additional cell
outside the block is forced safe but remains hidden, so reset does not return an
already-completed board. This mode requires at least one cell outside the 2x2 block.

## Probability Distributions

The environment generates a new probability field on each reset.

| Distribution | Parameters | Description |
|--------------|------------|-------------|
| `constant` | `p` | Every cell has the same mine probability |
| `uniform` | `low`, `high` | Each cell probability is sampled independently from `[low, high]` |
| `correlated` | `sigma`, `scale` | Gaussian-blurred noise creates spatially correlated risk regions |

All generated probabilities are clipped to `[0, 1]`.

## Reward Modes

`RewardConfig` maps reveal outcomes to scalar rewards. A winning reveal receives
the safe-reveal reward plus the win bonus. A no-op revealed-cell action receives
`0.0`.

| Mode | Safe reveal | Mine hit | Win bonus |
|------|-------------|----------|-----------|
| `risk_adjusted` | `1 - p_mine` | `-p_mine` | `+1` |
| `sparse` | `0` | `-1` | `+1` |
| `uniform` | `+1` | `-1` | `+1` |
| `completion` | `+0.1` | `-1` | `+10` |

The hidden-risk RL scripts default to `completion`, because it rewards board
completion without revealing local risk through the reward value.

## Main Hidden-Risk Setting

The recommended RL configuration is:

```text
obs_mode       = state
clue_mode      = actual_count
initial_reveal = safe_2x2
reward_mode    = completion
```

This makes the problem sequential: choosing a safe cell gives reward and reveals a
clue that can improve later decisions. In contrast, the original full-information
configuration is available with:

```text
obs_mode  = state+prob
clue_mode = prob_sum
```

`MinRiskAgent` is an oracle in hidden-risk mode because it reads `p_mine` directly
from the internal board, even though that field is not present in the agent's
observation.

## University-Level Theory

The fully observed variant can be described as a finite Markov decision process
(MDP):

```text
(S, A, P, R, gamma)
```

- `S`: board observation tensor, including revealed cells, clues, and optionally
  probabilities
- `A`: flat cell index actions
- `P`: stochastic transition induced by the sampled mine layout and reveal result
- `R`: reward function selected by `reward_mode`
- `gamma`: discount factor used by RL algorithms

The hidden-risk variant is partially observable with respect to the probability
field and hidden mine layout. In practice, the project still exposes a Gymnasium
state tensor to standard RL algorithms; the policy must infer useful behaviour from
revealed clues and previous reveals encoded in the visible board.

The standard RL objective is to maximize expected discounted return:

```text
E[sum_{t=0}^{T} gamma^t r_t]
```

DQN learns an action-value function `Q(s, a)`, the expected long-term value of
revealing cell `a` in state `s`. The Bellman target is:

```text
y = r + gamma * max_a' Q(s', a')
```

Training minimizes the temporal-difference error between the current prediction
`Q_theta(s, a)` and target `y`.

PPO instead learns a stochastic policy directly. MaskablePPO modifies the policy
distribution so invalid actions, here already revealed cells, receive zero
probability before sampling or choosing an action.

## Agents

| Agent | Location | Notes |
|-------|----------|-------|
| `RandomAgent` | `prob_minesweeper/agents/random_agent.py` | Uniform random valid-action baseline |
| `MinRiskAgent` | `prob_minesweeper/agents/min_risk_agent.py` | Chooses the valid cell with the lowest internal `p_mine`; oracle in hidden-risk mode |
| `DQNAgent` | `prob_minesweeper/agents/dqn_agent.py` | Loads a Stable-Baselines3 DQN model and repairs invalid predictions with a valid fallback |
| `MaskablePPOAgent` | `prob_minesweeper/agents/maskable_ppo_agent.py` | Loads an sb3-contrib MaskablePPO model and passes `action_mask` to prediction |
| `greedy-benchmark` | `agents/greedy.py` | Standalone p_mine-minimizing benchmark CLI |
| Rainbow `Agent` | `rainbow/agent.py` | Adapted masked Rainbow DQN agent |

DQN and MaskablePPO models are tied to board size and observation shape. Train and
evaluate with matching `--width`, `--height`, `--obs-mode`, `--clue-mode`,
`--initial-reveal`, and `--reward-mode`.

## Streamlit Application

Run:

```bash
uv run streamlit run app.py
```

The sidebar controls:

- board width and height
- probability distribution and distribution parameters
- clue mode
- initial reveal mode
- reward mode
- seed
- optional hidden probability display for human/debug play
- selected DQN and MaskablePPO model files, when present

Tabs:

- Game: manual play plus Random, Min-risk, DQN, and MaskablePPO controls
- Benchmark: compare available agents on the selected configuration
- Model: short explanation of the RL formulation

The app uses `obs_mode="state"` for gameplay and benchmarks. Showing hidden
probabilities in the UI is only for human/debug inspection.

## CLI Usage

Play manually in the terminal:

```bash
uv run prob-minesweeper play
```

Example:

```bash
uv run prob-minesweeper play \
  --width 9 \
  --height 9 \
  --distribution correlated \
  --seed 42
```

At the prompt, reveal a cell with either a flat index or `row col`, both 0-based.
Use `q`, `quit`, or `exit` to stop.

Run the random valid-action benchmark:

```bash
uv run prob-minesweeper benchmark --episodes 1000
```

Hidden-risk benchmark example:

```bash
uv run prob-minesweeper benchmark \
  --episodes 100 \
  --width 5 \
  --height 5 \
  --distribution correlated \
  --obs-mode state \
  --clue-mode actual_count \
  --initial-reveal safe_2x2 \
  --reward-mode completion
```

Greedy p_mine benchmark:

```bash
uv run greedy-benchmark --episodes 1000 --compare-random
```

## Gymnasium Usage

```python
import gymnasium as gym
import prob_minesweeper  # registers ProbMinesweeper-v0

env = gym.make(
    "ProbMinesweeper-v0",
    width=9,
    height=9,
    obs_mode="state",
    clue_mode="actual_count",
    initial_reveal="safe_2x2",
)

obs, info = env.reset(seed=42)
action = int(info["action_mask"].nonzero()[0][0])
obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

Important constructor arguments:

```text
width, height
distribution
distribution_kwargs
obs_mode
clue_mode
initial_reveal
reward_config
max_steps
render_mode
seed
```

Rendering modes are `None`, `human`, and `rgb_array`.

## Stable-Baselines3 DQN

Install RL dependencies:

```bash
uv sync --dev --extra rl
```

Train the default hidden-risk 5x5 DQN:

```bash
uv run python experiments/train_dqn.py --timesteps 500000
```

Evaluate:

```bash
uv run python experiments/evaluate_dqn.py --episodes 500
```

Default output:

```text
models/dqn_prob_minesweeper.zip
```

SB3 DQN uses flattened observations and does not use `action_mask` during training.
During evaluation and in the app, an invalid predicted revealed-cell action is
replaced by a random valid fallback action. Comparison scripts report DQN's
invalid-action rate.

Quick smoke test:

```bash
uv run python experiments/train_dqn.py --timesteps 1000
uv run python experiments/evaluate_dqn.py --episodes 10
```

## MaskablePPO

Train the default hidden-risk 5x5 MaskablePPO model:

```bash
uv run python experiments/train_maskable_ppo.py --timesteps 100000
```

Evaluate:

```bash
uv run python experiments/evaluate_maskable_ppo.py --episodes 500
```

Default output:

```text
models/maskable_ppo_prob_minesweeper.zip
```

MaskablePPO uses `action_mask` during both training and prediction, which matches
the Minesweeper rule that revealed cells should not be selected again.

Compare available agents:

```bash
uv run python experiments/compare_rl_agents.py --episodes 500
```

Useful easier learning run:

```bash
uv run python experiments/train_maskable_ppo.py \
  --timesteps 100000 \
  --width 5 \
  --height 5 \
  --distribution constant \
  --p 0.15 \
  --obs-mode state \
  --clue-mode actual_count \
  --initial-reveal safe_2x2 \
  --reward-mode completion \
  --output models/maskable_ppo_easy_constant_p015_100k.zip
```

## Rainbow DQN

The `rainbow/` package is adapted from Kaixhin's Rainbow implementation and rewired
for Probabilistic Minesweeper instead of Atari. It uses:

- channels-first board tensors `(C, H, W)`
- masked action selection
- C51 distributional value support
- dueling value/advantage heads
- NoisyLinear exploration
- prioritized replay
- multi-step returns
- target network updates

Run the helper script:

```bash
bash scripts/train.sh
```

Fast run:

```bash
FAST=1 bash scripts/train.sh
```

Resume from a checkpoint:

```bash
RESUME_CHECKPOINT=checkpoints/<run-id>/model.pt FAST=1 bash scripts/train.sh
```

Direct module invocation:

```bash
uv run python -m rainbow.main \
  --id correlated-9x9-state+prob \
  --board-width 9 \
  --board-height 9 \
  --distribution correlated \
  --obs-mode state+prob \
  --T-max 200000 \
  --disable-cuda
```

Rainbow outputs include:

```text
results/<run-id>/model.pth
results/<run-id>/metrics.pth
checkpoints/<run-id>/model.pt
runs/<run-id>/metrics.jsonl
```

## Tests

Run all tests:

```bash
uv run pytest -q
```

Useful targeted runs:

```bash
uv run pytest tests/test_env.py -q
uv run pytest tests/test_board.py -q
uv run pytest tests/test_rainbow.py -q
```

Coverage includes:

| Test file | Focus |
|-----------|-------|
| `tests/test_env.py` | Gymnasium compliance, observation/action spaces, masks, rewards, seeds, termination |
| `tests/test_board.py` | Board reveal logic, clues, win/loss rules, sampled mine outcomes |
| `tests/test_distributions.py` | Probability field generators |
| `tests/test_rewards.py` | Reward modes and reward factory |
| `tests/test_agents.py` | Random and Min-risk behaviour |
| `tests/test_dqn_agent.py` | DQN adapter shape checks and invalid-action fallback |
| `tests/test_maskable_ppo_agent.py` | MaskablePPO adapter behaviour |
| `tests/test_evaluation.py` | Shared agent evaluation utilities |
| `tests/test_cli.py` | CLI parsing and random benchmark |
| `tests/test_rendering.py` | ASCII and RGB renderers |
| `tests/test_rainbow.py` | Rainbow environment adapter and network shapes |
| `tests/test_rainbow_memory.py` | Rainbow replay-memory sampling shapes |
| `tests/test_app_imports.py` | Streamlit app importability |

`tests/test_env.py` also uses `gymnasium.utils.env_checker.check_env`.

## Interpreting Results

Compare agents only under the same board size, distribution, observation mode, clue
mode, initial reveal mode, reward mode, and seed sequence.

Random is a baseline. Min-risk is an oracle in hidden-risk mode. DQN and
MaskablePPO are learned visible-state policies. MaskablePPO has the cleanest action
masking among the SB3-style models, while basic SB3 DQN needs invalid-action repair
at inference time.

For hard hidden-risk configurations, win rate alone can be sparse. Mean reward,
mean steps, loss/truncation counts, and DQN invalid-action rate are also important.

## Limitations

- The current environment has reveal actions only; flagging is represented by a
  reserved observation channel but is not implemented as an action.
- SB3 DQN does not train with action masks.
- Saved SB3 and MaskablePPO models are shape-specific.
- The hidden-risk setting is partially observable; standard feed-forward policies
  receive the visible board tensor but no explicit memory beyond what is encoded in
  revealed cells.
- Min-risk should not be treated as a fair hidden-risk baseline because it reads
  privileged internal probabilities.

