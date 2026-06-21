# Probabilistic Minesweeper — Reinforcement Learning Project

## Overview

This project implements a reinforcement learning agent trained on a modified version of the classic Minesweeper game. The core modification replaces the deterministic mine placement with a **probabilistic model**: instead of a field either having a mine or not, each cell carries a continuous probability `p(i,j) ∈ [0,1]` of containing a mine, sampled at reveal time. This guarantees a permanent degree of uncertainty and prevents the agent from learning a simple lookup strategy.

The goal is to study whether a trained RL agent can outperform a greedy baseline (always reveal the cell with the lowest current mine probability) by learning to reason about **spatial correlations**, **multi-step information gain**, and **risk-vs-exploration trade-offs**.

---

## Motivation

Classical Minesweeper is largely deterministic once the board is known. A greedy policy — always pick the safest cell — is near-optimal under independent, visible probabilities. This environment is specifically designed to break that assumption:

- Mine probabilities are **hidden** from the agent; they must be inferred from neighbor counts and revealed cells.
- Probabilities are **spatially correlated** (mines cluster), so revealing one cell can significantly update the risk of adjacent cells.
- A **win condition requiring substantial board exploration** prevents the agent from parking on a few safe corners.

Under these conditions, greedy is no longer optimal, and an RL agent that learns to plan ahead and maximize information gain has a structural advantage.

---

## Project Goal

Train an RL agent that achieves a higher win rate than the following baselines:

| Baseline | Description |
|---|---|
| Random | Selects a random unrevealed cell each step |
| Greedy | Always reveals the cell with the lowest observed risk |
| Smart greedy | Greedy with Bayesian `p` updates after each reveal |

The agent should demonstrate measurable improvement specifically in scenarios with high spatial correlation among mines — the regime where multi-step planning yields the most benefit.

---

## Environment Design

### Board representation

The board is an `N×N` grid. Each cell has a latent mine probability assigned at episode start (e.g. sampled from a Gaussian Random Field to induce spatial correlation). When the agent reveals a cell, a mine appears with probability `p(i,j)` — the outcome is sampled at that moment.

### Observation space

The agent receives a multi-channel tensor of shape `(4, N, N)`:

| Channel | Content |
|---|---|
| 0 | Revealed neighbor counts (normalized), 0 if unrevealed |
| 1 | Cell state mask: unrevealed / revealed / flagged |
| 2 | Local entropy estimate (uncertainty of each unrevealed cell) |
| 3 | Step count / time pressure signal (normalized) |

The agent does **not** observe the true `p(i,j)` values directly.

### Action space

Discrete: select any of the `N×N` cells. Already-revealed cells are masked out (invalid actions) using action masking.

### Reward function

```
r(t) = α · ΔH(t) + β · (1 - p_true(i,j)) · revealed - γ · mine_hit - δ · timeout
```

- `ΔH(t)` — reduction in board entropy after the reveal (information gain bonus)
- `(1 - p_true) · revealed` — reward proportional to safety of the chosen cell
- `mine_hit` — large negative penalty on detonation
- `timeout` — small penalty if the episode ends without reaching the exploration threshold

### Win condition

The episode is won when the agent reveals at least **X% of non-mine cells** without detonating. This threshold forces active exploration and prevents degenerate safe-corner strategies.

---

## Tech Stack

### Core

| Tool | Role |
|---|---|
| Python 3.11+ | Primary language |
| `gymnasium` | Environment API (`ProbabilisticMinesweeper-v0` custom env) |
| `numpy` | Board state, probability fields, entropy computation |
| `scipy` | Gaussian Random Field generation for correlated mine layouts |

### RL Training

| Tool | Role |
|---|---|
| `PyTorch` | Neural network implementation |
| `stable-baselines3` | PPO / DQN / A2C implementations |
| `sb3-contrib` | `MaskablePPO` for action masking support |
| `RLlib` (optional) | Alternative for distributed / large-scale experiments |

### Neural Network Architecture

- **Input:** `(4, N, N)` multi-channel board tensor
- **Backbone:** CNN (shared feature extractor)
- **Policy head:** Softmax over `N×N` actions (masked)
- **Value head:** Scalar `V(s)` estimate

For advanced experiments: Graph Neural Network (GNN) backbone treating cells as nodes with edges to their 8 neighbors.

### Experiment Management

| Tool | Role |
|---|---|
| `wandb` | Metric logging, run comparison, sweep visualization |
| `hydra` + `omegaconf` | Configuration management across experiments |
| `optuna` | Hyperparameter search |
| `tensorboard` | Fallback logging / local use |

### Development

| Tool | Role |
|---|---|
| `pytest` | Unit tests for environment logic |
| `gymnasium.utils.env_checker` | Validates the custom environment conforms to the Gym API |
| `pygame` / `matplotlib` | Optional real-time visualization of agent play |

---

## Curriculum Learning Strategy

Training proceeds in stages of increasing difficulty:

1. **Stage 1 — Low density, independent `p`:** Small board (5×5), low mine probability (~0.1), no spatial correlation. Agent learns basic survival.
2. **Stage 2 — Correlated layouts:** Introduce spatially correlated mine fields. Agent must begin using neighbor information.
3. **Stage 3 — Hidden probabilities:** Agent no longer receives any direct risk signal; must infer entirely from revealed numbers.
4. **Stage 4 — Full difficulty:** Larger board (9×9 or 16×16), high correlation, strict exploration threshold, time pressure.

---

## Evaluation Metrics

- **Win rate** — primary metric; percentage of episodes where the exploration threshold is reached without hitting a mine
- **Mean episode length** — longer episodes suggest the agent is exploring rather than terminating early
- **Unnecessary risk score** — how often the agent selects a cell with `p > median(p_unrevealed)` when safer options exist
- **Information efficiency** — entropy reduction per step; higher is better

---

## Repository Structure (planned)

```
project/
├── env/
│   ├── minesweeper_env.py      # Gymnasium environment
│   ├── board_generator.py      # GRF-based probabilistic board generation
│   └── reward.py               # Reward function components
├── agent/
│   ├── cnn_policy.py           # Custom CNN feature extractor
│   └── gnn_policy.py           # Optional GNN backbone
├── baselines/
│   ├── random_agent.py
│   ├── greedy_agent.py
│   └── smart_greedy_agent.py   # Greedy + Bayesian p updates
├── train.py                    # Main training entry point
├── evaluate.py                 # Baseline comparison & metric logging
├── configs/                    # Hydra config files
└── tests/                      # Pytest test suite
```
