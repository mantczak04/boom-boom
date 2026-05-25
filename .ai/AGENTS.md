# prob-minesweeper

Probabilistic Minesweeper environment for reinforcement learning. Each cell carries a
continuous mine probability `p ∈ [0.0, 1.0]`; whether a mine is actually present is
sampled lazily at reveal time. Built as a fully Gymnasium-compatible environment.

---

## Project goals

- Provide a stochastic, configurable Minesweeper environment for RL research.
- Agent must learn to manage risk under uncertainty rather than memorise fixed patterns.
- Full Gymnasium compliance so any SB3 / RLlib / CleanRL agent works out of the box.

---

## Core mechanics

### Probabilistic board

- Every cell holds `p_mine: float` assigned at episode start.
- The mine outcome is **not** pre-determined — it is sampled from `Bernoulli(p_mine)` the
  moment the agent reveals a cell.
- Default distribution: **spatially correlated** (Gaussian-blurred random field) so mines
  cluster realistically and the agent can infer neighbour risk from revealed cells.
- Distribution is **configurable** via a `MineDistribution` ABC; multiple implementations
  ship out of the box.

### Display values

Revealed cells show `round(sum(p_mine for neighbour in neighbours), 1)` — a float
analogue of the classic integer clue. This naturally extends Minesweeper reasoning to
continuous risk.

---

## Gymnasium environment

**Registration id:** `ProbMinesweeper-v0`

```python
import gymnasium as gym
import prob_minesweeper  # triggers registration

env = gym.make("ProbMinesweeper-v0", width=9, height=9)
```

### Observation space

`Box(shape=(H, W, C), dtype=np.float32)` with two configurable channel sets:

| Mode | Channels | Description |
|------|----------|-------------|
| `obs_mode="state"` | 3 | `[is_revealed, display_value, is_flagged*]` |
| `obs_mode="state+prob"` | 4 | above + `p_mine` for unrevealed cells (0.0 for revealed) |

`*` flagging not implemented in v1; channel reserved for future use.

### Action space

`Discrete(H * W)` — flat index of cell to reveal.

An `action_mask` boolean array of shape `(H * W,)` is included in the `info` dict
returned by `reset()` and `step()`. Already-revealed cells are masked out.

### Reward function

Configurable via `RewardConfig`. Default (`RewardConfig.risk_adjusted`):

| Event | Reward |
|-------|--------|
| Safe reveal | `1.0 - p_mine` of the revealed cell |
| Mine hit (lose) | `-p_mine` of the revealed cell |
| Win (all safe cells revealed) | `+1.0` bonus |
| Step with no new reveal | `0.0` |

### Termination / truncation

| Condition | Type |
|-----------|------|
| Agent reveals a cell that samples as mine | `terminated = True` (loss) |
| All cells that sampled as safe are revealed | `terminated = True` (win) |
| Steps exceed `max_steps` (default `H * W * 2`) | `truncated = True` |

### Render modes

| Mode | Output |
|------|--------|
| `"human"` | ASCII grid printed to stdout |
| `"rgb_array"` | `np.ndarray` of shape `(H*16, W*16, 3)` for logging |
| `None` | No rendering (default, fastest) |

ASCII legend:
```
.   revealed, no mine-risk neighbours
1.3 revealed, neighbour p-sum = 1.3
#   unrevealed
X   revealed mine (terminal)
```

---

## Environment parameters

```python
gym.make(
    "ProbMinesweeper-v0",
    width=9,                          # board width
    height=9,                         # board height
    distribution="correlated",        # "correlated" | "uniform" | "constant" | custom MineDistribution
    distribution_kwargs={},           # passed to distribution constructor
    obs_mode="state",                 # "state" | "state+prob"
    reward_config=RewardConfig.risk_adjusted(),  # RewardConfig instance
    max_steps=None,                   # None → H*W*2
    render_mode=None,                 # None | "human" | "rgb_array"
    seed=None,
)
```

---

## Package structure

```
prob_minesweeper/
├── __init__.py          # gymnasium registration, public re-exports
├── env.py               # ProbMinesweeperEnv(gymnasium.Env)
├── distributions.py     # MineDistribution ABC + CorrelatedDistribution, UniformDistribution, ConstantDistribution
├── rewards.py           # RewardConfig dataclass + built-in factories
├── rendering.py         # ascii_render(), to_rgb_array()
└── cli.py               # CLI entry points
pyproject.toml
tests/
├── test_env.py          # unit tests + env_checker
└── test_distributions.py
```

---

## Key abstractions

### `MineDistribution` (ABC)

```python
class MineDistribution(ABC):
    @abstractmethod
    def generate(self, height: int, width: int, rng: np.random.Generator) -> np.ndarray:
        """Return float32 array of shape (H, W) with values in [0, 1]."""
```

Implementations: `CorrelatedDistribution(sigma, scale)`, `UniformDistribution(low, high)`,
`ConstantDistribution(p)`.

### `RewardConfig` (dataclass)

```python
@dataclass
class RewardConfig:
    reveal_reward_fn: Callable[[float], float]   # p_mine → reward on safe reveal
    mine_penalty_fn:  Callable[[float], float]   # p_mine → penalty on mine hit
    win_bonus: float
```

Factory: `RewardConfig.risk_adjusted()`, `RewardConfig.sparse()`, `RewardConfig.uniform()`.

---

## CLI

```bash
# Interactive play (human chooses cells)
prob-minesweeper play --width 9 --height 9 --distribution correlated

# Benchmark a random agent
prob-minesweeper benchmark --episodes 1000 --width 9 --height 9
```

---

## Testing

Run with:
```bash
uv run pytest
```

Test suite (`tests/test_env.py`):
- `gymnasium.utils.env_checker(env)` — validates full Gymnasium API compliance.
- Termination on mine hit.
- Win condition when all safe cells revealed.
- `action_mask` correctness (no already-revealed cell is unmasked).
- Reward shape matches `RewardConfig`.
- Both `obs_mode` channel counts.
- `reset(seed=...)` reproducibility.

---

## Stack

| Concern | Choice |
|---------|--------|
| Language | Python 3.11+ |
| Package manager | `uv` |
| RL interface | `gymnasium>=0.29` |
| Numerics | `numpy` |
| Spatial noise | `scipy.ndimage.gaussian_filter` |
| Testing | `pytest` |
| Rendering (rgb) | `numpy` only (no pygame) |

---

## Non-goals (v1)

- Flagging mechanics (action space reserved, not implemented).
- Multi-agent / cooperative play.
- GUI / pygame window.
- Pre-trained agents (environment only).