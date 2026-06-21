# Hidden-Risk Minesweeper RL Variant — Codex Implementation Plan

## Purpose

Modify the current `prob-minesweeper` codebase so the reinforcement learning problem becomes more meaningful than simply choosing the currently lowest `p_mine` cell.

The current environment already supports probabilistic mine fields, Gymnasium integration, rewards, rendering, CLI, tests, Streamlit planning, and DQN planning. However, the current `state+prob` setup gives the agent direct access to hidden mine probabilities. This makes `MinRiskAgent` extremely strong and makes DQN tend to learn the same local risk-minimization behavior.

This plan introduces a new rule variant:

```text
Hidden-risk probabilistic Minesweeper
```

In this variant:

1. Mines are sampled from a hidden probability field at episode start.
2. The RL agent does **not** observe the hidden probability field.
3. Revealing a safe cell shows the **actual number of neighbouring mines**, not the sum of neighbouring probabilities.
4. The reward is focused on game completion rather than immediate risk shaping.
5. DQN is trained on visible board state only.
6. `MinRiskAgent` is treated as an oracle baseline because it uses hidden probabilities.

This makes the project more coherent as a reinforcement learning problem.

---

## Target final behavior

The project should support two clue modes:

```text
prob_sum      — existing behavior, clue = sum of neighbour p_mine values
actual_count  — new behavior, clue = actual number of neighbouring mines
```

The main RL/DQN experiment should use:

```text
clue_mode = "actual_count"
obs_mode = "state"
reward_config = completion reward
```

Recommended final comparison:

```text
RandomAgent       — weak baseline
DQNAgent          — learned policy from visible state
MinRiskAgent      — oracle baseline with hidden probability access
```

Do **not** remove the original mode. Keep backwards compatibility where possible.

---

## Phase 0 — Baseline verification

### Task 0.1 — Run current test suite

Before changing behavior, verify the current state.

```bash
uv sync --dev
uv run pytest -q
```

### Acceptance criteria

- Existing tests pass before implementation begins.
- Any existing failing test is either fixed or documented before feature work.

---

### Task 0.2 — Identify current clue behavior

Review these files:

```text
prob_minesweeper/board.py
prob_minesweeper/env.py
prob_minesweeper/rewards.py
tests/test_board.py
tests/test_env.py
tests/test_rewards.py
```

Current behavior:

```python
cell.display_value = Board.neighbour_p_sum(self.p_mine_field(), row, col)
```

The new implementation must keep this behavior available under:

```text
clue_mode="prob_sum"
```

### Acceptance criteria

- Developer understands all call sites where clue/display values are created, tested, or rendered.

---

## Phase 1 — Add clue mode support to board logic

### Task 1.1 — Define clue mode constants

Add allowed clue modes in `prob_minesweeper/board.py` or a new small module if preferred.

Recommended simple approach:

```python
_CLUE_MODES = ("prob_sum", "actual_count")
```

Do not over-engineer with enums unless useful.

### Acceptance criteria

- Code has one clear source of truth for valid clue modes.
- Invalid clue modes raise a clear `ValueError`.

---

### Task 1.2 — Extend `Board` dataclass with `clue_mode`

Update `Board`:

```python
@dataclass
class Board:
    height: int
    width: int
    cells: list[list[Cell]]
    clue_mode: str = "prob_sum"
    _mine_outcomes: np.ndarray | None = None
```

Update `Board.create`:

```python
@classmethod
def create(
    cls,
    height: int,
    width: int,
    p_mines: np.ndarray,
    clue_mode: str = "prob_sum",
) -> Board:
    ...
```

Validate `clue_mode`.

### Acceptance criteria

- Existing code that calls `Board.create(height, width, p_mines)` still works.
- New code can call `Board.create(..., clue_mode="actual_count")`.
- Invalid clue mode raises `ValueError`.

---

### Task 1.3 — Add actual neighbouring mine count method

Add method to `Board`:

```python
def neighbour_mine_count(self, row: int, col: int) -> int:
    if self._mine_outcomes is None:
        raise RuntimeError("Call new_episode() before reading hidden outcomes")

    total = 0
    for nr, nc in self.iter_neighbours(row, col):
        total += int(self._mine_outcomes[nr, nc])
    return total
```

Return type may be `int`, but `display_value` can still store it as float for observation compatibility.

### Acceptance criteria

- Method returns the exact number of neighbouring mines.
- Method does not include the center cell.
- Method raises before `new_episode()`.

---

### Task 1.4 — Route clue generation through helper

Add a helper method:

```python
def clue_value(self, row: int, col: int) -> float:
    if self.clue_mode == "prob_sum":
        return self.neighbour_p_sum(self.p_mine_field(), row, col)
    if self.clue_mode == "actual_count":
        return float(self.neighbour_mine_count(row, col))
    raise ValueError(f"Unknown clue_mode: {self.clue_mode!r}")
```

Update `Board.reveal`:

```python
cell.display_value = self.clue_value(row, col)
```

### Acceptance criteria

- `prob_sum` behavior remains unchanged.
- `actual_count` behavior uses sampled hidden mine outcomes.
- Existing tests for `prob_sum` still pass.

---

## Phase 2 — Environment integration

### Task 2.1 — Add `clue_mode` parameter to `ProbMinesweeperEnv`

Update `ProbMinesweeperEnv.__init__` signature:

```python
def __init__(
    self,
    width: int = 9,
    height: int = 9,
    distribution: str | MineDistribution = "correlated",
    distribution_kwargs: dict[str, Any] | None = None,
    obs_mode: str = "state",
    reward_config: RewardConfig | None = None,
    max_steps: int | None = None,
    render_mode: str | None = None,
    seed: int | None = None,
    clue_mode: str = "prob_sum",
) -> None:
    ...
```

Store:

```python
self.clue_mode = clue_mode
```

Pass it to `Board.create` in `reset`:

```python
self.board = Board.create(
    self.height,
    self.width,
    p_mines,
    clue_mode=self.clue_mode,
)
```

### Acceptance criteria

- `ProbMinesweeperEnv(clue_mode="prob_sum")` works as before.
- `ProbMinesweeperEnv(clue_mode="actual_count")` works.
- Invalid clue mode fails clearly.

---

### Task 2.2 — Update Gymnasium registration compatibility

Ensure the registered environment still works with:

```python
gym.make("ProbMinesweeper-v0", width=5, height=5)
```

And also:

```python
gym.make("ProbMinesweeper-v0", width=5, height=5, clue_mode="actual_count")
```

### Acceptance criteria

- Gymnasium env checker still passes.
- `gym.make(..., clue_mode="actual_count")` does not crash.

---

### Task 2.3 — Review observation bounds

Current observation space high is based on `_MAX_DISPLAY_VALUE = 8.0`, which still works because actual neighbouring mine count is also in `[0, 8]`.

No change should be required.

### Acceptance criteria

- Observation space remains valid for both clue modes.
- `env.observation_space.contains(obs)` passes after reset and step in both modes.

---

## Phase 3 — New reward mode for RL

### Task 3.1 — Add completion-focused reward config

In `prob_minesweeper/rewards.py`, add a new factory:

```python
@classmethod
def completion(cls) -> RewardConfig:
    """Reward focused on completing the board rather than local risk shaping."""
    return cls(
        reveal_reward_fn=lambda _p: 0.1,
        mine_penalty_fn=lambda _p: -1.0,
        win_bonus=10.0,
    )
```

Optional: if no-op should be penalized, extend `RewardConfig` to support `noop_penalty`.

Current `reward_for_reveal` returns `0.0` for `NOOP`. To keep changes small, leave this unchanged unless explicitly needed.

Recommended minimal version:

```text
safe reveal: +0.1
mine hit: -1.0
win bonus: +10.0
noop: 0.0
```

### Acceptance criteria

- Existing reward modes still work.
- `RewardConfig.completion()` exists.
- Tests cover safe, mine, win, and noop behavior for completion reward.

---

### Task 3.2 — Optional: add explicit noop penalty

Only implement this if it does not create too much churn.

Change dataclass:

```python
@dataclass(frozen=True)
class RewardConfig:
    reveal_reward_fn: Callable[[float], float]
    mine_penalty_fn: Callable[[float], float]
    win_bonus: float
    noop_penalty: float = 0.0
```

Update:

```python
if result == RevealResult.NOOP:
    return self.noop_penalty
```

Set:

```python
RewardConfig.completion(..., noop_penalty=-0.1)
```

### Acceptance criteria

- Existing reward factories preserve previous behavior because default `noop_penalty=0.0`.
- Completion mode can penalize invalid/no-op moves.

---

## Phase 4 — Tests for hidden-risk rules

### Task 4.1 — Add board tests for `actual_count`

Update `tests/test_board.py`.

Test cases:

1. `Board.neighbour_mine_count` returns correct count on a deterministic board.
2. `actual_count` reveal sets `display_value` to actual neighbouring mine count.
3. `prob_sum` reveal still sets `display_value` to probability sum.
4. Invalid clue mode raises `ValueError`.

Implementation idea:

Use a deterministic `p_mines` field with `0.0` and `1.0` values so sampled outcomes are deterministic:

```python
p_mines = np.array(
    [
        [1.0, 0.0, 1.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
```

Reveal center `(1, 1)`; actual neighbouring mine count should be `4`.

### Acceptance criteria

- Tests clearly distinguish `prob_sum` and `actual_count`.
- Existing tests still pass.

---

### Task 4.2 — Add environment tests for `clue_mode`

Update `tests/test_env.py`.

Test cases:

1. `ProbMinesweeperEnv(clue_mode="actual_count")` can reset and step.
2. Observation stays within bounds.
3. `gym.make("ProbMinesweeper-v0", clue_mode="actual_count")` works.
4. Invalid clue mode raises `ValueError`.

### Acceptance criteria

- Env tests pass.
- Gymnasium checker still passes for default mode.

---

### Task 4.3 — Add reward tests for completion mode

Update `tests/test_rewards.py`.

Test cases:

```text
completion safe reveal = 0.1
completion mine hit = -1.0
completion win = 10.1
completion noop = 0.0 or -0.1 depending on implementation
```

### Acceptance criteria

- Completion reward behavior is documented by tests.
- Existing reward tests still pass.

---

## Phase 5 — Agent semantics update

### Task 5.1 — Keep `MinRiskAgent` but label it as oracle in docs/UI

Do **not** remove `MinRiskAgent`.

In hidden-risk mode, it accesses:

```python
env.board.cell(row, col).p_mine
```

This is hidden information not available in `obs_mode="state"`.

Therefore, in app/report/docs, label it as:

```text
MinRiskAgent (oracle)
```

### Acceptance criteria

- Agent code can remain mostly unchanged.
- UI/report clearly states that MinRiskAgent has privileged access to hidden probability values.

---

### Task 5.2 — Ensure DQN trains without probability channel

Update DQN training code to use:

```python
obs_mode="state"
clue_mode="actual_count"
reward_config=RewardConfig.completion()
```

This is the main RL setting.

### Acceptance criteria

- `train_dqn.py` defaults to hidden-risk mode.
- CLI allows overriding clue mode and obs mode if useful.
- DQN model input shape matches `state` observation: `(H, W, 3)` flattened.

---

### Task 5.3 — Ensure DQN evaluation uses same mode as training

Update `evaluate_dqn.py` so defaults match training:

```text
width = 5
height = 5
distribution = correlated
obs_mode = state
clue_mode = actual_count
reward_config = completion
```

### Acceptance criteria

- Evaluation environment matches training environment by default.
- README warns that DQN model is tied to board size and observation shape.

---

## Phase 6 — DQN plan update for hidden-risk mode

### Task 6.1 — Update `experiments/train_dqn.py`

Default settings:

```text
--width 5
--height 5
--distribution correlated
--obs-mode state
--clue-mode actual_count
--reward-mode completion
--timesteps 500000
```

Recommended DQN hyperparameters:

```python
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    buffer_size=100_000,
    learning_starts=5_000,
    batch_size=64,
    gamma=0.98,
    exploration_fraction=0.4,
    exploration_final_eps=0.05,
    target_update_interval=1_000,
    train_freq=4,
    gradient_steps=1,
    verbose=1,
    seed=args.seed,
    device="cpu",
)
```

### Acceptance criteria

- Training script can train hidden-risk DQN.
- Existing DQN train path still works if user passes old modes explicitly.

---

### Task 6.2 — Update `experiments/evaluate_dqn.py`

The evaluation script should print the environment configuration:

```text
width
height
distribution
obs_mode
clue_mode
reward_mode
```

### Acceptance criteria

- Evaluation output is report-ready.
- It is clear which rule variant produced the numbers.

---

### Task 6.3 — Update DQNAgent shape checks

If `DQNAgent` loads a model trained with `obs_mode="state"` and the app uses `obs_mode="state+prob"`, shapes may mismatch.

Add one of:

Option A, simple:

```text
DQN app mode always creates env with obs_mode="state" when DQN is selected.
```

Option B, robust:

```python
expected_obs_size = self.model.observation_space.shape[0]
actual_obs_size = int(np.prod(obs.shape))
if expected_obs_size != actual_obs_size:
    raise ValueError(...)
```

Recommended: implement Option B and configure app correctly.

### Acceptance criteria

- Shape mismatch raises a clear error.
- Streamlit does not silently pass wrong observation shape to DQN.

---

## Phase 7 — Streamlit frontend updates

### Task 7.1 — Add game rule controls

In sidebar or settings tab, add:

```text
Clue mode:
  - Probability sum
  - Actual mine count

Reward mode:
  - Risk-adjusted
  - Completion
```

Map UI labels to code:

```python
clue_mode = {
    "Probability sum": "prob_sum",
    "Actual mine count": "actual_count",
}[selected_label]

reward_config = {
    "Risk-adjusted": RewardConfig.risk_adjusted(),
    "Completion": RewardConfig.completion(),
}[selected_label]
```

### Acceptance criteria

- User can start a game in either clue mode.
- UI displays which mode is active.
- Defaults should favor the new RL mode:
  - `actual_count`
  - `completion`

---

### Task 7.2 — Add explanation warning for hidden probabilities

In manual play UI, keep optional probability display for human/debugging:

```text
Show hidden probabilities
```

But add warning:

```text
Hidden probabilities are shown only for human/debug mode. The DQN agent is trained in obs_mode="state" and does not observe these values.
```

### Acceptance criteria

- Human can inspect probabilities if desired.
- UI does not imply DQN sees hidden probabilities.

---

### Task 7.3 — Rename MinRisk in UI

In agent selector, show:

```text
Random
DQN
Min-risk (oracle)
```

### Acceptance criteria

- It is clear that MinRisk is not a fair baseline in hidden-risk mode.

---

### Task 7.4 — Update benchmark tab

Benchmark should support two comparison groups.

Group A — original/full-information mode:

```text
obs_mode = state+prob
clue_mode = prob_sum
reward = risk-adjusted
Agents: Random, Min-risk, DQN if trained for this mode
```

Group B — hidden-risk RL mode:

```text
obs_mode = state
clue_mode = actual_count
reward = completion
Agents: Random, DQN, Min-risk oracle
```

If implementing both groups is too much, implement only Group B.

### Acceptance criteria

- Benchmark output clearly labels the rule mode.
- DQN is evaluated only in compatible observation shape.
- Missing DQN model does not crash the app.

---

### Task 7.5 — Update model explanation tab

Add explanation:

```text
In the hidden-risk variant, the agent does not observe p_mine. Safe reveals produce actual neighbouring mine counts. Therefore, the agent must act under uncertainty and use revealed clues to choose future actions.
```

Add math:

```text
M_i ~ Bernoulli(p_i)

clue_i = Σ_{j ∈ N(i)} M_j

a_t = argmax_a Q(s_t, a)

Q target:
y = r_t + γ max_a' Q(s_{t+1}, a')
```

### Acceptance criteria

- Streamlit model tab explains why RL is now meaningful.
- It distinguishes full-information mode from hidden-risk mode.

---

## Phase 8 — CLI and evaluation integration

### Task 8.1 — Extend CLI benchmark options

Update `prob_minesweeper/cli.py` benchmark command to accept:

```text
--clue-mode prob_sum|actual_count
--reward-mode risk_adjusted|sparse|uniform|completion
--obs-mode state|state+prob
```

Use helper function to map reward mode string to `RewardConfig`.

### Acceptance criteria

- Existing CLI benchmark still works with defaults.
- New benchmark can run:

```bash
uv run prob-minesweeper benchmark \
  --episodes 100 \
  --width 5 \
  --height 5 \
  --clue-mode actual_count \
  --reward-mode completion
```

---

### Task 8.2 — Update evaluation helper

If `prob_minesweeper/evaluation.py` exists, update `evaluate_agent` to accept:

```python
obs_mode: str = "state"
clue_mode: str = "actual_count"
reward_config: RewardConfig | None = None
```

Default should be hidden-risk mode if this helper is mainly for RL evaluation.

### Acceptance criteria

- Evaluation can run both old and new variants.
- Agent comparison code clearly controls environment config.

---

## Phase 9 — Documentation/report updates

### Task 9.1 — Update README

Add section:

```text
## Hidden-risk RL mode
```

Explain:

```text
The RL-focused variant hides mine probabilities from the agent and uses actual neighbouring mine counts as clues. This creates a sequential decision problem under uncertainty.
```

Add example:

```bash
uv run streamlit run app.py
uv run python experiments/train_dqn.py --timesteps 500000
uv run python experiments/evaluate_dqn.py --episodes 500
```

### Acceptance criteria

- README clearly explains both original and hidden-risk modes.
- Commands are accurate.

---

### Task 9.2 — Update report

In `report/report.md`, add or update:

```text
## Wariant hidden-risk
## Dlaczego RL ma sens w tym wariancie
## Porównanie agentów
```

Include explanation:

```text
W pierwotnym wariancie state+prob agent znał prawdopodobieństwa min, przez co heurystyka MinRisk była bardzo silna. W wariancie hidden-risk prawdopodobieństwa są ukryte, a odsłonięcie pola dostarcza informacji w postaci rzeczywistej liczby min w sąsiedztwie. Decyzja agenta wpływa więc zarówno na nagrodę natychmiastową, jak i na informację dostępną w kolejnych stanach.
```

### Acceptance criteria

- Report explicitly justifies the rule change.
- Report does not pretend MinRisk is a fair baseline in hidden-risk mode.

---

### Task 9.3 — Update defense notes

In `report/defense_notes.md`, add answers:

```text
Why did we add hidden-risk mode?
Why is MinRiskAgent an oracle?
What information does DQN observe?
What does the clue represent in actual_count mode?
Why is this now a sequential RL problem?
```

### Acceptance criteria

- Defense notes prepare the student to explain the rule change.

---

## Phase 10 — Final validation

### Task 10.1 — Run full tests

```bash
uv run pytest -q
```

### Acceptance criteria

- All tests pass.

---

### Task 10.2 — Manual hidden-risk game check

Run Streamlit:

```bash
uv run streamlit run app.py
```

Manual checks:

1. Select `Actual mine count`.
2. Select `Completion` reward.
3. Start a 5x5 game.
4. Click cells manually.
5. Safe cells show integer clues `0..8`.
6. Mines end the game.
7. Total reward updates.
8. MinRisk is labeled oracle.

### Acceptance criteria

- Hidden-risk game is playable and understandable.

---

### Task 10.3 — DQN smoke training

Run:

```bash
uv run python experiments/train_dqn.py \
  --timesteps 10000 \
  --width 5 \
  --height 5 \
  --clue-mode actual_count \
  --obs-mode state \
  --reward-mode completion \
  --output models/test_hidden_risk_dqn.zip
```

Then:

```bash
uv run python experiments/evaluate_dqn.py \
  --model models/test_hidden_risk_dqn.zip \
  --episodes 10 \
  --width 5 \
  --height 5 \
  --clue-mode actual_count \
  --obs-mode state \
  --reward-mode completion
```

### Acceptance criteria

- Training creates model.
- Evaluation runs.
- No invalid action crashes occur.

---

### Task 10.4 — DQN real training for report

Recommended:

```bash
uv run python experiments/train_dqn.py \
  --timesteps 300000 \
  --width 5 \
  --height 5 \
  --clue-mode actual_count \
  --obs-mode state \
  --reward-mode completion \
  --output models/dqn_hidden_risk_300k.zip

uv run python experiments/train_dqn.py \
  --timesteps 500000 \
  --width 5 \
  --height 5 \
  --clue-mode actual_count \
  --obs-mode state \
  --reward-mode completion \
  --output models/dqn_hidden_risk_500k.zip
```

Evaluate:

```bash
uv run python experiments/evaluate_dqn.py \
  --model models/dqn_hidden_risk_500k.zip \
  --episodes 500
```

### Acceptance criteria

- Report has benchmark data for Random, DQN, and MinRisk oracle.
- DQN should beat Random in at least some metric.
- It is acceptable if DQN does not beat MinRisk oracle.

---

## Recommended implementation order

Implement in this order:

```text
1. Phase 1 — Board clue_mode and actual_count
2. Phase 2 — Env clue_mode parameter
3. Phase 3 — Completion reward
4. Phase 4 — Tests
5. Phase 5 — Agent semantics update
6. Phase 6 — DQN hidden-risk defaults
7. Phase 7 — Streamlit updates
8. Phase 8 — CLI/evaluation integration
9. Phase 9 — Docs/report
10. Phase 10 — Validation
```

Do not start by tuning DQN. First make the environment rules correct and testable.

---

## Minimal acceptable scope

If time is limited, implement only:

```text
Board clue_mode actual_count
Env clue_mode parameter
RewardConfig.completion()
Tests for actual_count and completion reward
DQN train/evaluate defaults using obs_mode=state + clue_mode=actual_count
Streamlit controls for clue/reward mode
Docs explaining MinRisk as oracle
```

This is enough to make the RL setup much more defensible.

---

## Non-goals

Do not implement unless explicitly requested:

- Full belief-state Bayesian solver.
- Exact Minesweeper constraint solver.
- Custom DQN from scratch.
- Separate FastAPI backend.
- Database persistence.
- Complex CNN policy.
- Large 9x9 training as default.

---

## Expected final code changes

```text
prob_minesweeper/
  board.py                 # clue_mode, neighbour_mine_count, clue_value
  env.py                   # clue_mode env parameter
  rewards.py               # completion reward
  cli.py                   # optional clue/reward mode args
  evaluation.py            # hidden-risk env config support
  agents/
    min_risk_agent.py      # docstring says oracle in hidden-risk mode
    dqn_agent.py           # shape checks

experiments/
  train_dqn.py             # hidden-risk defaults
  evaluate_dqn.py          # hidden-risk defaults

tests/
  test_board.py            # actual_count tests
  test_env.py              # clue_mode tests
  test_rewards.py          # completion tests
  test_evaluation.py       # hidden-risk evaluation tests if applicable

app.py                     # Streamlit controls and explanations
README.md
report/report.md
report/defense_notes.md
```

---

## Key explanation for report and defense

Use this wording:

```text
The original state+prob variant gives the agent direct access to mine probabilities, which makes a simple MinRisk heuristic very strong. To make reinforcement learning meaningful, we introduced a hidden-risk variant. The mine probabilities are still used to generate the board, but they are not observed by the DQN agent. Revealing a safe cell returns the actual number of neighbouring mines, so each action changes both the reward and the information available in future states. This creates a sequential decision problem under uncertainty.
```

---

## Definition of done

The hidden-risk variant is complete when:

- `clue_mode="actual_count"` exists and is tested.
- Actual clues show neighbouring mine counts, not probability sums.
- `obs_mode="state"` hides probabilities from DQN.
- `RewardConfig.completion()` exists and is tested.
- DQN train/evaluate scripts default to hidden-risk mode.
- Streamlit can run hidden-risk games.
- Streamlit labels MinRisk as oracle.
- Benchmark can compare Random, DQN, and MinRisk oracle.
- README and report explain why this rule change makes RL meaningful.
