# Probabilistic Minesweeper — Implementation Plan for Codex

## Goal

Upgrade the existing `prob-minesweeper` codebase into a complete student project targeting grade **4.0**.

The project should remain in the **reinforcement learning** area. The current codebase already provides a Gymnasium-compatible probabilistic Minesweeper environment, board logic, distributions, rewards, CLI, rendering, and tests. The required changes should add:

1. a usable Streamlit frontend,
2. agent functionality,
3. benchmark/evaluation tooling,
4. documentation and report material,
5. enough integration polish to present the project as an application, not only a library.

The target architecture is:

```text
Streamlit frontend
  ├── manual game UI
  ├── agent demo UI
  ├── benchmark UI
  └── model description UI

Python backend / domain package
  ├── prob_minesweeper.env.ProbMinesweeperEnv
  ├── prob_minesweeper.board.Board
  ├── prob_minesweeper.distributions
  ├── prob_minesweeper.rewards
  └── prob_minesweeper.agents
```

Streamlit is accepted as the frontend layer. The existing `prob_minesweeper` package is the backend/domain layer. Do **not** add a separate FastAPI backend unless explicitly requested later.

---

## Phase 0 — Baseline verification and cleanup

### Task 0.1 — Run the existing test suite

**Goal:** Confirm the current codebase is stable before adding features.

**Actions:**

- Run:

```bash
uv sync --dev
uv run pytest -q
uv run pytest --collect-only
```

- Record the number of collected tests.
- Fix only blocking issues required for the current suite to pass.

**Acceptance criteria:**

- `uv run pytest -q` passes.
- No new feature work starts before this is true.

---

### Task 0.2 — Fix obvious project metadata/documentation inconsistencies

**Goal:** Remove AI-generated or inconsistent leftovers.

**Actions:**

- In `README.md`, fix setup instructions that say `cd boom-boom`; use the actual project directory name, likely `prob-minesweeper`.
- Check `.ai/AGENTS.md` and README for consistency with implementation.
- The current code samples mine outcomes in `Board.new_episode()`, not lazily at reveal time. Update documentation to say:

```text
Hidden mine outcomes are sampled at episode start from Bernoulli(p_mine). The agent does not observe those outcomes until revealing cells.
```

**Acceptance criteria:**

- README no longer contains wrong project names.
- Documentation matches the actual implementation.

---

### Task 0.3 — Add Streamlit dependency

**Goal:** Prepare project for frontend implementation.

**Actions:**

- Add `streamlit` to project dependencies in `pyproject.toml`.
- Keep `pytest` as a dev dependency.
- After editing dependencies, refresh lockfile if the project uses `uv`.

Suggested dependency:

```toml
dependencies = [
    "gymnasium>=0.29",
    "numpy>=1.24",
    "scipy>=1.11",
    "streamlit>=1.35",
]
```

**Acceptance criteria:**

- `uv sync --dev` succeeds.
- `uv run streamlit --version` succeeds.

---

## Phase 1 — Agent layer

### Task 1.1 — Create agents package

**Goal:** Add a clean place for reusable agents.

**Files to create:**

```text
prob_minesweeper/agents/__init__.py
prob_minesweeper/agents/base.py
prob_minesweeper/agents/random_agent.py
prob_minesweeper/agents/min_risk_agent.py
```

**Implementation requirements:**

Create a minimal base protocol or abstract class in `base.py`:

```python
from typing import Protocol
import numpy as np

class Agent(Protocol):
    name: str

    def select_action(self, obs: np.ndarray, info: dict, env) -> int:
        ...
```

The `env` argument is allowed because the project is educational and agents may need access to board probabilities.

**Acceptance criteria:**

- Agents can be imported from `prob_minesweeper.agents`.
- No circular imports.

---

### Task 1.2 — Implement `RandomAgent`

**Goal:** Provide a baseline agent.

**Behavior:**

- Reads `info["action_mask"]`.
- Selects one valid action uniformly at random.
- Uses `numpy.random.Generator`.

**Suggested API:**

```python
class RandomAgent:
    name = "Random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, info: dict, env) -> int:
        valid = np.flatnonzero(info["action_mask"])
        if len(valid) == 0:
            raise RuntimeError("No valid actions available")
        return int(self.rng.choice(valid))
```

**Acceptance criteria:**

- Random agent never selects an already revealed cell.
- Add tests for valid-action behavior.

---

### Task 1.3 — Implement `MinRiskAgent`

**Goal:** Add a stronger deterministic baseline.

**Behavior:**

- Selects a valid hidden cell with the lowest `p_mine`.
- Uses `env.board.cell(row, col).p_mine`.
- If multiple cells have the same risk, choose the first or break ties randomly.

**Suggested API:**

```python
class MinRiskAgent:
    name = "Min-risk"

    def select_action(self, obs: np.ndarray, info: dict, env) -> int:
        ...
```

**Acceptance criteria:**

- On a known board, the agent selects the hidden cell with minimum probability.
- Agent respects `action_mask`.

---

### Task 1.4 — Add agent tests

**Goal:** Prevent regressions.

**Files to create:**

```text
tests/test_agents.py
```

**Test cases:**

- `RandomAgent` selects only masked-valid actions.
- `MinRiskAgent` selects the lowest-probability valid cell.
- `MinRiskAgent` ignores already revealed low-probability cells.
- Both agents raise a clear error if no valid actions exist.

**Acceptance criteria:**

- `uv run pytest tests/test_agents.py -q` passes.
- Full test suite passes.

---

## Phase 2 — Evaluation and benchmarking

### Task 2.1 — Create reusable evaluation function

**Goal:** Evaluate any agent over many episodes.

**Files to create:**

```text
prob_minesweeper/evaluation.py
```

**Required dataclass:**

```python
@dataclass(frozen=True)
class EvaluationResult:
    agent_name: str
    episodes: int
    wins: int
    losses: int
    truncated: int
    total_reward: float
    total_steps: int

    @property
    def win_rate(self) -> float: ...

    @property
    def loss_rate(self) -> float: ...

    @property
    def mean_reward(self) -> float: ...

    @property
    def mean_steps(self) -> float: ...
```

**Required function:**

```python
def evaluate_agent(
    agent,
    *,
    episodes: int,
    width: int = 5,
    height: int = 5,
    distribution: str = "correlated",
    distribution_kwargs: dict | None = None,
    obs_mode: str = "state+prob",
    seed: int | None = None,
) -> EvaluationResult:
    ...
```

**Acceptance criteria:**

- Works with `RandomAgent`.
- Works with `MinRiskAgent`.
- Correctly counts wins, losses, truncations, reward, and steps.
- Deterministic with fixed seed when the agent is deterministic.

---

### Task 2.2 — Add comparison helper

**Goal:** Make it easy for Streamlit and CLI/report scripts to compare agents.

**Add function:**

```python
def compare_agents(
    agents: list,
    *,
    episodes: int,
    width: int,
    height: int,
    distribution: str,
    distribution_kwargs: dict | None = None,
    seed: int | None = None,
) -> list[EvaluationResult]:
    ...
```

**Acceptance criteria:**

- Returns one result per agent.
- Does not mutate shared environment state between agents.
- Uses reproducible seeds.

---

### Task 2.3 — Add evaluation tests

**File:**

```text
tests/test_evaluation.py
```

**Test cases:**

- On `ConstantDistribution(p=0.0)`, agents should win all episodes.
- On `ConstantDistribution(p=1.0)`, agents should lose all episodes.
- `wins + losses + truncated == episodes`.
- `mean_reward` and `win_rate` properties behave correctly.

**Acceptance criteria:**

- Evaluation tests pass.
- Full test suite passes.

---

## Phase 3 — Streamlit frontend

### Task 3.1 — Create app entrypoint

**Goal:** Add a frontend application.

**Files to create:**

```text
app.py
```

Do not place Streamlit app inside the package unless there is a strong reason. A root-level `app.py` is simple for student presentation.

**Run command:**

```bash
uv run streamlit run app.py
```

**Acceptance criteria:**

- Streamlit app starts successfully.
- App imports the local `prob_minesweeper` package.

---

### Task 3.2 — Implement Streamlit session initialization

**Goal:** Preserve game state across Streamlit reruns.

**Implementation requirements:**

Use `st.session_state` for:

```text
env
obs
info
total_reward
last_reward
terminated
truncated
message
step_count
selected_agent
```

Create helper functions inside `app.py` or in a separate `prob_minesweeper/ui_state.py` if the file grows too large:

```python
def create_env(width, height, distribution, seed, show_probabilities=True): ...
def new_game(): ...
def reveal(action: int): ...
def reset_session_state(): ...
```

**Acceptance criteria:**

- Starting a new game resets board, reward, step count, and status.
- Clicking a cell updates the same game instead of creating a new game each rerun.

---

### Task 3.3 — Add sidebar controls

**Goal:** Make board configuration visible and adjustable.

**Sidebar controls:**

- Board width: integer, default `5`, range `2..12`.
- Board height: integer, default `5`, range `2..12`.
- Distribution: selectbox with `constant`, `uniform`, `correlated`.
- Seed: integer.
- Show hidden probabilities: checkbox, default `True`.
- New game button.

For `constant` distribution, add:

- Mine probability `p`: slider `0.0..1.0`, default `0.2`.

For `uniform` distribution, add:

- `low`: slider `0.0..1.0`, default `0.0`.
- `high`: slider `0.0..1.0`, default `1.0`.
- Ensure `low <= high`.

For `correlated` distribution, add:

- `sigma`: slider or number input, default `2.0`.
- `scale`: slider or number input, default `1.0`.

**Acceptance criteria:**

- Sidebar controls affect newly created games.
- Invalid uniform bounds are prevented or corrected.

---

### Task 3.4 — Render clickable Minesweeper board

**Goal:** Allow manual play.

**Implementation requirements:**

- Use `st.columns(width)` for each row.
- Each cell is a `st.button`.
- Button key must be stable and unique, e.g. `cell_{row}_{col}_{game_id}`.
- Revealed cells and terminal-game cells should be disabled.
- Already revealed cells should be disabled using `info["action_mask"]`.

**Cell labels:**

```text
Hidden cell without probabilities: ⬜
Hidden cell with probabilities: 0.23
Revealed safe empty cell: ·
Revealed safe clue cell: 1.7
Revealed mine: 💣
```

**Acceptance criteria:**

- User can play a full game manually.
- Clicking a valid hidden cell calls `env.step(action)`.
- Revealed cells cannot be clicked again.
- Terminal game disables the board.

---

### Task 3.5 — Display game status and metrics

**Goal:** Make the interaction explainable during presentation.

**Display:**

- Current status message.
- Last reward.
- Total reward.
- Step count.
- Win/loss/truncation status.
- Optional board parameters.

Suggested Streamlit widgets:

```python
st.metric("Last reward", ...)
st.metric("Total reward", ...)
st.metric("Steps", ...)
```

**Acceptance criteria:**

- After each move, metrics update correctly.
- On win/loss, the status is clear.

---

### Task 3.6 — Add agent demo controls

**Goal:** Show computational intelligence in the UI.

**Controls:**

- Select agent: `Random`, `Min-risk`.
- Button: `Agent move`.
- Button: `Run agent until terminal`.

**Behavior:**

- `Agent move` selects and performs one action.
- `Run agent until terminal` repeatedly performs actions until win/loss/truncation or no valid actions remain.
- Add a safe maximum loop guard, e.g. `height * width * 2`.

**Acceptance criteria:**

- User can watch an agent play the current game.
- Agent cannot make invalid moves.
- App does not freeze on terminal state.

---

### Task 3.7 — Add benchmark tab

**Goal:** Show quantitative comparison for report and defense.

Use Streamlit tabs:

```python
tab_game, tab_benchmark, tab_model = st.tabs([
    "Game",
    "Benchmark",
    "Model"
])
```

Benchmark tab should include:

- Episodes input, default `100`.
- Board size controls or reuse sidebar settings.
- Distribution controls or reuse sidebar settings.
- Button: `Run benchmark`.
- Compare at least:
  - `RandomAgent`
  - `MinRiskAgent`

Display results as a table with columns:

```text
Agent
Episodes
Wins
Losses
Truncated
Win rate
Mean reward
Mean steps
```

Optional:

- Add a simple bar chart for win rate.
- Add a simple bar chart for mean reward.

**Acceptance criteria:**

- Benchmark runs from the UI.
- Results are stored in `st.session_state`.
- Results can be used in the report.

---

### Task 3.8 — Add model description tab

**Goal:** Make the mathematical model visible in the app.

Add a tab or section explaining:

- Problem area: reinforcement learning.
- State `s_t`.
- Action `a_t`.
- Transition stochasticity.
- Reward function.
- Goal: maximize expected discounted reward.
- Agent comparison: random vs min-risk.

Keep it concise but explicit.

**Acceptance criteria:**

- App contains enough text to explain the project during defense.
- Mathematical notation is readable in Streamlit using `st.markdown` and/or `st.latex`.

---

## Phase 4 — Optional existing RL model for grade 4.0

This phase is recommended if time allows. The formal 4.0 target says an existing model may be used or modified. Adding Stable-Baselines3 DQN/PPO strengthens the project. If time is short, finish Phases 0–3 and Phase 5 first.

### Task 4.1 — Add optional Stable-Baselines3 dependency

**Goal:** Enable use of an existing RL model.

**Actions:**

- Add optional dependency group if preferred:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]
rl = [
    "stable-baselines3>=2.3",
]
```

Or add directly to dependencies if simplicity is preferred.

**Acceptance criteria:**

- Project installs with RL dependencies.
- Existing tests still pass.

---

### Task 4.2 — Implement training script

**Files to create:**

```text
experiments/train_sb3.py
experiments/evaluate_sb3.py
```

**Goal:** Train an existing DQN or PPO model on the environment.

Suggested approach:

- Start with small board, e.g. `5x5`.
- Use `obs_mode="state+prob"`.
- Flatten observation if required by the model.
- Train for a small number of timesteps suitable for demo.
- Save model to:

```text
models/dqn_prob_minesweeper.zip
```

**Acceptance criteria:**

- Training script runs without crashing.
- Model file is saved.
- Evaluation script can load the model and play episodes.

---

### Task 4.3 — Add SB3 model to benchmark comparison

**Goal:** Show existing RL model as the 4.0 model component.

**Actions:**

- Add wrapper agent:

```text
prob_minesweeper/agents/sb3_agent.py
```

- It should load a trained model and expose:

```python
select_action(obs, info, env) -> int
```

- Ensure selected action respects `action_mask`.
- If the model selects an invalid action, fallback to best valid action or random valid action.

**Acceptance criteria:**

- Streamlit can include `SB3 DQN` in benchmark if model file exists.
- App does not crash if model file is missing; show a clear message instead.

---

## Phase 5 — Reporting and documentation

### Task 5.1 — Update README with full app instructions

**Goal:** Make the project runnable by the instructor.

Add sections:

```text
## Run Streamlit app
## Agents
## Benchmarking
## Mathematical model
## Project scope for grade 4.0
```

Include commands:

```bash
uv sync --dev
uv run streamlit run app.py
uv run pytest -q
```

If SB3 is implemented:

```bash
uv sync --extra rl --dev
uv run python experiments/train_sb3.py
```

**Acceptance criteria:**

- README explains both CLI and Streamlit usage.
- README no longer describes only the environment library.

---

### Task 5.2 — Create report outline

**Files to create:**

```text
report/report.md
```

**Required sections:**

```text
# Probabilistyczny Saper jako środowisko uczenia ze wzmocnieniem

## 1. Cel projektu
## 2. Opis problemu
## 3. Model matematyczny
## 4. Architektura aplikacji
## 5. Implementacja
## 6. Agenci i metody porównawcze
## 7. Eksperymenty
## 8. Wyniki
## 9. Wnioski
## 10. Instrukcja uruchomienia
```

**Important content:**

- State that the selected area is reinforcement learning.
- Define MDP:

```text
S — state space
A — action space
P — transition function
R — reward function
γ — discount factor
```

- Include reward formula:

```text
r_t =
  1 - p_a        for safe reveal
 -p_a            for mine hit
  1 - p_a + B    for winning move
  0              for no-op
```

- Explain distributions:
  - constant,
  - uniform,
  - correlated.

- Explain Streamlit frontend.
- Explain backend/domain package.
- Include benchmark table placeholders.

**Acceptance criteria:**

- Report outline exists.
- It is specific to this project, not generic RL text.

---

### Task 5.3 — Add screenshots directory placeholder

**Goal:** Prepare report assets.

**Files/directories:**

```text
report/screenshots/.gitkeep
```

**Acceptance criteria:**

- Directory exists.
- README/report mention that screenshots from the Streamlit app should be inserted before submission.

---

### Task 5.4 — Add defense notes

**Files to create:**

```text
report/defense_notes.md
```

**Content should answer:**

1. What is the state in the MDP?
2. What is the action?
3. What is the reward function?
4. Why is the environment stochastic?
5. What does `action_mask` do?
6. How are mine probabilities generated?
7. What is the difference between RandomAgent and MinRiskAgent?
8. If SB3 is implemented: how does the RL model select actions?
9. What is the frontend?
10. What is the backend?
11. What did we add compared to the original environment-only codebase?

**Acceptance criteria:**

- Notes are concise.
- They are written in Polish, suitable for oral defense.

---

## Phase 6 — Integration polish

### Task 6.1 — Add Makefile or task commands

**Goal:** Make common commands obvious.

**File to create:**

```text
Makefile
```

Suggested targets:

```makefile
install:
	uv sync --dev

test:
	uv run pytest -q

app:
	uv run streamlit run app.py

benchmark:
	uv run prob-minesweeper benchmark --episodes 100
```

If SB3 is implemented:

```makefile
train:
	uv run python experiments/train_sb3.py
```

**Acceptance criteria:**

- `make test` works.
- `make app` starts Streamlit.

---

### Task 6.2 — Add smoke test for Streamlit imports

**Goal:** Catch broken imports without launching browser UI.

**File:**

```text
tests/test_app_imports.py
```

**Test:**

```python
def test_app_imports():
    import app  # noqa: F401
```

If importing `app.py` immediately launches Streamlit logic, refactor app into:

```text
app.py
prob_minesweeper_app/main.py
```

with:

```python
def main() -> None:
    ...
```

Then test:

```python
from prob_minesweeper_app.main import main
assert callable(main)
```

**Acceptance criteria:**

- App import test passes.
- Importing app does not create an environment unexpectedly unless guarded.

---

### Task 6.3 — Final full validation

**Goal:** Verify project is ready to present.

**Commands:**

```bash
uv sync --dev
uv run pytest -q
uv run streamlit run app.py
```

Manual checks:

- Start new game.
- Click several cells.
- Win/loss status appears.
- Agent move works.
- Run-agent-until-terminal works.
- Benchmark works.
- Model tab displays mathematical explanation.
- README instructions are accurate.

**Acceptance criteria:**

- Full test suite passes.
- Streamlit demo is usable.
- No obvious AI leftovers in README/report.
- Report files exist.

---

## Recommended minimal scope for grade 4.0

If time is limited, implement these first:

```text
Phase 0
Phase 1
Phase 2
Phase 3
Phase 5.1
Phase 5.2
Phase 5.4
Phase 6.3
```

Phase 4 with Stable-Baselines3 is recommended but optional if the instructor accepts heuristic/baseline agents plus the existing Gymnasium environment as sufficient. For a safer 4.0, implement Phase 4.

---

## Non-goals

Do not implement these unless explicitly requested later:

- Full React frontend.
- Separate FastAPI backend.
- User accounts.
- Database persistence.
- Multiplayer.
- Classic exact Minesweeper mode.
- Complex custom DQN from scratch; this would target 5.0, not 4.0.

---

## Coding constraints

- Keep existing public APIs backward compatible when possible.
- Do not break existing CLI commands.
- Keep tests fast.
- Prefer small board sizes in demos and tests.
- Avoid hardcoding absolute paths.
- Use type hints in new modules.
- Avoid large training runs in tests.
- Do not commit trained model binaries unless explicitly required; provide scripts and optional `models/.gitkeep`.

---

## Expected final repository shape

```text
prob_minesweeper/
  __init__.py
  board.py
  cli.py
  distributions.py
  env.py
  evaluation.py
  rendering.py
  rewards.py
  agents/
    __init__.py
    base.py
    random_agent.py
    min_risk_agent.py
    sb3_agent.py              # optional

app.py

experiments/
  train_sb3.py                # optional
  evaluate_sb3.py             # optional

models/
  .gitkeep

report/
  report.md
  defense_notes.md
  screenshots/
    .gitkeep

tests/
  test_agents.py
  test_app_imports.py
  test_board.py
  test_cli.py
  test_distributions.py
  test_env.py
  test_evaluation.py
  test_rendering.py
  test_rewards.py

README.md
pyproject.toml
uv.lock
Makefile
```

---

## Definition of done

The project is ready for submission when:

- Streamlit app provides a playable probabilistic Minesweeper board.
- User can play manually.
- User can run at least RandomAgent and MinRiskAgent.
- Benchmark comparison is available in the UI.
- Existing test suite plus new tests pass.
- README explains how to run the app.
- Report outline contains a project-specific mathematical model.
- Defense notes explain the code and math in Polish.
- Documentation clearly states that this is a reinforcement learning environment/application.
