# DQN Implementation Plan for Probabilistic Minesweeper

## Purpose

Add an existing reinforcement learning model based on **Deep Q-Network (DQN)** to the current `prob-minesweeper` project.

This is intended to strengthen the project for a **4.0 grade** by showing something more advanced than:

```text
always choose the hidden cell with the lowest p_mine
```

The DQN model should learn an action-value function:

```text
Q(s, a)
```

where:

- `s` is the current board observation,
- `a` is the selected cell,
- `Q(s, a)` estimates the long-term expected reward of selecting that cell.

The final comparison should include:

```text
RandomAgent vs MinRiskAgent vs DQNAgent
```

---

## High-level target behavior

The final implementation should allow:

1. training a DQN model on `ProbMinesweeperEnv`,
2. saving the trained model to disk,
3. loading the trained model,
4. evaluating it over many episodes,
5. comparing it against RandomAgent and MinRiskAgent,
6. using it inside the Streamlit app.

---

## Important design decisions

### DQN is suitable because the action space is discrete

The environment action is:

```text
a ∈ {0, 1, ..., H * W - 1}
```

Each action means revealing one cell. This fits DQN, which is intended for discrete action spaces.

---

### Use Stable-Baselines3 DQN as an existing model

For grade 4.0, use an existing DQN implementation instead of writing a custom DQN from scratch.

Recommended library:

```text
stable-baselines3
```

Do not implement a full custom neural network training loop unless the target changes to grade 5.0.

---

### Start with small boards

Recommended initial training setup:

```text
width = 5
height = 5
distribution = "correlated"
obs_mode = "state+prob"
```

Do not start with `9x9`. It increases action space size and makes training slower and less stable.

---

### Important limitation: action masking

The environment already exposes:

```python
info["action_mask"]
```

This tells which cells are still legal to reveal.

Stable-Baselines3 DQN does not automatically use this mask. Therefore:

- during training, the model may sometimes choose already revealed cells,
- during evaluation/demo, the wrapper agent must prevent invalid moves.

For this project, implement a safe evaluation-time mask:

```text
If DQN predicts an invalid action:
    choose a valid fallback action
```

Recommended fallback:

```text
MinRiskAgent
```

This keeps the demo stable and easy to explain.

Optional advanced approach:

```text
Extract Q-values from the DQN policy, set invalid actions to -inf, then choose argmax over valid actions.
```

This is better, but more implementation work.

---

## Phase 0 — Dependency setup

### Task 0.1 — Add optional RL dependency group

Update `pyproject.toml`.

Recommended:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]
rl = [
    "stable-baselines3>=2.3",
]
```

If simplicity is preferred, add Stable-Baselines3 directly to main dependencies:

```toml
dependencies = [
    "gymnasium>=0.29",
    "numpy>=1.24",
    "scipy>=1.11",
    "streamlit>=1.35",
    "stable-baselines3>=2.3",
]
```

Preferred approach: use optional `rl` extra.

### Acceptance criteria

- `uv sync --dev --extra rl` succeeds.
- `uv run python -c "from stable_baselines3 import DQN"` succeeds.
- Existing tests still pass.

---

### Task 0.2 — Add models directory

Create:

```text
models/
  .gitkeep
```

Do not commit large trained models unless explicitly required.

### Acceptance criteria

- `models/` exists.
- Git can track the directory through `.gitkeep`.

---

## Phase 1 — Observation wrapper

### Task 1.1 — Create wrapper module

Create:

```text
prob_minesweeper/wrappers.py
```

Add a wrapper that flattens the board observation for `MlpPolicy`.

Current observation shape:

```text
(height, width, channels)
```

DQN with `MlpPolicy` should receive:

```text
(height * width * channels,)
```

### Implementation sketch

```python
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlattenObservationWrapper(gym.ObservationWrapper):
    """Flatten H x W x C board observations into a single vector."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        flat_size = int(np.prod(env.observation_space.shape))
        self.observation_space = spaces.Box(
            low=0.0,
            high=float(env.observation_space.high.max()),
            shape=(flat_size,),
            dtype=np.float32,
        )

    def observation(self, observation: np.ndarray) -> np.ndarray:
        return observation.astype(np.float32).reshape(-1)
```

### Acceptance criteria

- Wrapper works with `ProbMinesweeperEnv`.
- Wrapped observation shape is one-dimensional.
- Wrapped observation remains inside `observation_space`.

---

### Task 1.2 — Add wrapper tests

Create or update:

```text
tests/test_wrappers.py
```

Test cases:

1. Wrapped observation has shape `(H * W * C,)`.
2. Wrapped observation dtype is `np.float32`.
3. Wrapped observation space contains the flattened observation.
4. `step()` still works after wrapping.

### Acceptance criteria

- `uv run pytest tests/test_wrappers.py -q` passes.
- Full test suite passes.

---

## Phase 2 — Training script

### Task 2.1 — Create experiments directory

Create:

```text
experiments/
  train_dqn.py
  evaluate_dqn.py
```

---

### Task 2.2 — Implement `make_dqn_env`

In `experiments/train_dqn.py`, add a helper:

```python
def make_dqn_env(
    *,
    width: int = 5,
    height: int = 5,
    distribution: str = "correlated",
    seed: int | None = None,
):
    ...
```

It should:

1. create `ProbMinesweeperEnv`,
2. use `obs_mode="state+prob"`,
3. apply `FlattenObservationWrapper`,
4. return the wrapped environment.

### Implementation sketch

```python
from prob_minesweeper.env import ProbMinesweeperEnv
from prob_minesweeper.wrappers import FlattenObservationWrapper


def make_dqn_env(
    *,
    width: int = 5,
    height: int = 5,
    distribution: str = "correlated",
    seed: int | None = None,
):
    env = ProbMinesweeperEnv(
        width=width,
        height=height,
        distribution=distribution,
        obs_mode="state+prob",
        render_mode=None,
        seed=seed,
    )
    return FlattenObservationWrapper(env)
```

### Acceptance criteria

- Helper returns a valid Gymnasium environment.
- Stable-Baselines3 can receive this environment.

---

### Task 2.3 — Implement DQN training script

In `experiments/train_dqn.py`, implement CLI arguments:

```text
--width
--height
--distribution
--timesteps
--seed
--output
```

Default values:

```text
width = 5
height = 5
distribution = correlated
timesteps = 50000
seed = 42
output = models/dqn_prob_minesweeper.zip
```

### Recommended model setup

```python
from stable_baselines3 import DQN

model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=50_000,
    learning_starts=1_000,
    batch_size=64,
    gamma=0.95,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    target_update_interval=500,
    train_freq=4,
    verbose=1,
    seed=args.seed,
)
```

Then:

```python
model.learn(total_timesteps=args.timesteps)
model.save(args.output)
```

### Acceptance criteria

- Training runs with:

```bash
uv run python experiments/train_dqn.py --timesteps 1000
```

- A model file is created.
- Script does not require Streamlit.
- Script does not run automatically during tests.

---

### Task 2.4 — Add training README note

Update README with:

```bash
uv sync --dev --extra rl
uv run python experiments/train_dqn.py --timesteps 50000
```

### Acceptance criteria

- README contains clear DQN training instructions.

---

## Phase 3 — DQN agent wrapper

### Task 3.1 — Create `DQNAgent`

Create:

```text
prob_minesweeper/agents/dqn_agent.py
```

The class should load a Stable-Baselines3 DQN model and expose the same interface as other agents:

```python
class DQNAgent:
    name = "DQN"

    def __init__(self, model_path: str | Path, fallback_agent=None) -> None:
        ...

    def select_action(self, obs: np.ndarray, info: dict, env) -> int:
        ...
```

### Acceptance criteria

- Can be imported without loading a model immediately if possible.
- Raises a clear error if model file does not exist.
- Uses the same `select_action` interface as other agents.

---

### Task 3.2 — Handle observation flattening inside `DQNAgent`

The Streamlit app and evaluation function use the unwrapped environment, where observations have shape:

```text
(H, W, C)
```

The trained DQN expects flattened observations:

```text
(H * W * C,)
```

Therefore `DQNAgent.select_action` should flatten the observation before calling the model.

### Implementation sketch

```python
flat_obs = obs.astype(np.float32).reshape(1, -1)
action, _ = self.model.predict(flat_obs, deterministic=True)
action = int(action)
```

### Acceptance criteria

- `DQNAgent` can act on observations returned by the original environment.
- No need to wrap the Streamlit environment.

---

### Task 3.3 — Implement valid-action fallback

DQN may return an invalid action. Implement:

```python
mask = info["action_mask"]

if mask[action]:
    return action

return fallback_agent.select_action(obs, info, env)
```

Default fallback:

```python
MinRiskAgent()
```

If fallback also fails, select a random valid action.

### Acceptance criteria

- `DQNAgent` never returns an invalid action.
- If model predicts revealed cell, fallback selects a valid cell.
- Error message is clear if no valid actions exist.

---

### Task 3.4 — Optional: implement Q-value masking

If time allows, implement a better selection method:

1. Convert observation to tensor.
2. Run model Q-network.
3. Get Q-values for all actions.
4. Set invalid actions to `-np.inf`.
5. Select valid action with maximum Q-value.

Possible method name:

```python
select_action_masked_q_values(...)
```

This is optional.

### Acceptance criteria for optional task

- DQN selects the best legal action according to Q-values.
- Unit test verifies invalid actions are never selected.

---

## Phase 4 — DQN evaluation script

### Task 4.1 — Implement `experiments/evaluate_dqn.py`

The script should load a trained model and evaluate it.

CLI arguments:

```text
--model
--episodes
--width
--height
--distribution
--seed
```

Default values:

```text
model = models/dqn_prob_minesweeper.zip
episodes = 100
width = 5
height = 5
distribution = correlated
seed = 123
```

### Behavior

- Create `DQNAgent`.
- Call existing `evaluate_agent`.
- Print:

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

### Acceptance criteria

- Running the script prints readable metrics.
- Missing model path gives a clear error.
- Evaluation does not retrain the model.

---

### Task 4.2 — Add DQN to `compare_agents`

Where appropriate, allow benchmark code to include DQN if a model file exists.

Do not make DQN mandatory for all tests.

### Acceptance criteria

- Random and MinRisk benchmark still works without DQN.
- DQN benchmark works when model exists.

---

## Phase 5 — Streamlit integration

### Task 5.1 — Add DQN option to agent selector

In `app.py`, update the agent selector:

```text
Random
Min-risk
DQN
```

Only show or enable DQN if the model file exists:

```text
models/dqn_prob_minesweeper.zip
```

If it does not exist, show:

```text
DQN model not found. Train it with:
uv run python experiments/train_dqn.py --timesteps 50000
```

### Acceptance criteria

- App does not crash if model is missing.
- DQN can be selected after model is trained.
- DQN can make one move.
- DQN can run until terminal.

---

### Task 5.2 — Add DQN to benchmark tab

Benchmark tab should compare:

```text
RandomAgent
MinRiskAgent
DQNAgent, if model exists
```

The table should include:

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

### Acceptance criteria

- DQN appears in benchmark only when available.
- Benchmark works without DQN.
- Benchmark works with DQN.

---

### Task 5.3 — Add DQN explanation to model tab

In the Streamlit model tab, add a concise explanation:

```text
The DQN model learns Q(s,a), the expected long-term value of selecting cell a in board state s. Unlike MinRiskAgent, which only minimizes immediate mine probability, DQN estimates future reward and can learn policies that trade off immediate risk and long-term board progress.
```

Add the Bellman target:

```text
y = r + γ max_a' Q(s', a')
```

Add the loss:

```text
L(θ) = (y - Qθ(s, a))²
```

### Acceptance criteria

- Streamlit app explains why DQN is more than a local probability-minimization heuristic.

---

## Phase 6 — Tests

### Task 6.1 — Add DQN import guard tests

Create:

```text
tests/test_dqn_agent.py
```

Tests should skip if Stable-Baselines3 is not installed:

```python
pytest.importorskip("stable_baselines3")
```

Test cases:

1. Missing model path raises `FileNotFoundError` or custom clear error.
2. DQNAgent fallback returns valid action if mocked model predicts invalid action.
3. Observation flattening shape is correct.

### Acceptance criteria

- Tests pass when RL dependencies are installed.
- Tests are skipped when RL dependencies are missing.
- Full non-RL test suite remains fast.

---

### Task 6.2 — Do not train during tests

Important:

- No test should call `model.learn()` for thousands of timesteps.
- If a model object is needed, use a mock object.

### Acceptance criteria

- Test suite remains fast.
- CI/local test run does not create model files.

---

## Phase 7 — Documentation and report updates

### Task 7.1 — Update `report/report.md`

Add subsection:

```text
## Deep Q-Network jako wykorzystany model RL
```

Content should include:

- DQN is the selected existing RL model.
- It learns `Q(s,a)`.
- The board has discrete actions, so DQN fits the environment.
- The model is trained on flattened board observations.
- It is compared with RandomAgent and MinRiskAgent.
- Invalid actions are handled using `action_mask` and fallback.

### Acceptance criteria

- Report clearly states DQN is the model used for grade 4.0.
- It explains how DQN differs from MinRiskAgent.

---

### Task 7.2 — Update `report/defense_notes.md`

Add answers for:

```text
What does DQN learn?
Why is DQN appropriate here?
What is Q(s,a)?
What is the Bellman target?
How is action masking handled?
Why might MinRiskAgent outperform DQN?
```

Expected answer idea:

```text
MinRiskAgent has direct access to p_mine and is a strong local heuristic. DQN learns from interaction and optimizes expected future reward. It may not always beat MinRiskAgent, especially on small boards, but it is a learned RL policy rather than a fixed rule.
```

### Acceptance criteria

- Defense notes prepare the student for DQN-specific questions.

---

### Task 7.3 — Update README

Add section:

```text
## DQN agent
```

Include:

```bash
uv sync --dev --extra rl
uv run python experiments/train_dqn.py --timesteps 50000
uv run python experiments/evaluate_dqn.py --episodes 100
uv run streamlit run app.py
```

Mention:

```text
If no trained model exists, the Streamlit app still works with RandomAgent and MinRiskAgent.
```

### Acceptance criteria

- README contains DQN training, evaluation, and UI usage instructions.

---

## Phase 8 — Final validation

### Task 8.1 — Validate without RL extras

Run:

```bash
uv sync --dev
uv run pytest -q
uv run streamlit run app.py
```

Expected:

- Tests pass.
- App works.
- DQN option is disabled or shows a clear missing-model message.

### Acceptance criteria

- Project remains usable without DQN dependencies.

---

### Task 8.2 — Validate with RL extras

Run:

```bash
uv sync --dev --extra rl
uv run pytest -q
uv run python experiments/train_dqn.py --timesteps 1000 --output models/test_dqn.zip
uv run python experiments/evaluate_dqn.py --model models/test_dqn.zip --episodes 5
```

Expected:

- Training creates a model file.
- Evaluation runs.
- No invalid actions crash the episode loop.

### Acceptance criteria

- DQN end-to-end path works.

---

### Task 8.3 — Manual Streamlit DQN demo

Run:

```bash
uv run python experiments/train_dqn.py --timesteps 50000
uv run streamlit run app.py
```

Manual checks:

- Start a new game.
- Select DQN.
- Click `Agent move`.
- Click `Run agent until terminal`.
- Run benchmark with Random, Min-risk, and DQN.
- Confirm result table renders.

### Acceptance criteria

- DQN is demonstrable in the frontend.
- Benchmark data can be copied into the report.

---

## Recommended implementation order

If time is limited, implement in this order:

```text
1. Phase 0 — dependency setup
2. Phase 1 — flatten observation wrapper
3. Phase 2 — train_dqn.py
4. Phase 3 — DQNAgent with fallback
5. Phase 4 — evaluate_dqn.py
6. Phase 5 — Streamlit integration
7. Phase 7 — documentation/report
8. Phase 8 — validation
```

Do not spend time on optional Q-value masking until the basic train/load/evaluate/demo path works.

---

## Minimal acceptable DQN scope

For the project to claim DQN usage, at minimum the repository must contain:

```text
prob_minesweeper/wrappers.py
prob_minesweeper/agents/dqn_agent.py
experiments/train_dqn.py
experiments/evaluate_dqn.py
README instructions for training/evaluation
Streamlit integration that can load and use the model
report section explaining DQN
```

---

## Explanation for the report

Use this explanation in the report or defense:

```text
MinRiskAgent selects the currently safest-looking cell by minimizing p_mine. This is a local heuristic. DQN instead learns an action-value function Q(s,a), where the value of revealing a cell depends on both the immediate reward and the expected future rewards after the board state changes. Therefore, DQN can represent policies that are not limited to immediate risk minimization.
```

---

## Known limitations

1. DQN may not outperform MinRiskAgent.
   - MinRiskAgent has direct access to mine probabilities.
   - This is acceptable if explained honestly.

2. DQN training can be unstable.
   - Use small boards first.
   - Use fixed seeds.
   - Compare average results over many episodes.

3. DQN does not natively use `action_mask`.
   - Use fallback or explicit Q-value masking during evaluation.

4. The trained model may overfit board size.
   - Train and evaluate on the same board size for the project demo.

5. Do not promise perfect gameplay.
   - The goal is to demonstrate RL modeling and comparison, not build a superhuman Minesweeper solver.

---

## Definition of done

DQN integration is complete when:

- DQN dependencies install through optional `rl` extra.
- Flatten observation wrapper exists and is tested.
- DQN training script saves a model.
- DQN evaluation script prints metrics.
- DQNAgent loads model and returns valid actions.
- Streamlit app can use DQN when model file exists.
- Benchmark compares RandomAgent, MinRiskAgent, and DQNAgent.
- README includes DQN commands.
- Report explains DQN mathematically.
- Defense notes cover DQN questions.
