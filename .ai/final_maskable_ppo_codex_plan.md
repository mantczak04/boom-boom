# Final Codex Plan — MaskablePPO, Honest DQN Fallback, and Two-Regime Evaluation

## Context

This plan updates the existing `prob-minesweeper` project by adding `MaskablePPO` as a second reinforcement-learning model and by cleaning up the current DQN evaluation.

The key issue is not only model quality. The current DQN path has a methodological weakness: Stable-Baselines3 DQN does not use `action_mask` during training, and the project currently stabilizes invalid DQN predictions with a fallback that can use `MinRiskAgent`. In hidden-risk mode, `MinRiskAgent` is an oracle because it reads hidden `p_mine` values unavailable to the RL observation. Therefore, a fair DQN-vs-MaskablePPO comparison requires:

1. explicit action-mask support for MaskablePPO,
2. a robust `action_masks()` API through wrappers,
3. DQN fallback changed to random valid action for fair evaluation,
4. `invalid_action_rate` as a mandatory reported metric,
5. separate model discovery for DQN and MaskablePPO in Streamlit,
6. easier and hard evaluation regimes reported as two first-class experiment tables.

Do not present MaskablePPO as guaranteed to beat the Min-risk oracle. Present it as a cleaner RL method that uses action masks during training and inference.

---

## Final target behavior

After implementation, the project should support:

```bash
uv sync --dev --extra rl
uv run pytest -q

uv run python experiments/train_maskable_ppo.py --timesteps 1000 --verbose 0
uv run python experiments/evaluate_maskable_ppo.py --episodes 5
uv run python experiments/compare_rl_agents.py --episodes 5
```

Final comparison should include:

- `RandomAgent`
- `MinRiskAgent` shown as `Min-risk (oracle)`
- `DQN` with random valid-action fallback
- `MaskablePPO`

Final report should include two experiment tables:

1. Easier learning regime, e.g. `constant p=0.15`
2. Hard hidden-risk stress test, e.g. current `correlated` setup

---

# Phase 1 — Dependency setup

## Task 1.1 — Add `sb3-contrib` with matching minor version

Edit:

```text
pyproject.toml
```

Current RL extra contains `stable-baselines3`. Add `sb3-contrib` and pin both packages to the same minor line.

Preferred if current project works with SB3 2.3:

```toml
rl = [
    "stable-baselines3>=2.3,<2.4",
    "sb3-contrib>=2.3,<2.4",
    "torch",
    "matplotlib",
    "fastapi",
    "uvicorn[standard]",
    "pydantic",
    "tqdm",
]
```

If the resolver upgrades the project to a newer SB3 line, keep the minor line matched, for example:

```toml
"stable-baselines3>=2.7,<2.8",
"sb3-contrib>=2.7,<2.8",
```

Do not leave both as loose unrelated `>=2.3` dependencies. `sb3-contrib` should match the `stable-baselines3` minor version.

## Acceptance criteria

Run:

```bash
uv sync --dev --extra rl
uv run python - <<'PY'
import stable_baselines3
import sb3_contrib
from sb3_contrib import MaskablePPO

print("stable-baselines3", stable_baselines3.__version__)
print("sb3-contrib", sb3_contrib.__version__)
print(MaskablePPO)
PY
```

Expected:

- import succeeds,
- SB3 and sb3-contrib major/minor versions match.

---

# Phase 2 — Environment action-mask API

MaskablePPO needs a valid-action mask. The base environment already returns `info["action_mask"]`, but `sb3-contrib` also expects an `action_masks()` method.

## Task 2.1 — Add `action_masks()` to base env

Edit:

```text
prob_minesweeper/env.py
```

Inside `ProbMinesweeperEnv`, near `_action_mask`, add:

```python
def action_masks(self) -> np.ndarray:
    """Return a boolean valid-action mask for sb3-contrib MaskablePPO."""
    return self._action_mask()
```

Required behavior:

- shape: `(height * width,)`
- dtype: `np.bool_`
- `True` means action is valid
- `False` means action is invalid/revealed
- implementation must call `_action_mask()` to avoid duplicated logic

## Task 2.2 — Add env test

Edit:

```text
tests/test_env.py
```

Add:

```python
def test_action_masks_method_matches_info_mask() -> None:
    env = _env(distribution=ConstantDistribution(p=0.0))
    _, info = env.reset(seed=0)

    np.testing.assert_array_equal(env.action_masks(), info["action_mask"])
    assert env.action_masks().dtype == np.bool_

    env.step(0)
    assert not env.action_masks()[0]
```

## Acceptance criteria

```bash
uv run pytest tests/test_env.py -q
```

---

# Phase 3 — Robust mask delegation through FlattenObservationWrapper

MaskablePPO will train on `FlattenObservationWrapper(ProbMinesweeperEnv(...))`. Do not rely only on Gymnasium attribute forwarding. Add explicit delegation.

## Task 3.1 — Add `action_masks()` to wrapper

Edit:

```text
prob_minesweeper/wrappers.py
```

Inside `FlattenObservationWrapper`, add:

```python
def action_masks(self) -> np.ndarray:
    """Delegate valid-action masks to the wrapped environment for MaskablePPO."""
    action_masks = getattr(self.env, "action_masks", None)
    if action_masks is None:
        raise AttributeError("Wrapped environment does not expose action_masks()")
    return np.asarray(action_masks(), dtype=np.bool_)
```

`numpy as np` is already imported in this file.

## Task 3.2 — Add wrapper test

Edit:

```text
tests/test_wrappers.py
```

Add:

```python
def test_flatten_wrapper_delegates_action_masks():
    env = FlattenObservationWrapper(
        ProbMinesweeperEnv(width=3, height=3, distribution="constant")
    )
    try:
        _, info = env.reset(seed=0)
        np.testing.assert_array_equal(env.action_masks(), info["action_mask"])
        assert env.action_masks().dtype == np.bool_

        env.step(0)
        assert not env.action_masks()[0]
    finally:
        env.close()
```

## Acceptance criteria

```bash
uv run pytest tests/test_wrappers.py -q
```

---

# Phase 4 — Distribution parameter support for experiment scripts

The easier regime needs commands like:

```bash
--distribution constant --p 0.15
```

Current DQN-style scripts mostly accept only the distribution name. Add reusable distribution-kwargs parsing.

## Task 4.1 — Create experiment helper module

Create:

```text
experiments/common.py
```

Add:

```python
"""Shared helpers for experiment scripts."""

from __future__ import annotations

import argparse
from typing import Any


def add_distribution_args(parser: argparse.ArgumentParser) -> None:
    """Add optional distribution parameter arguments to an experiment parser."""
    parser.add_argument(
        "--p",
        type=float,
        default=None,
        help="Mine probability for constant distribution",
    )
    parser.add_argument(
        "--low",
        type=float,
        default=None,
        help="Lower bound for uniform distribution",
    )
    parser.add_argument(
        "--high",
        type=float,
        default=None,
        help="Upper bound for uniform distribution",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Gaussian smoothing sigma for correlated distribution",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale parameter for correlated distribution",
    )


def build_distribution_kwargs(args: argparse.Namespace) -> dict[str, Any] | None:
    """Build ProbMinesweeperEnv distribution_kwargs from parsed CLI args."""
    if args.distribution == "constant":
        return {"p": args.p} if args.p is not None else None

    if args.distribution == "uniform":
        kwargs: dict[str, Any] = {}
        if args.low is not None:
            kwargs["low"] = args.low
        if args.high is not None:
            kwargs["high"] = args.high
        return kwargs or None

    if args.distribution == "correlated":
        kwargs: dict[str, Any] = {}
        if args.sigma is not None:
            kwargs["sigma"] = args.sigma
        if args.scale is not None:
            kwargs["scale"] = args.scale
        return kwargs or None

    return None
```

## Task 4.2 — Add tests

Create:

```text
tests/test_experiment_distribution_kwargs.py
```

Add:

```python
from argparse import Namespace

import pytest

from experiments.common import build_distribution_kwargs


def test_build_distribution_kwargs_constant_with_p():
    args = Namespace(
        distribution="constant",
        p=0.15,
        low=None,
        high=None,
        sigma=None,
        scale=None,
    )
    assert build_distribution_kwargs(args) == {"p": 0.15}


def test_build_distribution_kwargs_constant_without_p():
    args = Namespace(
        distribution="constant",
        p=None,
        low=None,
        high=None,
        sigma=None,
        scale=None,
    )
    assert build_distribution_kwargs(args) is None


def test_build_distribution_kwargs_uniform_partial():
    args = Namespace(
        distribution="uniform",
        p=None,
        low=0.05,
        high=0.25,
        sigma=None,
        scale=None,
    )
    assert build_distribution_kwargs(args) == {"low": 0.05, "high": 0.25}


def test_build_distribution_kwargs_correlated_partial():
    args = Namespace(
        distribution="correlated",
        p=None,
        low=None,
        high=None,
        sigma=1.5,
        scale=None,
    )
    assert build_distribution_kwargs(args) == {"sigma": 1.5}
```

## Acceptance criteria

```bash
uv run pytest tests/test_experiment_distribution_kwargs.py -q
```

---

# Phase 5 — MaskablePPO training script

## Task 5.1 — Create script

Create:

```text
experiments/train_maskable_ppo.py
```

Implementation:

```python
"""Train sb3-contrib MaskablePPO on probabilistic Minesweeper."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from experiments.common import add_distribution_args, build_distribution_kwargs
from prob_minesweeper.env import ProbMinesweeperEnv
from prob_minesweeper.rewards import REWARD_MODES, make_reward_config
from prob_minesweeper.wrappers import FlattenObservationWrapper


def make_maskable_ppo_env(
    *,
    width: int = 5,
    height: int = 5,
    distribution: str = "correlated",
    distribution_kwargs: dict[str, Any] | None = None,
    obs_mode: str = "state",
    clue_mode: str = "actual_count",
    initial_reveal: str = "safe_2x2",
    reward_mode: str = "completion",
    seed: int | None = None,
) -> FlattenObservationWrapper:
    env = ProbMinesweeperEnv(
        width=width,
        height=height,
        distribution=distribution,
        distribution_kwargs=distribution_kwargs,
        obs_mode=obs_mode,
        clue_mode=clue_mode,
        initial_reveal=initial_reveal,
        reward_config=make_reward_config(reward_mode),
        render_mode=None,
        seed=seed,
    )
    return FlattenObservationWrapper(env)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument(
        "--distribution",
        choices=("constant", "uniform", "correlated"),
        default="correlated",
    )
    add_distribution_args(parser)
    parser.add_argument(
        "--obs-mode", choices=("state", "state+prob"), default="state"
    )
    parser.add_argument(
        "--clue-mode", choices=("prob_sum", "actual_count"), default="actual_count"
    )
    parser.add_argument(
        "--initial-reveal", choices=("none", "safe_2x2"), default="safe_2x2"
    )
    parser.add_argument(
        "--reward-mode", choices=REWARD_MODES, default="completion"
    )
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/maskable_ppo_prob_minesweeper.zip"),
    )
    return parser


def train(args: argparse.Namespace) -> Path:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "sb3-contrib is required. Run `uv sync --dev --extra rl`."
        ) from exc

    if args.timesteps < 1:
        raise ValueError("timesteps must be >= 1")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    distribution_kwargs = build_distribution_kwargs(args)

    env = make_maskable_ppo_env(
        width=args.width,
        height=args.height,
        distribution=args.distribution,
        distribution_kwargs=distribution_kwargs,
        obs_mode=args.obs_mode,
        clue_mode=args.clue_mode,
        initial_reveal=args.initial_reveal,
        reward_mode=args.reward_mode,
        seed=args.seed,
    )

    mask = env.action_masks()
    if mask.shape != (args.width * args.height,):
        raise RuntimeError(f"Invalid action mask shape: {mask.shape}")
    if mask.dtype != np.bool_:
        raise RuntimeError(f"Invalid action mask dtype: {mask.dtype}")

    model: Any = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.98,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        device="cpu",
        verbose=args.verbose,
        seed=args.seed,
    )

    try:
        model.learn(total_timesteps=args.timesteps)
        model.save(args.output)
    finally:
        env.close()

    return args.output


def main() -> None:
    args = build_parser().parse_args()
    output = train(args)
    print(f"Saved MaskablePPO model to {output}")


if __name__ == "__main__":
    main()
```

## Important distinction

The explicit `env.action_masks()` check proves only that the flattened wrapper exposes the mask.

The smoke-train proves that `sb3-contrib` can discover and use the mask through its SB3 wrapping path.

## Task 5.2 — Add experiment config tests

Edit:

```text
tests/test_experiment_config.py
```

Add imports:

```python
from experiments.train_maskable_ppo import (
    build_parser as build_maskable_ppo_training_parser,
)
from experiments.train_maskable_ppo import make_maskable_ppo_env
```

Add tests:

```python
def test_maskable_ppo_training_defaults_to_hidden_risk() -> None:
    args = build_maskable_ppo_training_parser().parse_args([])
    assert args.obs_mode == "state"
    assert args.clue_mode == "actual_count"
    assert args.initial_reveal == "safe_2x2"
    assert args.reward_mode == "completion"
    assert args.timesteps == 100_000


def test_maskable_ppo_environment_has_visible_state_shape() -> None:
    env = make_maskable_ppo_env(
        width=3,
        height=3,
        distribution="constant",
        distribution_kwargs={"p": 0.15},
        seed=0,
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3 * 3 * 3,)
        assert env.unwrapped.obs_mode == "state"
        assert env.unwrapped.clue_mode == "actual_count"
        assert env.unwrapped.initial_reveal == "safe_2x2"
        assert env.unwrapped.reward_config.win_bonus == 10.0
        assert env.unwrapped.board.revealed_mask().sum() == 4
        assert env.unwrapped.board.p_mine_field().mean() == pytest.approx(0.15)
    finally:
        env.close()
```

Ensure `pytest` is imported if not already present.

## Acceptance criteria

```bash
uv run pytest tests/test_experiment_config.py -q
uv run python experiments/train_maskable_ppo.py --timesteps 1000 --verbose 0
uv run python experiments/train_maskable_ppo.py --timesteps 1000 --distribution constant --p 0.15 --verbose 0
```

---

# Phase 6 — MaskablePPO agent adapter

## Task 6.1 — Create agent

Create:

```text
prob_minesweeper/agents/maskable_ppo_agent.py
```

Implementation:

```python
"""sb3-contrib MaskablePPO adapter for the common agent interface."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class MaskablePPOAgent:
    """Load a trained MaskablePPO model and select valid masked actions."""

    name = "MaskablePPO"

    def __init__(self, model_path: str | Path) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"MaskablePPO model not found: {path}")

        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise ImportError(
                "MaskablePPOAgent requires sb3-contrib. Install it with "
                "`uv sync --dev --extra rl`."
            ) from exc

        self.model = MaskablePPO.load(path, device="cpu")

    def select_action(
        self, obs: np.ndarray, info: dict[str, Any], env: Any
    ) -> int:
        del env

        mask = np.asarray(info["action_mask"], dtype=np.bool_).reshape(-1)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            raise RuntimeError("No valid actions available")

        flat_obs = np.asarray(obs, dtype=np.float32).reshape(1, -1)

        model_space = getattr(self.model, "observation_space", None)
        expected_shape = getattr(model_space, "shape", None)
        if expected_shape is not None and tuple(expected_shape) != tuple(
            flat_obs.shape[1:]
        ):
            raise ValueError(
                "MaskablePPO model observation shape does not match this board: "
                f"expected {tuple(expected_shape)}, got {tuple(flat_obs.shape[1:])}. "
                "Train and evaluate MaskablePPO with the same width, height, and obs_mode."
            )

        predicted, _ = self.model.predict(
            flat_obs,
            deterministic=True,
            action_masks=mask,
        )
        action = int(np.asarray(predicted).reshape(-1)[0])

        if 0 <= action < len(mask) and mask[action]:
            return action

        raise RuntimeError(
            f"MaskablePPO predicted invalid action {action} despite action mask"
        )
```

## Task 6.2 — Export agent

Edit:

```text
prob_minesweeper/agents/__init__.py
```

Add:

```python
from prob_minesweeper.agents.maskable_ppo_agent import MaskablePPOAgent
```

Update `__all__`:

```python
__all__ = ["Agent", "DQNAgent", "MaskablePPOAgent", "MinRiskAgent", "RandomAgent"]
```

## Task 6.3 — Add tests

Create:

```text
tests/test_maskable_ppo_agent.py
```

Add:

```python
import numpy as np
import pytest

sb3_contrib = pytest.importorskip("sb3_contrib")

from prob_minesweeper.agents.maskable_ppo_agent import MaskablePPOAgent


class FakeMaskablePPOModel:
    def __init__(self, action: int) -> None:
        self.action = action
        self.last_observation = None
        self.last_action_masks = None
        self.observation_space = None

    def predict(self, observation, deterministic=True, action_masks=None):
        assert deterministic is True
        self.last_observation = observation
        self.last_action_masks = action_masks
        return np.array([self.action]), None


def test_missing_maskable_ppo_model_path_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="MaskablePPO model not found"):
        MaskablePPOAgent(tmp_path / "missing.zip")


def test_maskable_ppo_agent_passes_flattened_obs_and_mask(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.touch()

    fake_model = FakeMaskablePPOModel(action=1)

    monkeypatch.setattr(
        sb3_contrib.MaskablePPO,
        "load",
        lambda path, device="auto": fake_model,
    )

    agent = MaskablePPOAgent(model_path)

    obs = np.zeros((2, 2, 3), dtype=np.float32)
    info = {"action_mask": np.array([False, True, False, True])}

    action = agent.select_action(obs, info, env=None)

    assert action == 1
    assert fake_model.last_observation.shape == (1, 2 * 2 * 3)
    assert fake_model.last_observation.dtype == np.float32
    np.testing.assert_array_equal(
        fake_model.last_action_masks,
        info["action_mask"],
    )


def test_maskable_ppo_invalid_prediction_raises(monkeypatch, tmp_path):
    model_path = tmp_path / "model.zip"
    model_path.touch()

    fake_model = FakeMaskablePPOModel(action=0)

    monkeypatch.setattr(
        sb3_contrib.MaskablePPO,
        "load",
        lambda path, device="auto": fake_model,
    )

    agent = MaskablePPOAgent(model_path)
    obs = np.zeros((2, 2, 3), dtype=np.float32)
    info = {"action_mask": np.array([False, True, False, True])}

    with pytest.raises(RuntimeError, match="predicted invalid action"):
        agent.select_action(obs, info, env=None)
```

## Acceptance criteria

```bash
uv run pytest tests/test_maskable_ppo_agent.py -q
uv run python -c "from prob_minesweeper.agents import MaskablePPOAgent; print(MaskablePPOAgent.name)"
```

---

# Phase 7 — Honest DQN fallback and invalid-action metric

The existing DQN fallback to `MinRiskAgent` is useful for UI stability but contaminates fair evaluation because Min-risk is an oracle in hidden-risk mode.

## Task 7.1 — Add random fallback and counters

Edit:

```text
prob_minesweeper/agents/dqn_agent.py
```

Add this helper class near the top:

```python
class RandomValidFallback:
    name = "RandomValidFallback"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs: np.ndarray, info: dict[str, Any], env: Any) -> int:
        del obs, env
        valid = np.flatnonzero(np.asarray(info["action_mask"], dtype=np.bool_))
        if len(valid) == 0:
            raise RuntimeError("No valid actions available")
        return int(self.rng.choice(valid))
```

Change `DQNAgent.__init__` signature:

```python
def __init__(
    self,
    model_path: str | Path,
    fallback_agent: Any | None = None,
    *,
    fallback_mode: str = "random",
    seed: int | None = None,
) -> None:
```

After model load, set fallback:

```python
if fallback_agent is not None:
    self.fallback_agent = fallback_agent
elif fallback_mode == "min_risk":
    self.fallback_agent = MinRiskAgent()
elif fallback_mode == "random":
    self.fallback_agent = RandomValidFallback(seed)
else:
    raise ValueError("fallback_mode must be 'random' or 'min_risk'")

self.predictions = 0
self.invalid_predictions = 0
```

Inside `select_action`, after `model.predict` result is read, increment total predictions:

```python
self.predictions += 1
```

If the action is invalid, increment invalid predictions before fallback:

```python
self.invalid_predictions += 1
```

Add property:

```python
@property
def invalid_action_rate(self) -> float:
    if self.predictions == 0:
        return 0.0
    return self.invalid_predictions / self.predictions
```

Keep support for `fallback_mode="min_risk"` only for demo/debug use. Fair benchmark must use `fallback_mode="random"`.

## Task 7.2 — Add DQN tests

Edit:

```text
tests/test_dqn_agent.py
```

The existing `test_invalid_prediction_uses_valid_fallback` should still pass because on a 2x1 board only one valid action remains.

Add:

```python
def test_invalid_prediction_increments_invalid_action_counter(monkeypatch, tmp_path):
    model = FakeModel(action=0)
    agent = make_agent(monkeypatch, tmp_path, model)

    env = ProbMinesweeperEnv(
        width=2,
        height=1,
        distribution=ConstantDistribution(0.0),
        obs_mode="state+prob",
    )
    try:
        obs, _ = env.reset(seed=1)
        obs, _, _, _, info = env.step(0)

        action = agent.select_action(obs, info, env)

        assert action == 1
        assert agent.predictions == 1
        assert agent.invalid_predictions == 1
        assert agent.invalid_action_rate == pytest.approx(1.0)
    finally:
        env.close()


def test_valid_prediction_does_not_increment_invalid_counter(monkeypatch, tmp_path):
    model = FakeModel(action=0)
    agent = make_agent(monkeypatch, tmp_path, model)

    env = ProbMinesweeperEnv(
        width=2,
        height=1,
        distribution=ConstantDistribution(0.0),
        obs_mode="state+prob",
    )
    try:
        obs, info = env.reset(seed=1)

        action = agent.select_action(obs, info, env)

        assert action == 0
        assert agent.predictions == 1
        assert agent.invalid_predictions == 0
        assert agent.invalid_action_rate == pytest.approx(0.0)
    finally:
        env.close()
```

## Acceptance criteria

```bash
uv run pytest tests/test_dqn_agent.py -q
```

---

# Phase 8 — MaskablePPO evaluation script

## Task 8.1 — Create script

Create:

```text
experiments/evaluate_maskable_ppo.py
```

Implementation:

```python
"""Evaluate a saved MaskablePPO model without retraining it."""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.common import add_distribution_args, build_distribution_kwargs
from prob_minesweeper.agents import MaskablePPOAgent
from prob_minesweeper.evaluation import EvaluationResult, evaluate_agent
from prob_minesweeper.rewards import REWARD_MODES, make_reward_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/maskable_ppo_prob_minesweeper.zip"),
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument(
        "--distribution",
        choices=("constant", "uniform", "correlated"),
        default="correlated",
    )
    add_distribution_args(parser)
    parser.add_argument(
        "--obs-mode", choices=("state", "state+prob"), default="state"
    )
    parser.add_argument(
        "--clue-mode", choices=("prob_sum", "actual_count"), default="actual_count"
    )
    parser.add_argument(
        "--initial-reveal", choices=("none", "safe_2x2"), default="safe_2x2"
    )
    parser.add_argument(
        "--reward-mode", choices=REWARD_MODES, default="completion"
    )
    parser.add_argument("--seed", type=int, default=123)
    return parser


def print_result(result: EvaluationResult) -> None:
    rows = (
        ("Agent", result.agent_name),
        ("Episodes", result.episodes),
        ("Wins", result.wins),
        ("Losses", result.losses),
        ("Truncated", result.truncated),
        ("Win rate", f"{result.win_rate:.2%}"),
        ("Mean reward", f"{result.mean_reward:.3f}"),
        ("Mean steps", f"{result.mean_steps:.2f}"),
    )
    width = max(len(label) for label, _ in rows)
    for label, value in rows:
        print(f"{label:<{width}} : {value}")


def main() -> None:
    args = build_parser().parse_args()
    distribution_kwargs = build_distribution_kwargs(args)

    print("Environment configuration")
    configuration = (
        ("width", args.width),
        ("height", args.height),
        ("distribution", args.distribution),
        ("distribution_kwargs", distribution_kwargs),
        ("obs_mode", args.obs_mode),
        ("clue_mode", args.clue_mode),
        ("initial_reveal", args.initial_reveal),
        ("reward_mode", args.reward_mode),
    )
    label_width = max(len(label) for label, _ in configuration)
    for label, value in configuration:
        print(f"{label:<{label_width}} : {value}")
    print()

    agent = MaskablePPOAgent(args.model)
    result = evaluate_agent(
        agent,
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        distribution=args.distribution,
        distribution_kwargs=distribution_kwargs,
        obs_mode=args.obs_mode,
        clue_mode=args.clue_mode,
        initial_reveal=args.initial_reveal,
        reward_config=make_reward_config(args.reward_mode),
        seed=args.seed,
    )
    print_result(result)


if __name__ == "__main__":
    main()
```

## Acceptance criteria

```bash
uv run python experiments/train_maskable_ppo.py --timesteps 1000 --verbose 0
uv run python experiments/evaluate_maskable_ppo.py --episodes 5
```

---

# Phase 9 — Compare all agents script

## Task 9.1 — Create comparison script

Create:

```text
experiments/compare_rl_agents.py
```

Implementation:

```python
"""Compare available agents on the same probabilistic Minesweeper configuration."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from experiments.common import add_distribution_args, build_distribution_kwargs
from prob_minesweeper.agents import DQNAgent, MaskablePPOAgent, MinRiskAgent, RandomAgent
from prob_minesweeper.evaluation import compare_agents
from prob_minesweeper.rewards import REWARD_MODES, make_reward_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--width", type=int, default=5)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument(
        "--distribution",
        choices=("constant", "uniform", "correlated"),
        default="correlated",
    )
    add_distribution_args(parser)
    parser.add_argument(
        "--obs-mode", choices=("state", "state+prob"), default="state"
    )
    parser.add_argument(
        "--clue-mode", choices=("prob_sum", "actual_count"), default="actual_count"
    )
    parser.add_argument(
        "--initial-reveal", choices=("none", "safe_2x2"), default="safe_2x2"
    )
    parser.add_argument(
        "--reward-mode", choices=REWARD_MODES, default="completion"
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--dqn-model",
        type=Path,
        default=Path("models/dqn_prob_minesweeper.zip"),
    )
    parser.add_argument(
        "--maskable-ppo-model",
        type=Path,
        default=Path("models/maskable_ppo_prob_minesweeper.zip"),
    )
    return parser


def display_agent_name(agent_name: str) -> str:
    if agent_name == "Min-risk":
        return "Min-risk (oracle)"
    return agent_name


def print_table(results: list[Any]) -> None:
    headers = [
        "Agent",
        "Episodes",
        "Wins",
        "Losses",
        "Truncated",
        "Win rate",
        "Mean reward",
        "Mean steps",
    ]
    rows = [
        [
            display_agent_name(r.agent_name),
            str(r.episodes),
            str(r.wins),
            str(r.losses),
            str(r.truncated),
            f"{r.win_rate:.2%}",
            f"{r.mean_reward:.3f}",
            f"{r.mean_steps:.2f}",
        ]
        for r in results
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(row))))


def print_agent_diagnostics(agents: list[Any]) -> None:
    for agent in agents:
        invalid_action_rate = getattr(agent, "invalid_action_rate", None)
        if invalid_action_rate is not None:
            print(f"{agent.name} invalid action rate: {invalid_action_rate:.2%}")


def main() -> None:
    args = build_parser().parse_args()
    distribution_kwargs = build_distribution_kwargs(args)

    agents: list[Any] = [
        RandomAgent(args.seed),
        MinRiskAgent(),
    ]

    if args.dqn_model.is_file():
        try:
            dqn_agent = DQNAgent(
                args.dqn_model,
                fallback_mode="random",
                seed=args.seed,
            )
            dqn_agent.name = "DQN (random fallback)"
            agents.append(dqn_agent)
        except ImportError as exc:
            print(f"Skipping DQN: {exc}")
    else:
        print(f"Skipping DQN: model not found at {args.dqn_model}")

    if args.maskable_ppo_model.is_file():
        try:
            agents.append(MaskablePPOAgent(args.maskable_ppo_model))
        except ImportError as exc:
            print(f"Skipping MaskablePPO: {exc}")
    else:
        print(f"Skipping MaskablePPO: model not found at {args.maskable_ppo_model}")

    results = compare_agents(
        agents,
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        distribution=args.distribution,
        distribution_kwargs=distribution_kwargs,
        obs_mode=args.obs_mode,
        clue_mode=args.clue_mode,
        initial_reveal=args.initial_reveal,
        reward_config=make_reward_config(args.reward_mode),
        seed=args.seed,
    )

    print_table(results)
    print()
    print_agent_diagnostics(agents)


if __name__ == "__main__":
    main()
```

## Important behavior

- If DQN model file is missing, skip DQN.
- If MaskablePPO model file is missing, skip MaskablePPO.
- If optional dependency is missing, skip that optional model.
- If model file exists but is corrupt/incompatible, fail loudly.
- Do not catch all exceptions.
- Fair DQN comparison must use `fallback_mode="random"`.

## Acceptance criteria

```bash
uv run python experiments/compare_rl_agents.py --episodes 10
```

Expected:

- Random and Min-risk always appear.
- DQN appears only if model exists and SB3 is installed.
- MaskablePPO appears only if model exists and sb3-contrib is installed.
- DQN invalid action rate is printed when DQN is included.

---

# Phase 10 — Streamlit integration

## Task 10.1 — Import MaskablePPOAgent

Edit:

```text
app.py
```

Change:

```python
from prob_minesweeper.agents import DQNAgent, MinRiskAgent, RandomAgent
```

To:

```python
from prob_minesweeper.agents import DQNAgent, MaskablePPOAgent, MinRiskAgent, RandomAgent
```

## Task 10.2 — Fix model discovery

DQN discovery should not show `maskable_ppo_*.zip` files.

Replace `_dqn_model_paths()` with:

```python
def _dqn_model_paths() -> list[Path]:
    return sorted(
        (
            path
            for path in DQN_MODELS_DIR.glob("*.zip")
            if not path.name.lower().startswith("maskable_ppo")
        ),
        key=lambda path: path.name.lower(),
    )
```

Add:

```python
def _maskable_ppo_model_paths() -> list[Path]:
    return sorted(
        (
            path
            for path in DQN_MODELS_DIR.glob("*.zip")
            if path.name.lower().startswith("maskable_ppo")
        ),
        key=lambda path: path.name.lower(),
    )
```

Do not use prefix `"ppo"`, because it can catch non-maskable PPO models.

## Task 10.3 — Add selected MaskablePPO path helper

Add:

```python
def _selected_maskable_ppo_model_path() -> Path | None:
    model_name = st.session_state.get("maskable_ppo_model_selector")
    if not model_name:
        return None
    path = DQN_MODELS_DIR / str(model_name)
    return path if path.is_file() else None
```

## Task 10.4 — Add availability helper

Add:

```python
def _maskable_ppo_available() -> bool:
    return (
        _selected_maskable_ppo_model_path() is not None
        and importlib.util.find_spec("sb3_contrib") is not None
    )
```

## Task 10.5 — Add cached loader

Add:

```python
@st.cache_resource
def _load_maskable_ppo_agent(model_path: str, model_mtime_ns: int) -> MaskablePPOAgent:
    del model_mtime_ns
    return MaskablePPOAgent(model_path)
```

## Task 10.6 — Add compatibility check

Add:

```python
def _maskable_ppo_compatible(width: int, height: int) -> bool:
    if not _maskable_ppo_available():
        return False
    model_path = _selected_maskable_ppo_model_path()
    if model_path is None:
        return False
    try:
        agent = _load_maskable_ppo_agent(str(model_path), model_path.stat().st_mtime_ns)
    except (ImportError, OSError, RuntimeError, ValueError):
        return False
    model_space = getattr(agent.model, "observation_space", None)
    expected_shape = getattr(model_space, "shape", None)
    return tuple(expected_shape or ()) == (width * height * 3,)
```

The app currently creates envs with `obs_mode="state"`, so 3 channels are expected.

## Task 10.7 — Update `_agent()`

Change logic to:

```python
def _agent() -> Any:
    if st.session_state.selected_agent == "Random":
        return RandomAgent()

    if st.session_state.selected_agent == "DQN":
        model_path = _selected_dqn_model_path()
        if model_path is None:
            raise RuntimeError("No DQN model is selected")
        return _load_dqn_agent(str(model_path), model_path.stat().st_mtime_ns)

    if st.session_state.selected_agent == "MaskablePPO":
        model_path = _selected_maskable_ppo_model_path()
        if model_path is None:
            raise RuntimeError("No MaskablePPO model is selected")
        return _load_maskable_ppo_agent(str(model_path), model_path.stat().st_mtime_ns)

    return MinRiskAgent()
```

For Streamlit demo, DQN uses its default fallback mode. The default should now be random fallback after Phase 7.

## Task 10.8 — Update game agent options

Inside `_render_game`, use:

```python
agent_options = ["Random", "Min-risk (oracle)"]

dqn_compatible = _dqn_compatible(env.width, env.height)
maskable_ppo_compatible = _maskable_ppo_compatible(env.width, env.height)

if dqn_compatible:
    agent_options.append("DQN")
if maskable_ppo_compatible:
    agent_options.append("MaskablePPO")
```

Add MaskablePPO info messages:

```python
if not _maskable_ppo_model_paths():
    st.info(
        "MaskablePPO model not found. Train it with: "
        "`uv run python experiments/train_maskable_ppo.py --timesteps 100000`"
    )
elif not _maskable_ppo_available():
    st.info("Install the RL dependencies with: `uv sync --dev --extra rl`")
elif not maskable_ppo_compatible:
    st.info(
        "The selected MaskablePPO model is incompatible with this board size or "
        "the 3-channel state observation. Select or train a matching model."
    )
```

## Task 10.9 — Add sidebar selector

In `main()`, after the DQN selector block, add:

```python
maskable_ppo_models = _maskable_ppo_model_paths()
if maskable_ppo_models:
    model_names = [path.name for path in maskable_ppo_models]
    preferred_model = "maskable_ppo_prob_minesweeper.zip"
    st.sidebar.selectbox(
        "MaskablePPO model",
        model_names,
        index=(
            model_names.index(preferred_model)
            if preferred_model in model_names
            else 0
        ),
        key="maskable_ppo_model_selector",
    )
```

## Task 10.10 — Add MaskablePPO to benchmark tab

Inside `_benchmark_tab`, update agent list construction:

```python
agents: list[Any] = [RandomAgent(seed), MinRiskAgent()]

if _dqn_compatible(width, height):
    model_path = _selected_dqn_model_path()
    if model_path is not None:
        agents.append(
            _load_dqn_agent(str(model_path), model_path.stat().st_mtime_ns)
        )

if _maskable_ppo_compatible(width, height):
    model_path = _selected_maskable_ppo_model_path()
    if model_path is not None:
        agents.append(
            _load_maskable_ppo_agent(str(model_path), model_path.stat().st_mtime_ns)
        )
```

## Task 10.11 — Update app tests

Edit:

```text
tests/test_app_imports.py
```

Keep the existing DQN model dropdown test valid by excluding only `maskable_ppo` files, not by filtering DQN to only `dqn_*`.

Add/replace tests:

```python
def test_dqn_model_dropdown_excludes_maskable_ppo_zip_files(monkeypatch, tmp_path):
    import app

    (tmp_path / "z_model.zip").touch()
    (tmp_path / "a_model.zip").touch()
    (tmp_path / "maskable_ppo_model.zip").touch()
    (tmp_path / "notes.txt").touch()
    monkeypatch.setattr(app, "DQN_MODELS_DIR", tmp_path)

    assert [path.name for path in app._dqn_model_paths()] == [
        "a_model.zip",
        "z_model.zip",
    ]


def test_maskable_ppo_model_dropdown_discovers_only_maskable_zip_files(monkeypatch, tmp_path):
    import app

    (tmp_path / "maskable_ppo_model.zip").touch()
    (tmp_path / "ppo_model.zip").touch()
    (tmp_path / "dqn_model.zip").touch()
    (tmp_path / "notes.txt").touch()
    monkeypatch.setattr(app, "DQN_MODELS_DIR", tmp_path)

    assert [path.name for path in app._maskable_ppo_model_paths()] == [
        "maskable_ppo_model.zip",
    ]
```

Add compatibility test similar to DQN compatibility if desired:

```python
def test_maskable_ppo_compatibility_requires_three_channel_shape(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import app

    model_path = tmp_path / "maskable_ppo_model.zip"
    model_path.touch()
    monkeypatch.setattr(app, "_maskable_ppo_available", lambda: True)
    monkeypatch.setattr(app, "_selected_maskable_ppo_model_path", lambda: model_path)

    fake_agent = SimpleNamespace(
        model=SimpleNamespace(observation_space=SimpleNamespace(shape=(5 * 5 * 3,)))
    )
    monkeypatch.setattr(app, "_load_maskable_ppo_agent", lambda *_args: fake_agent)
    assert app._maskable_ppo_compatible(5, 5)

    fake_agent.model.observation_space.shape = (5 * 5 * 4,)
    assert not app._maskable_ppo_compatible(5, 5)
```

## Acceptance criteria

```bash
uv run pytest tests/test_app_imports.py -q
uv run streamlit run app.py
```

Expected:

- app starts without trained MaskablePPO model,
- if no model exists, training instruction appears,
- if model exists, MaskablePPO appears in agent dropdown,
- DQN dropdown does not show `maskable_ppo_*.zip`,
- benchmark tab includes MaskablePPO when compatible.

---

# Phase 11 — Optional DQN distribution-kwargs support

This phase is optional but recommended for symmetric experiments.

Apply `experiments.common.add_distribution_args` and `build_distribution_kwargs` to:

```text
experiments/train_dqn.py
experiments/evaluate_dqn.py
```

## Task 11.1 — Update `train_dqn.py`

- Import helper:

```python
from experiments.common import add_distribution_args, build_distribution_kwargs
```

- Add `distribution_kwargs` parameter to `make_dqn_env`.
- Pass it into `ProbMinesweeperEnv`.
- Add `add_distribution_args(parser)` after `--distribution`.
- In `train(args)`, call `distribution_kwargs = build_distribution_kwargs(args)` and pass to `make_dqn_env`.

## Task 11.2 — Update `evaluate_dqn.py`

- Import helper.
- Add `add_distribution_args(parser)` after `--distribution`.
- Build `distribution_kwargs`.
- Pass to `evaluate_agent`.

## Acceptance criteria

```bash
uv run python experiments/train_dqn.py --timesteps 1000 --distribution constant --p 0.15
uv run python experiments/evaluate_dqn.py --episodes 5 --distribution constant --p 0.15
```

---

# Phase 12 — README updates

## Task 12.1 — Add MaskablePPO instructions

Edit:

```text
README.md
```

Add section:

```md
### MaskablePPO agent

MaskablePPO is provided through `sb3-contrib`. It uses the environment's valid-action
mask during training and inference. This addresses the main limitation of the basic
DQN setup, where invalid revealed-cell actions are not masked during learning.

Install RL dependencies:

```bash
uv sync --dev --extra rl
```

Train the default hidden-risk 5x5 model:

```bash
uv run python experiments/train_maskable_ppo.py --timesteps 100000
```

Evaluate it:

```bash
uv run python experiments/evaluate_maskable_ppo.py --episodes 500
```

Compare all available agents:

```bash
uv run python experiments/compare_rl_agents.py --episodes 500
```

The default model is saved to:

```text
models/maskable_ppo_prob_minesweeper.zip
```

MaskablePPO models are tied to board size and observation shape, like DQN models.
Use matching `--width`, `--height`, `--obs-mode`, `--clue-mode`,
`--initial-reveal`, and `--reward-mode` during training and evaluation.
```

## Task 12.2 — Add two-regime experiment commands

Add:

```md
### Recommended experiments

Easier learning regime:

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

Hard hidden-risk stress test:

```bash
uv run python experiments/train_maskable_ppo.py \
  --timesteps 100000 \
  --width 5 \
  --height 5 \
  --distribution correlated \
  --obs-mode state \
  --clue-mode actual_count \
  --initial-reveal safe_2x2 \
  --reward-mode completion \
  --output models/maskable_ppo_hidden_risk_safe2x2_100k.zip
```
```

## Task 12.3 — Update agents section

Add:

```md
- **MaskablePPO** loads a trained sb3-contrib masked policy-gradient model and uses
  `action_mask` to avoid already revealed cells during training and inference.
```

Also clarify:

```md
- **DQN** uses random valid-action fallback for fair evaluation when it predicts an
  invalid revealed cell. Its invalid-action rate is reported in comparison runs.
```

---

# Phase 13 — Report and defense notes

## Task 13.1 — Update report

Edit:

```text
report/report.md
```

Add subsection:

```md
## MaskablePPO jako model z maskowaniem akcji

Drugim modelem RL jest MaskablePPO z pakietu `sb3-contrib`. W przeciwieństwie do
podstawowego DQN, MaskablePPO wykorzystuje maskę dozwolonych akcji podczas uczenia
i predykcji. W naszym środowisku maska blokuje pola już odkryte, więc agent nie traci
kroków treningowych na akcje, które nie zmieniają stanu planszy.

PPO uczy polityki stochastycznej bezpośrednio, a nie funkcji wartości każdej akcji
tak jak DQN. Wariant z maską lepiej pasuje do Sapera, ponieważ liczba formalnie
dostępnych akcji jest stała, ale część akcji staje się niedozwolona po odkryciu pól.
```

## Task 13.2 — Add two result tables

Do not invent numbers. Add placeholders until experiments are run.

### Table 1 — Easier learning regime

```md
| Agent | Epizody | Wygrane | Porażki | Ucięte | Win rate | Śr. nagroda | Śr. kroki |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Min-risk (oracle) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| DQN (random fallback) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| MaskablePPO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
```

Add below table:

```md
DQN invalid-action rate: TODO
```

### Table 2 — Hard hidden-risk stress test

```md
| Agent | Epizody | Wygrane | Porażki | Ucięte | Win rate | Śr. nagroda | Śr. kroki |
|---|---:|---:|---:|---:|---:|---:|---:|
| Random | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| Min-risk (oracle) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| DQN (random fallback) | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| MaskablePPO | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
```

Add below table:

```md
DQN invalid-action rate: TODO
```

Suggested explanation:

```md
Łatwiejszy reżim pokazuje, czy pipeline RL uczy się użytecznej polityki w warunkach,
w których gra jest realnie wygrywalna. Trudny reżim `correlated` traktujemy jako
stress test. Jeśli nawet Min-risk oracle ma niski win rate, to średnia nagroda, średnia
liczba kroków i invalid-action rate DQN są bardziej informacyjne niż sam win rate.
```

## Task 13.3 — Update defense notes

Edit:

```text
report/defense_notes.md
```

Add:

```md
24. **Dlaczego dodano MaskablePPO?** Ponieważ podstawowy DQN ze Stable-Baselines3
    nie używa `action_mask` podczas treningu. MaskablePPO obsługuje maskowanie akcji,
    więc lepiej pasuje do gry, w której część pól staje się niedozwolona.

25. **Czym różni się DQN od MaskablePPO?** DQN uczy funkcji wartości akcji `Q(s,a)`,
    a PPO uczy bezpośrednio polityki wyboru akcji. MaskablePPO dodatkowo zeruje
    prawdopodobieństwo wyboru akcji niedozwolonych.

26. **Czy Min-risk jest uczciwym baseline'em?** Nie w hidden-risk. Min-risk korzysta
    z ukrytego `p_mine`, którego modele RL w trybie `state` nie obserwują. Dlatego
    jest oracle/reference, a nie fair baseline.

27. **Po co mierzyć invalid-action rate DQN?** To pokazuje, jak często DQN wybiera
    formalnie istniejącą, ale niedozwoloną akcję, czyli odkryte pole. MaskablePPO
    rozwiązuje ten problem przez maskowanie podczas uczenia i predykcji.

28. **Dlaczego są dwa reżimy eksperymentów?** Łatwiejszy reżim sprawdza, czy agent
    potrafi nauczyć się użytecznej polityki, a trudny reżim hidden-risk pokazuje
    ograniczenia i trudność oryginalnego środowiska.
```

---

# Phase 14 — Final smoke workflow

Run after all implementation phases:

```bash
uv sync --dev --extra rl
uv run pytest -q
uv run python experiments/train_maskable_ppo.py --timesteps 1000 --verbose 0
uv run python experiments/evaluate_maskable_ppo.py --episodes 5
uv run python experiments/compare_rl_agents.py --episodes 5
```

Expected:

- all tests pass,
- MaskablePPO model trains,
- MaskablePPO evaluation works,
- comparison script works,
- DQN invalid-action rate prints if DQN model exists,
- no crash if DQN or MaskablePPO model is missing.

---

# Phase 15 — Real experiment workflow

## 15.1 Easier learning regime

Train MaskablePPO:

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

Compare:

```bash
uv run python experiments/compare_rl_agents.py \
  --episodes 500 \
  --width 5 \
  --height 5 \
  --distribution constant \
  --p 0.15 \
  --obs-mode state \
  --clue-mode actual_count \
  --initial-reveal safe_2x2 \
  --reward-mode completion \
  --seed 123 \
  --maskable-ppo-model models/maskable_ppo_easy_constant_p015_100k.zip
```

If DQN should be included in this regime, train/evaluate DQN with matching distribution kwargs after Phase 11.

## 15.2 Hard hidden-risk stress test

Train MaskablePPO:

```bash
uv run python experiments/train_maskable_ppo.py \
  --timesteps 100000 \
  --width 5 \
  --height 5 \
  --distribution correlated \
  --obs-mode state \
  --clue-mode actual_count \
  --initial-reveal safe_2x2 \
  --reward-mode completion \
  --output models/maskable_ppo_hidden_risk_safe2x2_100k.zip
```

Compare:

```bash
uv run python experiments/compare_rl_agents.py \
  --episodes 500 \
  --width 5 \
  --height 5 \
  --distribution correlated \
  --obs-mode state \
  --clue-mode actual_count \
  --initial-reveal safe_2x2 \
  --reward-mode completion \
  --seed 123 \
  --maskable-ppo-model models/maskable_ppo_hidden_risk_safe2x2_100k.zip
```

## 15.3 Optional longer run

If 100k steps gives promising results:

```bash
uv run python experiments/train_maskable_ppo.py \
  --timesteps 300000 \
  --width 5 \
  --height 5 \
  --distribution constant \
  --p 0.15 \
  --obs-mode state \
  --clue-mode actual_count \
  --initial-reveal safe_2x2 \
  --reward-mode completion \
  --output models/maskable_ppo_easy_constant_p015_300k.zip
```

---

# Phase 16 — Definition of Done

Implementation is complete when:

- `sb3-contrib` is installed through the RL extra with SB3 minor-version compatibility.
- `ProbMinesweeperEnv.action_masks()` exists and matches `info["action_mask"]`.
- `FlattenObservationWrapper.action_masks()` delegates masks to the wrapped env.
- `experiments/common.py` builds distribution kwargs from CLI args.
- `experiments/train_maskable_ppo.py` trains and saves a MaskablePPO model.
- `experiments/evaluate_maskable_ppo.py` evaluates a saved MaskablePPO model.
- `MaskablePPOAgent` exists and uses `action_masks` during prediction.
- `prob_minesweeper.agents.__init__` exports `MaskablePPOAgent`.
- `DQNAgent` defaults to random valid-action fallback, not Min-risk fallback.
- `DQNAgent` reports `invalid_action_rate`.
- `experiments/compare_rl_agents.py` compares Random, Min-risk oracle, DQN, and MaskablePPO when available.
- DQN dropdown excludes `maskable_ppo_*.zip`.
- MaskablePPO dropdown includes only `maskable_ppo_*.zip`.
- Streamlit can select and run MaskablePPO when a compatible model exists.
- README contains MaskablePPO training/evaluation/comparison commands.
- Report contains two first-class result regimes: easier learning and hard stress test.
- Defense notes explain DQN vs MaskablePPO, Min-risk oracle, and invalid-action rate.
- Smoke workflow passes:

```bash
uv sync --dev --extra rl
uv run pytest -q
uv run python experiments/train_maskable_ppo.py --timesteps 1000 --verbose 0
uv run python experiments/evaluate_maskable_ppo.py --episodes 5
uv run python experiments/compare_rl_agents.py --episodes 5
```
