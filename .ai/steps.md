1. **Scaffold project** — `pyproject.toml` (uv), package structure, empty modules, gymnasium dependency
2. **`MineDistribution` ABC + implementations** — `CorrelatedDistribution` (Gaussian blur), `UniformDistribution`, `ConstantDistribution`
3. **Board state logic** — cell dataclass, lazy reveal sampling, neighbour p-sum calculation, win/loss detection
4. **`RewardConfig`** — dataclass + `risk_adjusted()`, `sparse()`, `uniform()` factories
5. **`ProbMinesweeperEnv` core** — `reset()`, `step()`, observation tensor (both `obs_mode`s), `action_mask`, termination/truncation
6. **Gymnasium registration** — `__init__.py` entry point, `gym.make("ProbMinesweeper-v0")` works
7. **Rendering** — ASCII renderer (`human` mode), `to_rgb_array()` (`rgb_array` mode)
8. **CLI** — `play` command (human input loop), `benchmark` command (random agent stats)
9. **Tests** — `env_checker`, termination, win condition, mask correctness, reward shape, obs channels, seed reproducibility
10. **README** — install instructions, quickstart snippet, link to CLAUDE.md