# boom-boom — Implementation Plan (Probabilistic Minesweeper + Rainbow + Policy-Viz Frontend)

> **Audience:** Claude Code (agentic).
> **Goal of the project:** University computational-intelligence project, **reinforcement-learning** track. Grade target **4.0** = *application (backend + frontend) + a **modified** existing model*. The model is a **modification of Kaixhin/Rainbow** trained on the existing `prob_minesweeper` Gymnasium environment; the frontend is a **policy visualizer** that lets a human watch the trained agent play and inspect its reasoning.
> **Defense:** 3 questions on code + the math of the model, each correct answer +0.5. Understanding the math is as valuable as the implementation — Stage 8 produces the material for this.

---

## How to use this plan

1. Execute **one stage at a time, in order.** Do not start a stage until the previous stage's **Acceptance criteria** all pass.
2. At each **CHECKPOINT**, stop and summarize what changed, paste the verification command output, and wait for human go-ahead before continuing.
3. Make a git commit at the end of every stage with a conventional-commit message (e.g. `feat(rainbow): add masked act()`), so each stage is revertible.
4. If a step is ambiguous, prefer the smallest change that satisfies the acceptance criterion and note the assumption in the commit body. Do **not** invent new scope.
5. **Never modify `prob_minesweeper/` in a way that breaks `uv run pytest`.** The environment is finished and tested (79 tests); treat it as a frozen dependency. If you believe the env needs a change, stop and ask.

---

## Global invariants (hold for the whole project)

- **Python 3.11+, `uv`** for the RL/backend side. Frontend is **React + TypeScript + yarn**.
- **Separation of concerns:** training/agent code must not import any web/GUI code, and the backend must not embed training logic. The frontend talks to the backend only over HTTP. (Same discipline as the env's module split.)
- **Reproducibility:** every run takes a `--seed`; the same seed reproduces the same episode stream.
- **Headline metric is win-rate**, not reward. Reuse `prob_minesweeper.cli.BenchmarkStats` semantics.
- **License:** files adapted from Kaixhin/Rainbow (MIT) keep a header noting the origin and what was changed.
- The existing env's observation is channels-last `(H, W, C)`; PyTorch convs need `(C, H, W)`. The adapter is the only place this transpose happens.
- `obs_mode="state+prob"` (4 channels) is the default training mode (agent sees per-cell `p_mine`). `"state"` (3 channels) is the harder variant — keep it switchable via a flag, but train `state+prob` first.

---

## Target repository layout (create as you go)

```
boom-boom/
├── prob_minesweeper/        # EXISTING env — frozen, do not break
├── agents/
│   └── greedy.py            # Stage 1
├── rainbow/                 # vendored + modified Kaixhin/Rainbow
│   ├── agent.py             # modified (masking)            Stage 3
│   ├── model.py             # PROVIDED (replaces original)  Stage 2
│   ├── memory.py            # modified (float-safe)         Stage 3
│   ├── env_adapter.py       # PROVIDED                      Stage 2
│   ├── main.py              # modified (wiring + args)      Stage 3
│   └── test.py              # modified (win-rate)           Stage 4
├── serve/
│   ├── inference.py         # load checkpoint, run agent    Stage 6
│   └── app.py               # FastAPI                       Stage 6
├── frontend/                # React + TS (yarn)             Stage 7
├── scripts/
│   ├── train.sh
│   └── eval.sh
├── reports/
│   ├── mdp_model.md         # math formalization            Stage 8
│   ├── results.md           # tables + plots                Stage 5/8
│   └── defense_qa.md        # Stage 8
├── .ai/
│   ├── AGENTS.md            # update non-goals               Stage 8
│   └── specs/
│       ├── training.md      # Stage 8
│       └── frontend.md      # Stage 8
├── checkpoints/             # .gitignored
└── runs/                    # logs/metrics, .gitignored
```

## Provided files (already written — do not regenerate from scratch)

- `env_adapter.py` → place at `rainbow/env_adapter.py`.
- `model.py` → place at `rainbow/model.py` (replaces the original).

Both document the state contract `(C, H, W)`, `history_length=1` at the top. Read them before Stage 2.

---

# STAGE 0 — Setup & vendoring

**Goal:** Working repo skeleton with Rainbow's source vendored and dependencies installed.

**Steps**
1. Ensure `pyproject.toml` adds RL/backend deps to a new optional group, e.g. `[project.optional-dependencies] rl = ["torch", "matplotlib", "fastapi", "uvicorn[standard]", "pydantic"]`. Keep `gymnasium`, `numpy`, `scipy` as-is. Run `uv sync --extra rl --dev`.
2. Clone Kaixhin/Rainbow into a temp dir and copy **only** `agent.py`, `memory.py`, `main.py`, `test.py` into `rainbow/`. (Do **not** copy its `env.py` or `model.py` — those are replaced by the provided files.) Add an `__init__.py` so `rainbow` is importable.
3. Drop the two provided files into `rainbow/env_adapter.py` and `rainbow/model.py`. Fix their internal imports to the `rainbow` package layout.
4. Add `checkpoints/` and `runs/` to `.gitignore`.

**Acceptance criteria**
- `uv run pytest -q` still passes (env untouched).
- `uv run python -c "import rainbow.model, rainbow.env_adapter"` imports without error.

**CHECKPOINT 0** — report dependency versions (esp. `torch`) and confirm imports.

---

# STAGE 1 — Greedy baseline

**Goal:** A non-learning baseline stronger than random, so later "agent beats greedy" is a meaningful result and the report has a fair comparison.

**Steps**
1. Create `agents/greedy.py` with a function `run_greedy(*, episodes, width, height, distribution, distribution_kwargs=None, seed=None) -> BenchmarkStats` that mirrors the signature of `prob_minesweeper.cli.run_benchmark`.
2. Policy: at each step reveal the **unrevealed cell with the lowest `p_mine`**. Read `p_mine` from the env (`env.board.p_mine_field()`) masked by `info["action_mask"]`; break ties deterministically (lowest flat index). Reuse the env's `reset/step` and the `BenchmarkStats` accumulation pattern from the CLI.
3. Add a CLI subcommand or a small `__main__` so it can be run like the existing benchmark.

**Acceptance criteria**
- On `correlated` 9×9, 1000 episodes, fixed seed: greedy **win-rate > random win-rate** (run the existing random benchmark for the same config to compare). Print both.

**CHECKPOINT 1** — report greedy vs random win-rates.

---

# STAGE 2 — Env adapter + network integration

**Goal:** The provided adapter and network instantiate against this env with correct shapes.

**Steps**
1. Confirm `rainbow/env_adapter.py` constructs `ProbMinesweeperEnv` and that `reset()` returns a `(C, H, W)` float32 tensor and `step()` returns `(state, reward, done)`.
2. Write a throwaway shape test `rainbow/_smoke_shapes.py` (or a pytest in `tests/test_rainbow.py`) that:
   - builds a tiny `args` namespace (`board_width=9, board_height=9, obs_mode="state+prob", obs_channels=4, atoms=51, hidden_size=256, noisy_std=0.1, device="cpu", seed=0`),
   - creates `Env(args)` and `DQN(args, action_space=81)`,
   - asserts `env.reset().shape == (4, 9, 9)`,
   - asserts `DQN(...)(env.reset().unsqueeze(0)).shape == (1, 81, 51)`,
   - asserts `env.action_mask().shape == (81,)` and dtype bool.
3. Verify `obs_channels` is derived correctly for both obs modes (3 vs 4).

**Acceptance criteria**
- `uv run pytest tests/test_rainbow.py -q` passes.
- A forward pass on `state+prob` and `state` both produce `(1, A, atoms)` without shape errors.

**CHECKPOINT 2** — report both forward-pass shapes.

---

# STAGE 3 — Replay, masking, training wiring (end-to-end run)

**Goal:** `main.py` runs a (tiny) training loop end-to-end on this env with action masking. This is the core of the "modification" deliverable.

### 3a. Float-safe replay (`rainbow/memory.py`)
Rainbow's buffer is built for 8-bit Atari frames. Fix:
- Find the **blank/transition state** definition (Atari uses `torch.zeros(84, 84, dtype=torch.uint8)`); change it to shape **`(C, H, W)` float32** using config values.
- Find where sampled states are converted with `.to(dtype=torch.float32).div_(255)`; **remove the `/255`** and keep float32 (our values live in `[0, 8]` — dividing by 255 destroys them).
- Disable temporal stacking: with `history_length = 1`, the stored state is already a full `(C, H, W)` observation. Ensure the returned batch is `(batch, C, H, W)`, not `(batch, 1, C, H, W)`.

### 3b. Masked action selection (`rainbow/agent.py`)
Modify `act()` (and `act_e_greedy`/eval variant if present) to accept and apply a mask:
```python
def act(self, state, action_mask=None):
    with torch.no_grad():
        q = (self.online_net(state.unsqueeze(0)) * self.support).sum(2)  # (1, A) expected Q
        if action_mask is not None:
            q = q.masked_fill(~action_mask.unsqueeze(0), float("-inf"))
        return q.argmax(1).item()
```
**Optional refinement (do not block on it):** also mask the next-state argmax in the Double-DQN target inside `learn()`. This requires storing the next-state mask in the buffer. Skip for the 4.0 baseline; note it in `reports/defense_qa.md` as a known approximation.

### 3c. Training wiring (`rainbow/main.py`)
- Remove the `--game` argument and ALE/Atari-only flags.
- Add args: `--board-width`, `--board-height`, `--distribution`, `--obs-mode`, and derive `obs_channels` (3 or 4) from `obs-mode`. Set `obs_channels`, `board_height`, `board_width` onto `args` before constructing `DQN`.
- Set `history_length = 1`.
- Replace Atari `Env` construction with `from rainbow.env_adapter import Env`.
- At every action selection in the train and eval loops, pass the mask: `action = dqn.act(state, env.action_mask())`.
- **Distributional support (`V-min`/`V-max`) — critical:** Atari defaults (`-10/10`) do not match this reward scale. See the hyperparameter appendix; expose `--V-min`/`--V-max` and set sane defaults for a 9×9 board.
- Treat `done` as terminal for bootstrapping (with masking, episodes end in win/loss well before `max_steps`, so truncation almost never fires).

**Acceptance criteria**
- A **smoke run** completes without error:
  `uv run python -m rainbow.main --board-width 9 --board-height 9 --obs-mode state+prob --T-max 2000 --learn-start 500 --memory-capacity 5000 --evaluation-interval 1000 --seed 0`
- No `nan` loss in the first 1000 learn steps; the eval win-rate is printed.

**CHECKPOINT 3** — paste the smoke-run log (loss not NaN, one eval printed).

---

# STAGE 4 — Train, checkpoint, log

**Goal:** A real training run that produces a saved checkpoint and a metrics log.

**Steps**
1. Ensure `main.py` (a) checkpoints the online net to `checkpoints/{run_name}/model.pt` at each evaluation interval and at the end, and (b) appends `{step, win_rate, mean_reward, mean_steps}` to `runs/{run_name}/metrics.jsonl`.
2. Add `scripts/train.sh` with the full-run command (see appendix for hyperparameters).
3. Run the full training on `correlated` 9×9, `state+prob`.

**Acceptance criteria**
- A checkpoint file exists and reloads: `torch.load(...)` into a fresh `DQN` without key mismatch.
- `runs/{run_name}/metrics.jsonl` shows win-rate **trending up** and the final eval win-rate **beats the Stage-1 greedy baseline**. (If it does not after a reasonable budget, stop and report — likely `V-min/V-max` or `learn-start` mis-set; do not silently tune forever.)

**CHECKPOINT 4** — report final win-rate vs greedy, and attach the metrics file.

---

# STAGE 5 — Evaluation & comparison

**Goal:** Defensible results: learning curve + a sweep comparing agent / greedy / random.

**Steps**
1. Modify `rainbow/test.py` (or add `rainbow/evaluate.py`) to load a checkpoint and report **win-rate** (not just reward) over N episodes, using masked greedy action selection (`act` in eval/noise-off mode).
2. Produce, into `reports/`:
   - **Learning curve:** win-rate vs training step (from `metrics.jsonl`), as a matplotlib PNG.
   - **Sweep table** (`results.md`): win-rate for **agent vs greedy vs random** across board sizes `{5×5, 9×9, 16×16}` and distributions `{correlated, uniform, constant(p=0.2)}`. Use a fixed eval seed and ≥1000 episodes per cell.
3. Save the raw numbers as `reports/results.json` so the table is reproducible.

**Acceptance criteria**
- `reports/results.md` renders a complete table; `reports/learning_curve.png` exists.
- Agent ≥ greedy on at least the trained config; note any configs where it is not (honesty matters for the defense).

**CHECKPOINT 5** — attach the table and curve.

---

# STAGE 6 — FastAPI inference backend

**Goal:** A small API that drives one episode of the trained agent and exposes its reasoning for the frontend. **No training code here.**

**Steps**
1. `serve/inference.py`: a `Policy` class that loads a checkpoint + builds the env via the adapter, and exposes:
   - `reset(seed, width, height, distribution, obs_mode) -> EpisodeState`
   - `agent_action() -> int` (masked argmax of expected Q)
   - `step(action) -> EpisodeState`
   - `expected_q_grid() -> H×W float` (expected Q per cell = `sum(softmax_dist * support)`, `-inf`/`null` for revealed cells)
   - `value_distribution(cell) -> {support: float[atoms], probs: float[atoms]}` (the C51 distribution — this is the "show the math" artifact)
   - `true_p_mine_grid() -> H×W float` (ground-truth risk, for the overlay)
2. `serve/app.py`: FastAPI with `POST /reset`, `POST /step` (body `{action?: int}` — if omitted, the agent picks), and include in every response: `board` (per-cell: revealed?/clue/mine), `mask`, `expected_q`, `true_p_mine`, `chosen_action`, `value_distribution` for the chosen cell, `reward`, `cum_reward`, `done`, `outcome` (`win`/`loss`/`ongoing`). Enable CORS for the dev frontend origin.
3. `(Optional, not required for the local demo)` add a `Dockerfile` for the API.

**Acceptance criteria**
- `uvicorn serve.app:app` starts; `curl -X POST localhost:8000/reset` returns a full JSON state; repeated `/step` plays an episode to `win`/`loss`.
- `expected_q` is `null` exactly on revealed cells; `value_distribution.probs` sums to ≈1.

**CHECKPOINT 6** — paste a `/reset` then two `/step` responses.

---

# STAGE 7 — React policy-viz frontend

**Goal:** A clean single-page app to **watch the trained agent play** and read its reasoning. This is the defense centerpiece.

**Stack:** React + TypeScript, **yarn**, Vite. Charts via `recharts`. No browser storage; all state in React hooks.

**Components**
1. `BoardGrid` — H×W grid. Each cell shows its symbol (unrevealed / continuous clue / mine) and is tinted by the active overlay.
2. `Controls` — Reset (new seed + config selectors for size/distribution), Step (one agent move), Auto-play (interval, with a speed slider), and a Manual mode toggle (click a cell to force that action via `/step {action}`).
3. `Overlays` — radio/toggles: **Expected-Q heatmap** (sequential color scale; the argmax cell is outlined = "next move"), **True p_mine** (separate diverging scale), and **Clues only**. Show only one heatmap at a time; always allow the clue text.
4. `DistributionChart` — `recharts` bar chart of the **C51 value distribution** for the currently selected/argmax cell (x = support atoms, y = probability). Label the mean. This is the literal picture of distributional RL.
5. `EpisodeStatus` — cumulative reward, step count, outcome banner.

**Design direction (build it intentionally, not default-bootstrap):**
- Restrained, corporate/dashboard palette; one accent color for "agent's next move".
- Monospace for all numeric cell values and chart axes; consistent 8px spacing scale; the board and the distribution chart side-by-side on wide screens, stacked on narrow.
- Color scales must be colorblind-safe and have a visible legend; never encode meaning by hue alone (also outline the argmax cell).

**Data flow:** on Reset/Step, call the backend, store the returned `EpisodeState` in a hook, and re-render grid + chart from it. Selecting a cell requests/uses that cell's `value_distribution`.

**Acceptance criteria**
- `yarn dev` runs; with the backend up, Reset shows a board, Step advances the agent, and the Q-heatmap's outlined cell matches the agent's actual next move.
- Toggling to True-p_mine visibly shows whether high-Q cells coincide with low-risk cells (the key "did it learn risk?" demo).
- The distribution chart updates per step and its bars sum to ≈1.

**CHECKPOINT 7** — describe the running UI and note any backend contract gaps.

---

# STAGE 8 — Docs, spec, and defense prep

**Goal:** The written artifacts that satisfy "do it properly" and arm you for the 3 defense questions (+0.5 each).

**Steps**
1. **`reports/mdp_model.md` — the mathematical model (PEU_U02).** Formalize the environment as an MDP:
   - State space (partially-revealed grid + continuous clue values; the `(H,W,C)` observation tensor).
   - Action space `Discrete(H·W)` with the legality mask.
   - Transition kernel: per-cell mine realization `~ Bernoulli(p_mine)` sampled at reveal; clue = `round(Σ neighbour p_mine, 1)`.
   - Reward function (risk-adjusted: safe `1−p`, mine `−p`, win bonus `+1`).
   - Objective: discounted return; note γ.
2. **`reports/defense_qa.md` — math of the model.** Concise, correct explanations you can defend, each tied to where it lives in code:
   - Distributional RL / C51: categorical Bellman operator, the projection step onto the fixed support, why `V-min/V-max` must bracket returns (link to your chosen values).
   - Prioritized replay: TD-error priorities, importance-sampling weights / bias correction.
   - Dueling: `Q = V + (A − mean A)` and why the mean-subtraction identifiability trick.
   - Noisy Nets: factorized Gaussian noise as learned exploration (replaces ε-greedy).
   - Double DQN target; n-step returns.
   - The masking approximation (acting-time vs target-time) from Stage 3b.
3. **`.ai/specs/training.md` and `.ai/specs/frontend.md`** — distilled specs derived from this plan (contracts, file responsibilities, acceptance criteria), so the codebase stays as disciplined as the env.
4. **Update `.ai/AGENTS.md`** — the GUI and a learning agent were v1 *non-goals*; move them into scope and record the architecture (env ↔ rainbow ↔ serve ↔ frontend boundaries).
5. **README**: add a "Train / Evaluate / Serve / Frontend" quickstart and the results table.

**Acceptance criteria**
- All five artifacts exist and are internally consistent with the code (file paths, hyperparameters, and metrics match what was actually run).

**CHECKPOINT 8** — list the artifacts and flag anything in them you could not verify against the code.

---

## Appendix A — Hyperparameter starting point (9×9, `state+prob`)

Start from **data-efficient Rainbow**, scaled down for a tiny, fast env:

| Arg | Start value | Note |
|---|---|---|
| `T-max` | 200_000 | total env steps; raise if still improving |
| `learn-start` | 1_600 | steps before first gradient update |
| `memory-capacity` | 100_000 | |
| `replay-frequency` | 1 | |
| `multi-step` | 3 | |
| `target-update` | 2_000 | |
| `batch-size` | 32 | |
| `hidden-size` | 256 | |
| `learning-rate` | 1e-4 | Adam |
| `discount (γ)` | 0.99 | |
| `atoms` | 51 | C51 |
| `noisy-std` | 0.1 | |
| `V-min` / `V-max` | **−5 / 40** | **Must bracket the discounted return.** A 9×9 win accrues ≈ Σ(1−p) over safe cells + 1 (tens of points). After the smoke run, check returns fall inside `[V-min, V-max]`; widen `V-max` for bigger boards, shrink for `constant(p)` high-`p` boards. This is the most common silent failure — get it right. |

For board sizes other than 9×9, recompute `V-max` from the reward range and adjust `T-max` upward with board area.

## Appendix B — Order of operations (dependency graph)

```
Stage 0 ─ setup
   └─ Stage 1 greedy baseline ─────────────┐ (needed for Stage 4/5 comparison)
   └─ Stage 2 adapter+model shapes          │
        └─ Stage 3 replay+mask+wiring        │
             └─ Stage 4 train+checkpoint ────┴─→ Stage 5 eval
                  └─ Stage 6 backend (needs checkpoint)
                       └─ Stage 7 frontend (needs backend)
   Stage 8 docs ── write incrementally; finalize after Stage 7
```

## Appendix C — Done definition for the 4.0 deliverable

- [ ] Env untouched, all original tests pass.
- [ ] Rainbow modified (adapter, resized net, float-safe replay, masking, rewired main) — trains and beats greedy.
- [ ] Evaluation table + learning curve in `reports/`.
- [ ] FastAPI backend serves a live episode with per-cell expected Q, C51 distribution, and true-`p_mine` overlay.
- [ ] React policy-viz frontend renders all of the above with play/step controls.
- [ ] `mdp_model.md` + `defense_qa.md` written and consistent with the code.