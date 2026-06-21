"""Streamlit frontend for probabilistic Minesweeper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import streamlit as st

from prob_minesweeper.agents import DQNAgent, MinRiskAgent, RandomAgent
from prob_minesweeper.env import ProbMinesweeperEnv
from prob_minesweeper.evaluation import compare_agents
from prob_minesweeper.rewards import make_reward_config


DQN_MODELS_DIR = Path(__file__).resolve().parent / "models"


# Classic Windows-style Minesweeper look. Scoped to the `minefield` container so
# the agent controls and the rest of the page keep the default Streamlit theme.
_MINEFIELD_CSS = """
<style>
/* Beveled board frame (sunken: dark top/left, light bottom/right). */
.st-key-minefield {
    background: #c0c0c0;
    padding: 6px;
    width: fit-content;
    border-top: 3px solid #7b7b7b;
    border-left: 3px solid #7b7b7b;
    border-right: 3px solid #ffffff;
    border-bottom: 3px solid #ffffff;
}

/* Each row stays on a single line; columns are fixed-width squares. */
.st-key-minefield [data-testid="stHorizontalBlock"] {
    gap: 0 !important;
    flex-wrap: nowrap !important;
}
.st-key-minefield [data-testid="stColumn"] {
    width: 36px !important;
    min-width: 36px !important;
    flex: 0 0 36px !important;
}
.st-key-minefield [data-testid="stVerticalBlock"] {
    gap: 0 !important;
}

/* Strip the wrappers Streamlit puts around buttons / markdown. */
.st-key-minefield [data-testid="stButton"],
.st-key-minefield [data-testid="stMarkdown"],
.st-key-minefield [data-testid="stMarkdownContainer"] {
    margin: 0 !important;
    padding: 0 !important;
}
.st-key-minefield [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
}

/* Hidden cells: raised tile. */
.st-key-minefield button {
    width: 36px !important;
    height: 36px !important;
    min-height: 36px !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    font-family: "Courier New", monospace !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    line-height: 1 !important;
    color: #1b1b1b !important;
    background: #c0c0c0 !important;
    border-top: 3px solid #ffffff !important;
    border-left: 3px solid #ffffff !important;
    border-right: 3px solid #808080 !important;
    border-bottom: 3px solid #808080 !important;
}
.st-key-minefield button:hover {
    background: #b8b8b8 !important;
    color: #000 !important;
}
.st-key-minefield button:active {
    border-width: 1px !important;
    border-color: #808080 #c0c0c0 #c0c0c0 #808080 !important;
}
.st-key-minefield button:disabled {
    opacity: 1 !important;
    color: #1b1b1b !important;
    background: #c0c0c0 !important;
}

/* Revealed cells: sunken flat tile. */
.st-key-minefield .ms-cell {
    box-sizing: border-box;
    width: 36px;
    height: 36px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: "Courier New", monospace;
    font-weight: 700;
    font-size: 11px;
    background: #c0c0c0;
    border: 1px solid #7b7b7b;
}
.st-key-minefield .ms-mine {
    background: #ff4d4d;
    font-size: 16px;
    border-color: #b00000;
}

/* Classic number colors. */
.st-key-minefield .ms-n1 { color: #0000ff; }
.st-key-minefield .ms-n2 { color: #008000; }
.st-key-minefield .ms-n3 { color: #ff0000; }
.st-key-minefield .ms-n4 { color: #000080; }
.st-key-minefield .ms-n5 { color: #800000; }
.st-key-minefield .ms-n6 { color: #008080; }
.st-key-minefield .ms-n7 { color: #000000; }
.st-key-minefield .ms-n8 { color: #808080; }
</style>
"""


def create_env(
    width: int,
    height: int,
    distribution: str,
    distribution_kwargs: dict[str, Any],
    clue_mode: str,
    initial_reveal: str,
    reward_mode: str,
) -> ProbMinesweeperEnv:
    return ProbMinesweeperEnv(
        width=width,
        height=height,
        distribution=distribution,
        distribution_kwargs=distribution_kwargs,
        obs_mode="state",
        clue_mode=clue_mode,
        initial_reveal=initial_reveal,
        reward_config=make_reward_config(reward_mode),
    )


def reset_session_state(
    width: int,
    height: int,
    distribution: str,
    distribution_kwargs: dict[str, Any],
    clue_mode: str,
    initial_reveal: str,
    reward_mode: str,
    seed: int,
) -> None:
    old_env = st.session_state.get("env")
    if old_env is not None:
        old_env.close()
    env = create_env(
        width,
        height,
        distribution,
        distribution_kwargs,
        clue_mode,
        initial_reveal,
        reward_mode,
    )
    obs, info = env.reset(seed=seed)
    game_id = st.session_state.get("game_id", 0) + 1
    st.session_state.update(
        env=env,
        obs=obs,
        info=info,
        total_reward=0.0,
        last_reward=0.0,
        terminated=False,
        truncated=False,
        message="Game in progress",
        step_count=0,
        selected_agent="Min-risk (oracle)",
        game_id=game_id,
        game_config=(
            width,
            height,
            distribution,
            distribution_kwargs.copy(),
            clue_mode,
            initial_reveal,
            reward_mode,
            seed,
        ),
    )


def reveal(action: int) -> None:
    if st.session_state.terminated or st.session_state.truncated:
        return
    obs, reward, terminated, truncated, info = st.session_state.env.step(action)
    st.session_state.obs = obs
    st.session_state.info = info
    st.session_state.last_reward = reward
    st.session_state.total_reward += reward
    st.session_state.step_count += 1
    st.session_state.terminated = terminated
    st.session_state.truncated = truncated
    board = st.session_state.env.board
    if truncated:
        st.session_state.message = "Episode truncated: step limit reached"
    elif terminated and board is not None and board.is_win():
        st.session_state.message = "Win: every safe cell was revealed"
    elif terminated:
        st.session_state.message = "Loss: a mine was revealed"
    else:
        st.session_state.message = "Safe reveal"


def _distribution_controls() -> tuple[str, dict[str, float]]:
    distribution = st.sidebar.selectbox(
        "Distribution", ["constant", "uniform", "correlated"], index=2
    )
    if distribution == "constant":
        return distribution, {"p": st.sidebar.slider("Mine probability p", 0.0, 1.0, 0.2)}
    if distribution == "uniform":
        low = st.sidebar.slider("Low", 0.0, 1.0, 0.0)
        high = st.sidebar.slider("High", low, 1.0, 1.0)
        return distribution, {"low": low, "high": high}
    sigma = st.sidebar.number_input("Sigma", 0.1, 10.0, 2.0, 0.1)
    scale = st.sidebar.number_input("Scale", 0.0, 2.0, 1.0, 0.1)
    return distribution, {"sigma": sigma, "scale": scale}


def _agent() -> Any:
    if st.session_state.selected_agent == "Random":
        return RandomAgent()
    if st.session_state.selected_agent == "DQN":
        model_path = _selected_dqn_model_path()
        if model_path is None:
            raise RuntimeError("No DQN model is selected")
        return _load_dqn_agent(str(model_path), model_path.stat().st_mtime_ns)
    return MinRiskAgent()


def _dqn_model_paths() -> list[Path]:
    return sorted(DQN_MODELS_DIR.glob("*.zip"), key=lambda path: path.name.lower())


def _selected_dqn_model_path() -> Path | None:
    model_name = st.session_state.get("dqn_model_selector")
    if not model_name:
        return None
    path = DQN_MODELS_DIR / str(model_name)
    return path if path.is_file() else None


def _dqn_available() -> bool:
    return (
        _selected_dqn_model_path() is not None
        and importlib.util.find_spec("stable_baselines3") is not None
    )


def _dqn_compatible(width: int, height: int) -> bool:
    """Return whether the selected model accepts this app's visible-state input."""
    if not _dqn_available():
        return False
    model_path = _selected_dqn_model_path()
    if model_path is None:
        return False
    try:
        agent = _load_dqn_agent(str(model_path), model_path.stat().st_mtime_ns)
    except (ImportError, OSError, RuntimeError, ValueError):
        return False
    model_space = getattr(agent.model, "observation_space", None)
    expected_shape = getattr(model_space, "shape", None)
    return tuple(expected_shape or ()) == (width * height * 3,)


@st.cache_resource
def _load_dqn_agent(model_path: str, model_mtime_ns: int) -> DQNAgent:
    del model_mtime_ns  # It is part of the cache key and reloads replaced models.
    return DQNAgent(model_path)


def _revealed_cell_html(cell: Any, clue_mode: str) -> str:
    """Render a revealed cell as a sunken Minesweeper tile."""
    if cell.has_mine:
        return '<div class="ms-cell ms-mine">💣</div>'
    if cell.display_value == 0:
        return '<div class="ms-cell"></div>'
    color_index = min(8, max(1, int(round(cell.display_value))))
    value = (
        str(int(cell.display_value))
        if clue_mode == "actual_count"
        else f"{cell.display_value:.1f}"
    )
    return f'<div class="ms-cell ms-n{color_index}">{value}</div>'


def _render_game(show_probabilities: bool) -> None:
    env = st.session_state.env
    board = env.board
    if board is None:
        return
    st.subheader(st.session_state.message)
    st.caption(
        f"Rules: clue_mode={env.clue_mode}, initial_reveal={env.initial_reveal}, "
        f"obs_mode={env.obs_mode}. "
        "Min-risk is an oracle with privileged probability access."
    )
    if show_probabilities:
        st.warning(
            "Hidden probabilities are shown only for human/debug mode. "
            "The DQN observation does not contain them."
        )
    metrics = st.columns(3)
    metrics[0].metric("Last reward", f"{st.session_state.last_reward:.2f}")
    metrics[1].metric("Total reward", f"{st.session_state.total_reward:.2f}")
    metrics[2].metric("Steps", st.session_state.step_count)

    disabled_game = st.session_state.terminated or st.session_state.truncated

    st.markdown(_MINEFIELD_CSS, unsafe_allow_html=True)

    board_container = st.container(key="minefield")
    for row in range(env.height):
        columns = board_container.columns(env.width, gap=None)
        for col in range(env.width):
            action = board.flat_index(row, col)
            cell = board.cell(row, col)
            column = columns[col]
            if cell.is_revealed:
                column.markdown(
                    _revealed_cell_html(cell, env.clue_mode), unsafe_allow_html=True
                )
            else:
                label = f"{cell.p_mine:.2f}" if show_probabilities else "\u00a0"
                column.button(
                    label,
                    key=f"cell_{row}_{col}_{st.session_state.game_id}",
                    disabled=disabled_game
                    or not bool(st.session_state.info["action_mask"][action]),
                    on_click=reveal,
                    args=(action,),
                    use_container_width=True,
                )

    agent_options = ["Random", "Min-risk (oracle)"]
    dqn_compatible = _dqn_compatible(env.width, env.height)
    if dqn_compatible:
        agent_options.append("DQN")
    st.session_state.selected_agent = st.selectbox(
        "Agent", agent_options, index=1, key="agent_selector"
    )
    if not _dqn_model_paths():
        st.info(
            "DQN model not found. Train it with: "
            "`uv run python experiments/train_dqn.py --timesteps 500000`"
        )
    elif not _dqn_available():
        st.info("Install the RL dependencies with: `uv sync --dev --extra rl`")
    elif not dqn_compatible:
        st.info(
            "The selected DQN model is incompatible with this board size or the "
            "3-channel state observation. Select or train a matching model."
        )
    move_col, run_col = st.columns(2)
    if move_col.button("Agent move", disabled=disabled_game, use_container_width=True):
        reveal(_agent().select_action(st.session_state.obs, st.session_state.info, env))
        st.rerun()
    if run_col.button(
        "Run agent until terminal", disabled=disabled_game, use_container_width=True
    ):
        agent = _agent()
        for _ in range(env.width * env.height * 2):
            if st.session_state.terminated or st.session_state.truncated:
                break
            if not st.session_state.info["action_mask"].any():
                break
            reveal(agent.select_action(st.session_state.obs, st.session_state.info, env))
        st.rerun()


def _benchmark_tab(
    width: int,
    height: int,
    distribution: str,
    distribution_kwargs: dict[str, Any],
    clue_mode: str,
    initial_reveal: str,
    reward_mode: str,
    seed: int,
) -> None:
    st.caption(
        f"Benchmark configuration: obs_mode=state, clue_mode={clue_mode}, "
        f"initial_reveal={initial_reveal}, "
        f"reward_mode={reward_mode}. Min-risk has oracle access to p_mine."
    )
    episodes = st.number_input("Episodes", min_value=1, max_value=10_000, value=100)
    if st.button("Run benchmark", type="primary"):
        agents: list[Any] = [RandomAgent(seed), MinRiskAgent()]
        if _dqn_compatible(width, height):
            model_path = _selected_dqn_model_path()
            if model_path is not None:
                agents.append(
                    _load_dqn_agent(str(model_path), model_path.stat().st_mtime_ns)
                )
        with st.spinner("Evaluating agents..."):
            st.session_state.benchmark_results = compare_agents(
                agents,
                episodes=int(episodes),
                width=width,
                height=height,
                distribution=distribution,
                distribution_kwargs=distribution_kwargs,
                obs_mode="state",
                clue_mode=clue_mode,
                initial_reveal=initial_reveal,
                reward_config=make_reward_config(reward_mode),
                seed=seed,
            )
    results = st.session_state.get("benchmark_results")
    if results:
        rows = [
            {
                "Agent": (
                    "Min-risk (oracle)" if r.agent_name == "Min-risk" else r.agent_name
                ),
                "Episodes": r.episodes,
                "Wins": r.wins,
                "Losses": r.losses,
                "Truncated": r.truncated,
                "Win rate": r.win_rate,
                "Mean reward": r.mean_reward,
                "Mean steps": r.mean_steps,
            }
            for r in results
        ]
        st.dataframe(rows, use_container_width=True)
        st.bar_chart({row["Agent"]: row["Win rate"] for row in rows})


def _model_tab() -> None:
    st.markdown(
        r"""
### Reinforcement-learning model

The main RL variant is **hidden-risk**. Mine outcomes are sampled as
$M_i \sim \mathrm{Bernoulli}(p_i)$, but the DQN state contains only revealed-cell
state, revealed clues, and the reserved flag channel. It does not contain $p_i$.
A safe reveal returns the actual neighbouring mine count
$c_i=\sum_{j\in N(i)}M_j$. Each action therefore changes both reward and the
information available for future decisions.

At reset, a random 2×2 opening is guaranteed safe and revealed without reward. One
additional hidden cell is guaranteed safe so reset never returns an already-complete
board. The four visible clues give the policy evidence before its first action.

Completion reward gives `+0.1` for a safe reveal, `-1` for a mine, and an additional
`+10` for winning. The original full-information `state+prob`, probability-sum clue,
and risk-adjusted reward remain available in the backend.

Random chooses uniformly among valid actions. Min-risk reads hidden $p_i$ values and
is therefore an **oracle**, not a fair visible-state baseline for hidden-risk DQN.

### Deep Q-Network

The DQN model learns $Q(s,a)$: the expected long-term value of selecting cell $a$
in board state $s$. Training uses flattened visible-state observations.

The Bellman target and squared temporal-difference loss are

$$y = r + \gamma \max_{a'} Q(s',a'),$$

$$L(\theta) = \left(y - Q_\theta(s,a)\right)^2.$$

Stable-Baselines3 DQN does not apply the environment's `action_mask` during training.
At inference time, an invalid prediction is replaced by a valid action. DQN models
are tied to the training board size and observation mode; incompatible shapes raise
a clear error.
"""
    )


def main() -> None:
    st.set_page_config(page_title="Probabilistic Minesweeper", page_icon="💣", layout="wide")
    st.title("Probabilistic Minesweeper")
    width = st.sidebar.number_input("Width", 2, 12, 5)
    height = st.sidebar.number_input("Height", 2, 12, 5)
    distribution, distribution_kwargs = _distribution_controls()
    clue_label = st.sidebar.selectbox(
        "Clue mode", ["Actual mine count", "Probability sum"]
    )
    clue_mode = {
        "Actual mine count": "actual_count",
        "Probability sum": "prob_sum",
    }[clue_label]
    initial_options = (
        ["Safe 2x2 opening", "None"]
        if int(width) * int(height) > 4
        else ["None"]
    )
    initial_label = st.sidebar.selectbox("Initial reveal", initial_options)
    if len(initial_options) == 1:
        st.sidebar.caption("A safe 2x2 opening needs at least one cell outside it.")
    initial_reveal = {
        "Safe 2x2 opening": "safe_2x2",
        "None": "none",
    }[initial_label]
    reward_label = st.sidebar.selectbox(
        "Reward mode", ["Completion", "Risk-adjusted"]
    )
    reward_mode = {
        "Completion": "completion",
        "Risk-adjusted": "risk_adjusted",
    }[reward_label]
    seed = st.sidebar.number_input("Seed", value=42)
    show_probabilities = st.sidebar.checkbox("Show hidden probabilities", value=False)
    dqn_models = _dqn_model_paths()
    if dqn_models:
        model_names = [path.name for path in dqn_models]
        preferred_model = "dqn_hidden_risk_safe2x2_500k.zip"
        st.sidebar.selectbox(
            "DQN model",
            model_names,
            index=(
                model_names.index(preferred_model)
                if preferred_model in model_names
                else 0
            ),
            key="dqn_model_selector",
        )
    config = (
        int(width),
        int(height),
        distribution,
        distribution_kwargs.copy(),
        clue_mode,
        initial_reveal,
        reward_mode,
        int(seed),
    )
    if "env" not in st.session_state:
        reset_session_state(*config)
    if st.sidebar.button("New game", type="primary"):
        reset_session_state(*config)

    game_tab, benchmark_tab, model_tab = st.tabs(["Game", "Benchmark", "Model"])
    with game_tab:
        _render_game(show_probabilities)
    with benchmark_tab:
        _benchmark_tab(*config)
    with model_tab:
        _model_tab()


if __name__ == "__main__":
    main()
