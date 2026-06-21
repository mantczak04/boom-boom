"""Streamlit frontend for probabilistic Minesweeper."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import streamlit as st

from prob_minesweeper.agents import DQNAgent, MinRiskAgent, RandomAgent
from prob_minesweeper.env import ProbMinesweeperEnv
from prob_minesweeper.evaluation import compare_agents


DQN_MODEL_PATH = Path(__file__).resolve().parent / "models" / "dqn_prob_minesweeper.zip"


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
) -> ProbMinesweeperEnv:
    return ProbMinesweeperEnv(
        width=width,
        height=height,
        distribution=distribution,
        distribution_kwargs=distribution_kwargs,
        obs_mode="state+prob",
    )


def reset_session_state(
    width: int,
    height: int,
    distribution: str,
    distribution_kwargs: dict[str, Any],
    seed: int,
) -> None:
    old_env = st.session_state.get("env")
    if old_env is not None:
        old_env.close()
    env = create_env(width, height, distribution, distribution_kwargs)
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
        selected_agent="Min-risk",
        game_id=game_id,
        game_config=(width, height, distribution, distribution_kwargs.copy(), seed),
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
        return _load_dqn_agent(
            str(DQN_MODEL_PATH), DQN_MODEL_PATH.stat().st_mtime_ns
        )
    return MinRiskAgent()


def _dqn_available() -> bool:
    return (
        DQN_MODEL_PATH.is_file()
        and importlib.util.find_spec("stable_baselines3") is not None
    )


@st.cache_resource
def _load_dqn_agent(model_path: str, model_mtime_ns: int) -> DQNAgent:
    del model_mtime_ns  # It is part of the cache key and reloads replaced models.
    return DQNAgent(model_path)


def _revealed_cell_html(cell: Any) -> str:
    """Render a revealed cell as a sunken Minesweeper tile."""
    if cell.has_mine:
        return '<div class="ms-cell ms-mine">💣</div>'
    if cell.display_value == 0:
        return '<div class="ms-cell"></div>'
    color_index = min(8, max(1, int(round(cell.display_value))))
    return f'<div class="ms-cell ms-n{color_index}">{cell.display_value:.1f}</div>'


def _render_game(show_probabilities: bool) -> None:
    env = st.session_state.env
    board = env.board
    if board is None:
        return
    st.subheader(st.session_state.message)
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
                column.markdown(_revealed_cell_html(cell), unsafe_allow_html=True)
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

    agent_options = ["Random", "Min-risk"]
    if _dqn_available():
        agent_options.append("DQN")
    st.session_state.selected_agent = st.selectbox(
        "Agent", agent_options, index=1, key="agent_selector"
    )
    if not DQN_MODEL_PATH.is_file():
        st.info(
            "DQN model not found. Train it with: "
            "`uv run python experiments/train_dqn.py --timesteps 50000`"
        )
    elif not _dqn_available():
        st.info("Install the RL dependencies with: `uv sync --dev --extra rl`")
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
    seed: int,
) -> None:
    episodes = st.number_input("Episodes", min_value=1, max_value=10_000, value=100)
    if st.button("Run benchmark", type="primary"):
        agents: list[Any] = [RandomAgent(seed), MinRiskAgent()]
        if _dqn_available():
            agents.append(
                _load_dqn_agent(
                    str(DQN_MODEL_PATH), DQN_MODEL_PATH.stat().st_mtime_ns
                )
            )
        with st.spinner("Evaluating agents..."):
            st.session_state.benchmark_results = compare_agents(
                agents,
                episodes=int(episodes),
                width=width,
                height=height,
                distribution=distribution,
                distribution_kwargs=distribution_kwargs,
                seed=seed,
            )
    results = st.session_state.get("benchmark_results")
    if results:
        rows = [
            {
                "Agent": r.agent_name,
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

The environment is a Markov decision process. State $s_t$ contains revealed-cell
state and the mine-probability channel. Action $a_t$ is a flat index selecting one
hidden cell. At reset, hidden outcomes are sampled independently from each cell's
Bernoulli distribution, making episode transitions stochastic.

The risk-adjusted reward is $1-p_a$ for a safe reveal, $-p_a$ for a mine hit,
$1-p_a+B$ for the winning reveal, and $0$ for a no-op. The learning objective is to
maximize expected discounted return $\mathbb{E}[\sum_t \gamma^t r_t]$.

Random chooses uniformly among valid actions. Min-risk uses the visible probability
field and chooses the valid cell with the lowest $p_a$.

### Deep Q-Network

The DQN model learns $Q(s,a)$: the expected long-term value of selecting cell $a$
in board state $s$. Unlike Min-risk, which only minimizes immediate mine probability,
DQN estimates future reward and can learn policies that trade off immediate risk and
long-term board progress. Training uses flattened board observations.

The Bellman target and squared temporal-difference loss are

$$y = r + \gamma \max_{a'} Q(s',a'),$$

$$L(\theta) = \left(y - Q_\theta(s,a)\right)^2.$$

Stable-Baselines3 DQN does not apply the environment's `action_mask` during training.
At inference time, an invalid prediction is replaced by a valid Min-risk action.
"""
    )


def main() -> None:
    st.set_page_config(page_title="Probabilistic Minesweeper", page_icon="💣", layout="wide")
    st.title("Probabilistic Minesweeper")
    width = st.sidebar.number_input("Width", 2, 12, 5)
    height = st.sidebar.number_input("Height", 2, 12, 5)
    distribution, distribution_kwargs = _distribution_controls()
    seed = st.sidebar.number_input("Seed", value=42)
    show_probabilities = st.sidebar.checkbox("Show hidden probabilities", value=True)
    config = (int(width), int(height), distribution, distribution_kwargs.copy(), int(seed))
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
