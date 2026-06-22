def test_app_imports():
    from app import main

    assert callable(main)


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


def test_maskable_ppo_model_dropdown_discovers_only_maskable_zip_files(
    monkeypatch, tmp_path
):
    import app

    (tmp_path / "maskable_ppo_model.zip").touch()
    (tmp_path / "ppo_model.zip").touch()
    (tmp_path / "dqn_model.zip").touch()
    (tmp_path / "notes.txt").touch()
    monkeypatch.setattr(app, "DQN_MODELS_DIR", tmp_path)

    assert [path.name for path in app._maskable_ppo_model_paths()] == [
        "maskable_ppo_model.zip",
    ]


def test_create_env_uses_hidden_risk_configuration():
    import app

    env = app.create_env(
        3,
        3,
        "constant",
        {"p": 0.0},
        "actual_count",
        "safe_2x2",
        "completion",
    )
    try:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (3, 3, 3)
        assert env.obs_mode == "state"
        assert env.clue_mode == "actual_count"
        assert env.initial_reveal == "safe_2x2"
        assert env.board.revealed_mask().sum() == 4
        assert env.reward_config.win_bonus == 10.0
    finally:
        env.close()


def test_dqn_compatibility_requires_three_channel_shape(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import app

    model_path = tmp_path / "model.zip"
    model_path.touch()
    monkeypatch.setattr(app, "_dqn_available", lambda: True)
    monkeypatch.setattr(app, "_selected_dqn_model_path", lambda: model_path)

    fake_agent = SimpleNamespace(
        model=SimpleNamespace(observation_space=SimpleNamespace(shape=(5 * 5 * 3,)))
    )
    monkeypatch.setattr(app, "_load_dqn_agent", lambda *_args: fake_agent)
    assert app._dqn_compatible(5, 5)

    fake_agent.model.observation_space.shape = (5 * 5 * 4,)
    assert not app._dqn_compatible(5, 5)


def test_maskable_ppo_compatibility_requires_three_channel_shape(
    monkeypatch, tmp_path
):
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
