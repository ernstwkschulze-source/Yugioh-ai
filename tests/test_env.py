import numpy as np

from yugioh_ai.env import YGOEnv


def test_env_reset_returns_obs():
    env = YGOEnv("mock")
    obs, info = env.reset(seed=0)
    assert obs.planes.ndim == 1
    assert "to_move" in info


def test_env_step_until_terminal():
    env = YGOEnv("mock")
    env.reset(seed=1)
    rng = np.random.default_rng(0)
    terminated = False
    steps = 0
    reward = 0.0
    while not terminated and steps < 5000:
        legal = env.legal_actions()
        obs, reward, terminated, truncated, info = env.step(int(rng.choice(legal)))
        steps += 1
    assert terminated
    assert reward in (-1.0, 0.0, 1.0)


def test_env_legal_mask_matches_actions():
    env = YGOEnv("mock")
    env.reset(seed=2)
    mask = env.legal_mask()
    legal = set(env.legal_actions())
    assert set(np.flatnonzero(mask).tolist()) == legal
