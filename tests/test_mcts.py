import numpy as np

from yugioh_ai.agents import MCTS, MCTSAgent, RandomAgent, RandomRolloutEvaluator
from yugioh_ai.engine import make_backend
from yugioh_ai.rl.selfplay import play_match, play_self_play_game


def test_mcts_returns_legal_policy():
    be = make_backend("mock"); be.reset(seed=0)
    mcts = MCTS(n_simulations=32, seed=0)
    pi = mcts.policy(be, temperature=1.0)
    assert pi.shape == (be.action_space_size,)
    assert abs(pi.sum() - 1.0) < 1e-5
    illegal = ~be.legal_mask()
    assert pi[illegal].sum() < 1e-6


def test_mcts_agent_plays_legal_moves():
    be = make_backend("mock"); be.reset(seed=1)
    agent = MCTSAgent(MCTS(n_simulations=16, seed=1))
    a = agent.act(be)
    assert a in be.legal_actions()


def test_self_play_generates_samples():
    mcts = MCTS(n_simulations=16, seed=0)
    samples, winner = play_self_play_game(mcts, backend="mock", seed=0, temp_moves=4)
    assert len(samples) > 0
    s = samples[0]
    assert s.value in (-1.0, 0.0, 1.0)
    assert abs(s.policy.sum() - 1.0) < 1e-5


def test_mcts_beats_random():
    # With rollout-based search, MCTS should not lose to a random agent overall.
    a = MCTSAgent(MCTS(RandomRolloutEvaluator(seed=0), n_simulations=64, seed=0))
    b = RandomAgent(seed=0)
    res = play_match(a, b, backend="mock", games=20, seed=0)
    assert res["a"] > res["b"]
