import numpy as np

from yugioh_ai.engine import make_backend
from yugioh_ai.engine.mock.backend import A_ATTACK0, A_END_TURN, A_SUMMON0, MockBackend


def test_reset_and_legal_actions():
    be = make_backend("mock")
    be.reset(seed=1)
    assert not be.is_terminal()
    legal = be.legal_actions()
    assert A_END_TURN in legal
    assert all(0 <= a < be.action_space_size for a in legal)


def test_determinism_same_seed():
    be1 = MockBackend(); be1.reset(seed=42)
    be2 = MockBackend(); be2.reset(seed=42)
    assert be1.hand == be2.hand
    assert be1.deck == be2.deck


def test_clone_is_independent():
    be = MockBackend(); be.reset(seed=3)
    clone = be.clone()
    clone.step(clone.legal_actions()[0])
    assert be.lp == [20, 20]
    assert clone is not be


def test_first_player_cannot_attack_turn0():
    be = MockBackend(); be.reset(seed=7)
    be.step(A_SUMMON0)  # summon a monster
    legal = be.legal_actions()
    assert not any(A_ATTACK0 <= a < A_ATTACK0 + 3 for a in legal)


def test_summon_once_per_turn():
    be = MockBackend(); be.reset(seed=7)
    be.step(A_SUMMON0)
    legal = be.legal_actions()
    assert not any(A_SUMMON0 <= a < A_SUMMON0 + 5 for a in legal)


def test_direct_attack_deals_damage():
    be = MockBackend(); be.reset(seed=7)
    be.field = [[4], []]
    be.lp = [20, 20]
    be.active = 0
    be.turn = 2  # not first turn
    be._attacked = [set(), set()]
    be.summoned_this_turn = True
    be.step(A_ATTACK0)
    assert be.lp[1] == 16


def test_battle_destroys_weaker_and_deals_difference():
    be = MockBackend(); be.reset(seed=7)
    be.field = [[4], [1]]
    be.lp = [20, 20]
    be.active = 0
    be.turn = 2
    be._attacked = [set(), set()]
    be.summoned_this_turn = True
    be.step(A_ATTACK0)
    assert be.field[1] == []      # weaker defender destroyed
    assert be.lp[1] == 17         # took the 3 ATK difference


def test_game_terminates_and_has_winner():
    be = MockBackend(); be.reset(seed=5)
    rng = np.random.default_rng(0)
    steps = 0
    while not be.is_terminal() and steps < 5000:
        legal = be.legal_actions()
        be.step(int(rng.choice(legal)))
        steps += 1
    assert be.is_terminal()
    assert be.winner() in (0, 1, None)


def test_observation_hides_opponent_hand():
    be = MockBackend(); be.reset(seed=9)
    obs = be.observation(0)
    assert "hand" in obs.info
    assert isinstance(obs.info["opp_hand_size"], int)
    assert obs.legal_mask.shape == (be.action_space_size,)
