"""Self-play game generation and agent-vs-agent evaluation."""

from __future__ import annotations

import numpy as np

from yugioh_ai.agents.base import Agent
from yugioh_ai.agents.mcts import MCTS
from yugioh_ai.engine import Backend, make_backend
from yugioh_ai.rl.replay import Sample


def play_self_play_game(
    mcts: MCTS,
    *,
    backend: Backend | str = "mock",
    seed: int = 0,
    temperature: float = 1.0,
    temp_moves: int = 10,
    max_moves: int = 400,
    backend_kwargs: dict | None = None,
) -> tuple[list[Sample], int | None]:
    """Play one game with MCTS on both sides; return (samples, winner).

    Moves before ``temp_moves`` are sampled from the visit distribution at
    ``temperature`` (exploration); afterwards play is greedy. Value targets are
    filled in from the final result relative to the player to move.
    """
    be: Backend = (
        make_backend(backend, **(backend_kwargs or {})) if isinstance(backend, str) else backend
    )
    be.reset(seed=seed)
    rng = np.random.default_rng(seed)

    records: list[tuple[np.ndarray, np.ndarray, int]] = []
    move = 0
    while not be.is_terminal() and move < max_moves:
        player = be.current_player()
        temp = temperature if move < temp_moves else 1e-6
        pi = mcts.policy(be, temperature=temp, add_noise=True)
        obs = be.observation(player)
        records.append((obs.planes.copy(), pi.copy(), player))
        action = int(rng.choice(len(pi), p=pi)) if temp > 1e-6 else int(pi.argmax())
        be.step(action)
        move += 1

    winner = be.winner()
    samples = []
    for planes, pi, player in records:
        z = 0.0 if winner is None else (1.0 if winner == player else -1.0)
        samples.append(Sample(planes=planes, policy=pi, value=z))
    return samples, winner


def play_match(
    agent_a: Agent,
    agent_b: Agent,
    *,
    backend: str = "mock",
    games: int = 20,
    seed: int = 0,
    backend_kwargs: dict | None = None,
) -> dict:
    """Play ``games`` games, alternating who goes first. Returns win stats."""
    results = {"a": 0, "b": 0, "draw": 0}
    for g in range(games):
        be = make_backend(backend, **(backend_kwargs or {}))
        be.reset(seed=seed + g)
        agent_a.reset()
        agent_b.reset()
        a_is_player0 = g % 2 == 0
        agents = (agent_a, agent_b) if a_is_player0 else (agent_b, agent_a)
        while not be.is_terminal():
            cur = agents[be.current_player()]
            be.step(cur.act(be))
        w = be.winner()
        if w is None:
            results["draw"] += 1
        else:
            winner_agent = agents[w]
            if winner_agent is agent_a:
                results["a"] += 1
            else:
                results["b"] += 1
    results["a_winrate"] = results["a"] / games
    return results
