"""A thin, Gymnasium-style wrapper around any :class:`Backend`.

We don't hard-depend on gymnasium (so the project runs with numpy alone), but
the interface mirrors it: ``reset() -> (obs, info)`` and
``step(action) -> (obs, reward, terminated, truncated, info)``. If gymnasium is
installed you also get :class:`YGOGymEnv`, a registered ``gymnasium.Env``.

This is a two-player, turn-based, zero-sum env. ``step`` returns the reward from
the perspective of the player who just moved; the alternation of players is
exposed via ``obs.to_move`` / ``info['to_move']`` so training code (self-play)
can assign value targets correctly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from yugioh_ai.engine import Backend, Observation, make_backend


class YGOEnv:
    def __init__(self, backend: Backend | str = "mock", **backend_kwargs):
        self.backend: Backend = (
            make_backend(backend, **backend_kwargs) if isinstance(backend, str) else backend
        )
        self.action_space_size = self.backend.action_space_size

    def reset(self, *, seed: int = 0, decks: Sequence[Sequence[int]] | None = None):
        self.backend.reset(seed=seed, decks=decks)
        obs = self.backend.observation(self.backend.current_player())
        return obs, {"to_move": self.backend.current_player()}

    def step(self, action: int):
        mover = self.backend.current_player()
        result = self.backend.step(action)
        terminated = result.terminal
        to_move = None if terminated else self.backend.current_player()
        obs = self.backend.observation(to_move if to_move is not None else mover)
        info = {
            "to_move": to_move,
            "mover": mover,
            "winner": result.winner,
            **result.info,
        }
        return obs, result.reward, terminated, False, info

    def legal_actions(self) -> list[int]:
        return self.backend.legal_actions()

    def legal_mask(self) -> np.ndarray:
        return self.backend.legal_mask()

    def current_player(self) -> int:
        return self.backend.current_player()

    def render(self) -> str:
        obs = self.backend.observation(self.backend.current_player())
        return f"to_move={obs.to_move} info={obs.info}"


def _maybe_register_gym():  # pragma: no cover - optional dependency
    try:
        import gymnasium as gym
        from gymnasium import spaces
    except Exception:
        return None

    class YGOGymEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, backend: str = "mock", **kw):
            self._env = YGOEnv(backend, **kw)
            n = self._env.action_space_size
            self.action_space = spaces.Discrete(n)
            probe, _ = self._env.reset(seed=0)
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=probe.planes.shape, dtype=np.float32
            )

        def reset(self, *, seed=None, options=None):
            obs, info = self._env.reset(seed=seed or 0)
            return obs.planes, info

        def step(self, action):
            obs, reward, term, trunc, info = self._env.step(int(action))
            return obs.planes, reward, term, trunc, info

    return YGOGymEnv


YGOGymEnv = _maybe_register_gym()

__all__ = ["YGOEnv", "YGOGymEnv", "Observation"]
