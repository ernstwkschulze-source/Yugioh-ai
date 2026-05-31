"""A uniform-random legal-move agent. Baseline opponent / sanity check."""

from __future__ import annotations

import numpy as np

from yugioh_ai.agents.base import Agent
from yugioh_ai.engine import Backend


class RandomAgent(Agent):
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, backend: Backend) -> int:
        legal = backend.legal_actions()
        if not legal:
            raise RuntimeError("RandomAgent.act called with no legal actions")
        return int(self.rng.choice(legal))
