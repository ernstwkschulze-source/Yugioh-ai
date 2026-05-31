"""Yugioh-ai: an AlphaZero-style self-play agent for Yu-Gi-Oh.

Goal: a self-improving engine that simulates games, explores decision trees via
search, and learns from self-play -- the "Stockfish of Yu-Gi-Oh".

Top-level layout::

    yugioh_ai.engine   duel backends (mock + real ocgcore) behind one interface
    yugioh_ai.env      Gymnasium-style environment wrapper
    yugioh_ai.agents   Random / MCTS agents (PUCT search)
    yugioh_ai.rl       self-play, replay buffer, AlphaZero training
"""

__version__ = "0.1.0"

from yugioh_ai.engine import make_backend
from yugioh_ai.env import YGOEnv

__all__ = ["YGOEnv", "make_backend", "__version__"]
