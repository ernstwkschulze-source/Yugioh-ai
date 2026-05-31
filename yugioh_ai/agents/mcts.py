"""AlphaZero-style PUCT Monte Carlo Tree Search over a :class:`Backend`.

This is the search core of the "Stockfish of Yu-Gi-Oh" plan. It is engine- and
evaluator-agnostic:

* It searches on :meth:`Backend.clone` copies, so it works for any backend whose
  ``clone`` is faithful (the mock backend; ocgcore once determinization lands).
* It takes an :class:`Evaluator` mapping a state to ``(priors, value)``. With the
  neural evaluator this is full AlphaZero; with :class:`RandomRolloutEvaluator`
  it is classic MCTS that plays reasonably with no trained network -- which is
  what lets the whole self-play loop run before any training.

Two-player handling: turns are NOT assumed to strictly alternate (a player may
take several actions in one turn). Every leaf is evaluated from the perspective
of the player to move at that leaf, and during backup each edge accumulates
value from the perspective of the player who chose it. This is correct for
multi-action turns and is the key subtlety vs. a naive negamax MCTS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from yugioh_ai.agents.base import Agent
from yugioh_ai.engine import Backend


class Evaluator(Protocol):
    def evaluate(self, backend: Backend) -> tuple[np.ndarray, float]:
        """Return (priors over full action space, value for current player)."""
        ...


class RandomRolloutEvaluator:
    """Uniform priors + a random-playout value estimate. No network required."""

    def __init__(self, seed: int = 0, max_rollout: int = 256):
        self.rng = np.random.default_rng(seed)
        self.max_rollout = max_rollout

    def evaluate(self, backend: Backend) -> tuple[np.ndarray, float]:
        n = backend.action_space_size
        priors = np.zeros(n, dtype=np.float32)
        legal = backend.legal_actions()
        if legal:
            priors[legal] = 1.0 / len(legal)
        value = self._rollout(backend)
        return priors, value

    def _rollout(self, backend: Backend) -> float:
        persp = backend.current_player()
        sim = backend.clone()
        for _ in range(self.max_rollout):
            if sim.is_terminal():
                break
            legal = sim.legal_actions()
            if not legal:
                break
            sim.step(int(self.rng.choice(legal)))
        if not sim.is_terminal():
            return 0.0
        w = sim.winner()
        if w is None:
            return 0.0
        return 1.0 if w == persp else -1.0


@dataclass
class _Edge:
    prior: float
    n: int = 0
    w: float = 0.0
    child: "_Node | None" = None

    @property
    def q(self) -> float:
        return self.w / self.n if self.n else 0.0


@dataclass
class _Node:
    backend: Backend
    to_move: int
    terminal: bool
    edges: dict[int, _Edge] = field(default_factory=dict)
    expanded: bool = False


class MCTS:
    def __init__(
        self,
        evaluator: Evaluator | None = None,
        *,
        n_simulations: int = 128,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_eps: float = 0.25,
        seed: int = 0,
    ):
        self.evaluator = evaluator or RandomRolloutEvaluator(seed=seed)
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps
        self.rng = np.random.default_rng(seed)

    # -- public ----------------------------------------------------------
    def run(self, backend: Backend, *, add_noise: bool = True) -> np.ndarray:
        """Return a visit-count vector over the full action space at the root."""
        root = _Node(backend.clone(), backend.current_player(), backend.is_terminal())
        self._expand(root)
        if add_noise:
            self._add_root_noise(root)
        for _ in range(self.n_simulations):
            self._simulate(root)
        counts = np.zeros(backend.action_space_size, dtype=np.float32)
        for a, e in root.edges.items():
            counts[a] = e.n
        return counts

    def policy(self, backend: Backend, *, temperature: float = 1.0, add_noise: bool = False) -> np.ndarray:
        counts = self.run(backend, add_noise=add_noise)
        if counts.sum() == 0:
            mask = backend.legal_mask().astype(np.float32)
            return mask / mask.sum()
        if temperature <= 1e-6:
            out = np.zeros_like(counts)
            out[int(counts.argmax())] = 1.0
            return out
        logits = counts ** (1.0 / temperature)
        return logits / logits.sum()

    # -- internals -------------------------------------------------------
    def _expand(self, node: _Node) -> float:
        if node.terminal:
            node.expanded = True
            return self._terminal_value(node)
        priors, value = self.evaluator.evaluate(node.backend)
        legal = node.backend.legal_actions()
        total = float(sum(priors[a] for a in legal))
        for a in legal:
            p = priors[a] / total if total > 0 else 1.0 / len(legal)
            node.edges[a] = _Edge(prior=p)
        node.expanded = True
        return value

    def _terminal_value(self, node: _Node) -> float:
        w = node.backend.winner()
        if w is None:
            return 0.0
        return 1.0 if w == node.to_move else -1.0

    def _add_root_noise(self, root: _Node) -> None:
        if not root.edges:
            return
        actions = list(root.edges)
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
        for a, eta in zip(actions, noise):
            e = root.edges[a]
            e.prior = (1 - self.dirichlet_eps) * e.prior + self.dirichlet_eps * eta

    def _select(self, node: _Node) -> int:
        total_n = sum(e.n for e in node.edges.values())
        sqrt_total = math.sqrt(total_n) if total_n > 0 else 1.0
        best_a, best_score = -1, -float("inf")
        for a, e in node.edges.items():
            u = self.c_puct * e.prior * sqrt_total / (1 + e.n)
            score = e.q + u
            if score > best_score:
                best_score, best_a = score, a
        return best_a

    def _simulate(self, root: _Node) -> None:
        path: list[tuple[_Node, int]] = []
        node = root
        while True:
            if node.terminal:
                value, persp = self._terminal_value(node), node.to_move
                break
            if not node.expanded:
                value, persp = self._expand(node), node.to_move
                break
            a = self._select(node)
            edge = node.edges[a]
            path.append((node, a))
            if edge.child is None:
                nb = node.backend.clone()
                nb.step(a)
                edge.child = _Node(nb, nb.current_player(), nb.is_terminal())
                node = edge.child
                value = self._expand(node)
                persp = node.to_move
                break
            node = edge.child
        for n, a in path:
            e = n.edges[a]
            e.n += 1
            e.w += value if n.to_move == persp else -value


class MCTSAgent(Agent):
    def __init__(self, mcts: MCTS | None = None, *, temperature: float = 0.0, **mcts_kwargs):
        self.mcts = mcts or MCTS(**mcts_kwargs)
        self.temperature = temperature

    def act(self, backend: Backend) -> int:
        pi = self.mcts.policy(backend, temperature=self.temperature, add_noise=False)
        return int(pi.argmax())
