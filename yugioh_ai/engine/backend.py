"""Engine-agnostic interface for a Yu-Gi-Oh duel backend.

Everything above this layer (env, agents, MCTS, training) is written against
``Backend`` and never touches a concrete engine. This is what lets us swap the
self-contained :class:`~yugioh_ai.engine.mock.backend.MockBackend` (used for
development, tests and CI) for the real :class:`~yugioh_ai.engine.ocgcore.backend.OcgcoreBackend`
(ygopro-core / ocgcore) without changing a line of agent or training code.

Design notes
------------
* A "decision point" is a moment where one player must choose among several
  legal actions. ``step`` advances the duel through any deterministic, forced,
  or non-decision processing until the next decision point (or terminal).
* Yu-Gi-Oh is an imperfect-information game (hidden hands, set cards, deck
  order). ``observation(player)`` returns *that player's* view. ``clone`` is
  used by search; see :mod:`yugioh_ai.agents.mcts` for how hidden information is
  handled (determinization is on the roadmap; v0.1 searches a full clone).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


# An action is identified by a stable integer id within a global action space,
# plus optional engine-specific payload. Keeping a flat integer id is what the
# policy network and MCTS index into.
Action = int


@dataclass(frozen=True)
class ActionMeta:
    """Human-readable description of an action id, for logging/debugging."""

    action_id: Action
    label: str
    payload: Any = None


@dataclass
class Observation:
    """A single player's view of the duel state.

    ``planes`` is the model-facing tensor encoding (filled in by the encoder for
    real engines; the mock backend produces a compact hand-rolled encoding).
    ``legal_mask`` is a boolean vector over the global action space marking which
    actions are currently legal for the player to move.
    """

    planes: np.ndarray
    legal_mask: np.ndarray
    to_move: int
    # Optional rich, human-readable snapshot (life points, zones, etc.).
    info: dict = field(default_factory=dict)


@dataclass
class StepResult:
    terminal: bool
    winner: int | None  # 0 or 1, or None on draw / not terminal
    reward: float = 0.0  # from the perspective of the player who just moved
    info: dict = field(default_factory=dict)


class Backend(ABC):
    """Abstract duel engine.

    Subclasses must be deterministic given the seed passed to :meth:`reset`, so
    that :meth:`clone` produces a faithful copy for search rollouts.
    """

    #: Size of the global, fixed action space the policy head indexes into.
    action_space_size: int

    @abstractmethod
    def reset(self, *, seed: int, decks: Sequence[Sequence[int]] | None = None) -> None:
        """Start a new duel. ``decks`` is a per-player list of card ids."""

    @abstractmethod
    def current_player(self) -> int:
        """Index (0/1) of the player who must act at the current decision point."""

    @abstractmethod
    def legal_actions(self) -> list[Action]:
        """Legal action ids for :meth:`current_player` right now."""

    @abstractmethod
    def step(self, action: Action) -> StepResult:
        """Apply ``action`` and advance to the next decision point or terminal."""

    @abstractmethod
    def observation(self, player: int) -> Observation:
        """Return ``player``'s (imperfect-information) view."""

    @abstractmethod
    def is_terminal(self) -> bool: ...

    @abstractmethod
    def winner(self) -> int | None:
        """0/1 winner, or None for a draw / ongoing duel."""

    @abstractmethod
    def clone(self) -> "Backend":
        """Deep copy including RNG state, for MCTS simulations."""

    # -- convenience -----------------------------------------------------
    def legal_mask(self) -> np.ndarray:
        mask = np.zeros(self.action_space_size, dtype=bool)
        mask[self.legal_actions()] = True
        return mask

    def describe_action(self, action: Action) -> ActionMeta:  # pragma: no cover
        return ActionMeta(action, f"action#{action}")
