"""Agent interface: pick an action given the current backend state."""

from __future__ import annotations

from abc import ABC, abstractmethod

from yugioh_ai.engine import Backend


class Agent(ABC):
    @abstractmethod
    def act(self, backend: Backend) -> int:
        """Return a legal action id for ``backend.current_player()``."""

    def reset(self) -> None:  # optional per-game hook
        pass
