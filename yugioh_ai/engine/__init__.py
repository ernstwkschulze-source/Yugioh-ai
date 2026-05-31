"""Duel engine backends and the engine-agnostic interface they implement."""

from yugioh_ai.engine.backend import (
    Action,
    ActionMeta,
    Backend,
    Observation,
    StepResult,
)


def make_backend(name: str = "mock", **kwargs) -> Backend:
    """Factory for backends by name.

    ``mock``     -> self-contained toy duel (no external deps, used for tests).
    ``ocgcore``  -> real ygopro-core engine (requires a built libocgcore + data;
                    see scripts/fetch_engine.sh).
    """
    name = name.lower()
    if name == "mock":
        from yugioh_ai.engine.mock.backend import MockBackend

        return MockBackend(**kwargs)
    if name in ("ocgcore", "ygopro", "real"):
        from yugioh_ai.engine.ocgcore.backend import OcgcoreBackend

        return OcgcoreBackend(**kwargs)
    raise ValueError(f"Unknown backend {name!r}. Options: 'mock', 'ocgcore'.")


__all__ = [
    "Action",
    "ActionMeta",
    "Backend",
    "Observation",
    "StepResult",
    "make_backend",
]
