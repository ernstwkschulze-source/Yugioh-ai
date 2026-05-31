"""Replay buffer of self-play training samples.

Each sample is ``(planes, policy_target, value_target)`` where the value target
is the game outcome from the perspective of the player who was to move in that
position (z in [-1, 1]), and the policy target is the MCTS visit distribution.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    planes: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000, seed: int = 0):
        self.buffer: deque[Sample] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def add(self, sample: Sample) -> None:
        self.buffer.append(sample)

    def extend(self, samples) -> None:
        for s in samples:
            self.add(s)

    def __len__(self) -> int:
        return len(self.buffer)

    def sample(self, batch_size: int):
        n = min(batch_size, len(self.buffer))
        batch = self.rng.sample(list(self.buffer), n)
        planes = np.stack([s.planes.flatten() for s in batch]).astype(np.float32)
        policies = np.stack([s.policy for s in batch]).astype(np.float32)
        values = np.array([s.value for s in batch], dtype=np.float32)
        return planes, policies, values

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            planes=np.stack([s.planes.flatten() for s in self.buffer]),
            policies=np.stack([s.policy for s in self.buffer]),
            values=np.array([s.value for s in self.buffer], dtype=np.float32),
        )

    def load(self, path: str) -> None:
        data = np.load(path)
        for p, pi, v in zip(data["planes"], data["policies"], data["values"]):
            self.add(Sample(p, pi, float(v)))
