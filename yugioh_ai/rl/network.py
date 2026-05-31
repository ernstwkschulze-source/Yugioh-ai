"""Policy/value network and the evaluator that adapts it for MCTS.

PyTorch is an *optional* dependency. If it is not installed, importing the net
classes raises a clear error, but the rest of the project (mock backend, env,
classic MCTS via rollouts, self-play) keeps working. Install with::

    pip install -r requirements-rl.txt
"""

from __future__ import annotations

import numpy as np

from yugioh_ai.engine import Backend

try:  # optional
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def has_torch() -> bool:
    return _HAS_TORCH


if _HAS_TORCH:

    class PolicyValueNet(nn.Module):
        """A small MLP trunk with policy and value heads.

        Input is the flattened observation planes; the action space is fixed by
        the backend. For the real ocgcore encoder this trunk should be replaced
        by a residual conv/attention stack over the structured field encoding
        (see docs/ARCHITECTURE.md), but the head interface stays the same.
        """

        def __init__(self, obs_size: int, action_size: int, hidden: int = 256, blocks: int = 3):
            super().__init__()
            self.input = nn.Linear(obs_size, hidden)
            self.trunk = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(blocks)])
            self.policy_head = nn.Linear(hidden, action_size)
            self.value_head = nn.Linear(hidden, 1)

        def forward(self, x):
            h = F.relu(self.input(x))
            for layer in self.trunk:
                h = F.relu(layer(h) + h)  # residual
            policy_logits = self.policy_head(h)
            value = torch.tanh(self.value_head(h)).squeeze(-1)
            return policy_logits, value

    class NeuralEvaluator:
        """Adapts a :class:`PolicyValueNet` to the MCTS ``Evaluator`` protocol."""

        def __init__(self, net: "PolicyValueNet", device: str = "cpu"):
            self.net = net.to(device).eval()
            self.device = device

        @torch.no_grad()
        def evaluate(self, backend: Backend) -> tuple[np.ndarray, float]:
            player = backend.current_player()
            obs = backend.observation(player)
            x = torch.as_tensor(obs.planes, dtype=torch.float32, device=self.device).flatten()[None]
            logits, value = self.net(x)
            logits = logits[0].cpu().numpy()
            mask = obs.legal_mask
            masked = np.where(mask, logits, -1e9)
            masked = masked - masked.max()
            exp = np.exp(masked)
            probs = exp / exp.sum() if exp.sum() > 0 else mask / max(mask.sum(), 1)
            return probs.astype(np.float32), float(value.item())

else:  # pragma: no cover - torch missing

    class PolicyValueNet:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError(
                "PyTorch is required for the neural network. "
                "Install it with: pip install -r requirements-rl.txt"
            )

    class NeuralEvaluator:  # type: ignore
        def __init__(self, *a, **k):
            raise ImportError(
                "PyTorch is required for NeuralEvaluator. "
                "Install it with: pip install -r requirements-rl.txt"
            )
