"""AlphaZero training loop: alternate self-play and network optimization.

Each iteration:
  1. Generate self-play games with the current network (MCTS on both sides).
  2. Add the resulting (state, policy, value) samples to the replay buffer.
  3. Optimize the network: policy = cross-entropy to MCTS visits, value = MSE to
     game outcome.

Requires PyTorch (``pip install -r requirements-rl.txt``). Without it, use the
classic-MCTS self-play loop (``yugioh_ai.rl.selfplay``) which needs only numpy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from yugioh_ai.agents.mcts import MCTS
from yugioh_ai.engine import make_backend
from yugioh_ai.rl.network import has_torch
from yugioh_ai.rl.replay import ReplayBuffer
from yugioh_ai.rl.selfplay import play_self_play_game


@dataclass
class TrainConfig:
    backend: str = "mock"
    iterations: int = 5
    games_per_iter: int = 20
    simulations: int = 64
    epochs_per_iter: int = 5
    batch_size: int = 256
    lr: float = 1e-3
    buffer_capacity: int = 50_000
    checkpoint_dir: str = "checkpoints"
    hidden: int = 256
    seed: int = 0


def _obs_size(backend: str) -> int:
    be = make_backend(backend)
    be.reset(seed=0)
    return int(be.observation(be.current_player()).planes.flatten().shape[0])


def train(config: TrainConfig):
    if not has_torch():
        raise ImportError(
            "Training the neural network requires PyTorch. Install with:\n"
            "  pip install -r requirements-rl.txt\n"
            "Meanwhile you can run classic-MCTS self-play with `yugioh-ai selfplay`."
        )
    import torch
    import torch.nn.functional as F

    from yugioh_ai.rl.network import NeuralEvaluator, PolicyValueNet

    rng = np.random.default_rng(config.seed)
    action_size = make_backend(config.backend).action_space_size
    obs_size = _obs_size(config.backend)

    net = PolicyValueNet(obs_size, action_size, hidden=config.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=config.lr)
    buffer = ReplayBuffer(capacity=config.buffer_capacity, seed=config.seed)
    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for it in range(config.iterations):
        # 1-2. self-play
        evaluator = NeuralEvaluator(net)
        n_games = 0
        for _ in range(config.games_per_iter):
            mcts = MCTS(evaluator, n_simulations=config.simulations, seed=int(rng.integers(1 << 30)))
            samples, _ = play_self_play_game(
                mcts, backend=config.backend, seed=int(rng.integers(1 << 30))
            )
            buffer.extend(samples)
            n_games += 1

        # 3. optimize
        net.train()
        last_loss = float("nan")
        for _ in range(config.epochs_per_iter):
            if len(buffer) < config.batch_size:
                break
            planes, target_pi, target_v = buffer.sample(config.batch_size)
            x = torch.as_tensor(planes, dtype=torch.float32)
            tpi = torch.as_tensor(target_pi, dtype=torch.float32)
            tv = torch.as_tensor(target_v, dtype=torch.float32)
            logits, value = net(x)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(tpi * logp).sum(dim=-1).mean()
            value_loss = F.mse_loss(value, tv)
            loss = policy_loss + value_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = float(loss.item())
        net.eval()

        ckpt = os.path.join(config.checkpoint_dir, f"net_iter{it:03d}.pt")
        torch.save(net.state_dict(), ckpt)
        print(
            f"[iter {it:02d}] games={n_games} buffer={len(buffer)} "
            f"loss={last_loss:.4f} -> {ckpt}"
        )

    return net
