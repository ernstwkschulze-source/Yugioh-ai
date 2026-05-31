"""Command-line entrypoint: demo, selfplay, eval, train.

    yugioh-ai demo                 # play one mock game, MCTS vs random
    yugioh-ai selfplay --games 5   # generate self-play games (numpy only)
    yugioh-ai eval --games 20      # MCTS vs random win rate
    yugioh-ai train --iterations 5 # AlphaZero training (needs torch)
"""

from __future__ import annotations

import argparse
import sys


def _cmd_demo(args):
    from yugioh_ai.agents import MCTS, MCTSAgent, RandomAgent
    from yugioh_ai.engine import make_backend

    be = make_backend(args.backend)
    be.reset(seed=args.seed)
    agents = [
        MCTSAgent(MCTS(n_simulations=args.simulations, seed=args.seed)),
        RandomAgent(seed=args.seed),
    ]
    print(f"Player 0: MCTS({args.simulations} sims)   Player 1: Random")
    turn = 0
    while not be.is_terminal():
        p = be.current_player()
        a = agents[p].act(be)
        meta = be.describe_action(a)
        be.step(a)
        turn += 1
        if turn <= 40:
            print(f"  ply {turn:>3} | P{p} -> {meta.label}")
    w = be.winner()
    print(f"Result: {'draw' if w is None else f'player {w} wins'} after {turn} plies")


def _cmd_selfplay(args):
    from yugioh_ai.agents import MCTS
    from yugioh_ai.rl.selfplay import play_self_play_game

    total = 0
    wins = {0: 0, 1: 0, None: 0}
    for g in range(args.games):
        mcts = MCTS(n_simulations=args.simulations, seed=args.seed + g)
        samples, winner = play_self_play_game(mcts, backend=args.backend, seed=args.seed + g)
        total += len(samples)
        wins[winner] += 1
        print(f"  game {g+1}/{args.games}: {len(samples)} samples, winner={winner}")
    print(f"Generated {total} samples. Wins p0={wins[0]} p1={wins[1]} draws={wins[None]}")


def _cmd_eval(args):
    from yugioh_ai.agents import MCTS, MCTSAgent, RandomAgent
    from yugioh_ai.rl.selfplay import play_match

    a = MCTSAgent(MCTS(n_simulations=args.simulations, seed=args.seed))
    b = RandomAgent(seed=args.seed)
    res = play_match(a, b, backend=args.backend, games=args.games, seed=args.seed)
    print(f"MCTS vs Random over {args.games} games: {res}")


def _cmd_train(args):
    from yugioh_ai.rl.train import TrainConfig, train

    cfg = TrainConfig(
        backend=args.backend,
        iterations=args.iterations,
        games_per_iter=args.games,
        simulations=args.simulations,
        seed=args.seed,
    )
    try:
        train(cfg)
    except ImportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="yugioh-ai", description="Stockfish-of-Yu-Gi-Oh toolkit")
    p.add_argument("--backend", default="mock", help="duel backend: mock | ocgcore")
    p.add_argument("--simulations", type=int, default=128, help="MCTS simulations per move")
    p.add_argument("--seed", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("demo", help="play one game, MCTS vs random")
    sp.set_defaults(func=_cmd_demo)

    sp = sub.add_parser("selfplay", help="generate self-play games (numpy only)")
    sp.add_argument("--games", type=int, default=5)
    sp.set_defaults(func=_cmd_selfplay)

    sp = sub.add_parser("eval", help="MCTS vs random win rate")
    sp.add_argument("--games", type=int, default=20)
    sp.set_defaults(func=_cmd_eval)

    sp = sub.add_parser("train", help="AlphaZero training (needs torch)")
    sp.add_argument("--games", type=int, default=20)
    sp.add_argument("--iterations", type=int, default=5)
    sp.set_defaults(func=_cmd_train)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
