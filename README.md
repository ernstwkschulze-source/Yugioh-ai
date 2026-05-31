# Yugioh-ai — toward the *Stockfish of Yu-Gi-Oh*

A self-play reinforcement-learning agent for Yu-Gi-Oh. The aim: simulate games,
play autonomously, explore decision trees with search, learn from self-play, and
keep improving — an engine that can probe and ultimately *solve* lines in the
current TCG meta, the way Stockfish/AlphaZero do for chess.

This repository is the **v0.1 foundation**: a clean, engine-agnostic
architecture with a working end-to-end loop (search + self-play + training)
running today on a built-in toy duel, plus the real-engine integration layer for
[ygopro-core / ocgcore](https://github.com/edo9300/ygopro-core).

> **Honest status.** Yu-Gi-Oh has thousands of cards with Lua-scripted effects;
> "solving the meta" is a long research program, not a weekend script. What's
> here is the *correct scaffolding* for that program — every component above the
> engine works now, and the real-engine backend is wired up to the point where
> the remaining work is decoding the engine's decision messages (clearly marked).

---

## What works today

```bash
pip install -e .                  # numpy only
yugioh-ai demo                    # MCTS vs random on the toy duel, move by move
yugioh-ai eval --games 40         # MCTS beats random (sanity that search works)
yugioh-ai selfplay --games 10     # generate self-play training data (numpy only)
pytest                            # full test suite
```

Add the neural network (AlphaZero) training:

```bash
pip install -r requirements-rl.txt   # adds torch
yugioh-ai train --iterations 5 --games 20
```

## Architecture

```
yugioh_ai/
  engine/                 duel engines behind ONE interface (Backend)
    backend.py            the interface everything else is written against
    mock/backend.py       self-contained toy duel (no deps) — the dev/CI testbed
    ocgcore/              REAL ygopro-core integration (ffi + lifecycle + msg loop)
  env.py                  Gymnasium-style env wrapper
  agents/
    mcts.py               AlphaZero-style PUCT search (the core of the engine)
    random_agent.py       baseline opponent
  rl/
    network.py            policy/value net (PyTorch, optional) + MCTS evaluator
    selfplay.py           self-play game generation + agent-vs-agent eval
    replay.py             replay buffer
    train.py              AlphaZero training loop
  cli.py                  demo / selfplay / eval / train
scripts/fetch_engine.sh   clone + build ocgcore, card DB and Lua scripts
docs/ARCHITECTURE.md      design + how the real engine plugs in
docs/ROADMAP.md           the path from toy to meta-solver
```

The key design choice: **the agent, search, and training code never touch a
concrete engine.** They talk to `Backend`. The `mock` backend lets the whole RL
stack run and be tested instantly; the `ocgcore` backend swaps in the real rules
without changing anything above it.

## The real engine (ocgcore)

```bash
bash scripts/fetch_engine.sh      # needs GitHub access + a C++17 toolchain
```

This clones and builds `libocgcore`, assembles the Lua card scripts, and merges
the card database into `cards.cdb`. The Python integration
(`yugioh_ai/engine/ocgcore/`) already implements the full duel lifecycle — card
reader, script reader, duel creation, deck loading, and the
process/message/response loop. The remaining work to play full games is decoding
each `MSG_SELECT_*` decision message into the flat action space; that's isolated
and documented in `docs/ARCHITECTURE.md`.

> The hosted Claude Code sandbox blocks GitHub egress, so the engine can't be
> built inside it — run `fetch_engine.sh` locally or in CI that allows GitHub.

## How it learns (AlphaZero)

1. **Search** — PUCT MCTS explores the decision tree from the current position,
   guided by a neural net's policy priors and value estimate (or, before any
   training, by random rollouts).
2. **Self-play** — the agent plays itself; each move records the MCTS visit
   distribution (policy target) and the eventual game result (value target).
3. **Train** — the net is optimized to match the search's policy and the game
   outcomes. A stronger net makes stronger search, which generates better data.
   Repeat.

## Roadmap

See [`docs/ROADMAP.md`](docs/ROADMAP.md). Highlights: finish ocgcore decision
decoding, a structured field-aware observation encoder + ResNet, determinization
/ ISMCTS for hidden information, deck representation, distributed self-play, and
meta-deck benchmarking.

## License

MIT.
