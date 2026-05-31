# Architecture

## Layers

```
            +-------------------------------------------------+
   agents / |  MCTS (PUCT)   self-play   training   CLI       |   <- engine-agnostic
   rl       +-------------------------------------------------+
                                |  Backend interface
            +-------------------------------------------------+
   engine   |  MockBackend         |   OcgcoreBackend         |
            |  (toy, pure-python)  |   (real ygopro-core)     |
            +-------------------------------------------------+
```

The `Backend` interface (`yugioh_ai/engine/backend.py`) is the contract:
`reset / current_player / legal_actions / step / observation / is_terminal /
winner / clone`. Nothing above the line knows which engine it is using.

* **MockBackend** — a small, fully deterministic two-player board game. It is not
  Yu-Gi-Oh; it exists so the entire RL stack runs and is tested in milliseconds,
  and it is deliberately deep enough (summons, board, battles, life totals) that
  search beats random play — proving the pipeline actually optimizes.
* **OcgcoreBackend** — the real engine.

## The action space

The policy head and MCTS index a single flat integer action space. For the mock
backend it is 9 actions (summon slots, attack slots, end turn). For ocgcore it is
a fixed-size space (`YGO_ACTION_SPACE`, default 4096) decomposed by message type:

| Decision message      | Sub-space meaning                                   |
|-----------------------|-----------------------------------------------------|
| `MSG_SELECT_IDLECMD`  | (summon / set / activate / sp-summon / to-bp / to-ep) × card index |
| `MSG_SELECT_BATTLECMD`| attack with monster i / activate / to main2 / end   |
| `MSG_SELECT_CARD`     | choose card index/combination from the offered list |
| `MSG_SELECT_CHAIN`    | which chainable effect (or none)                    |
| `MSG_SELECT_YESNO` / `EFFECTYN` | yes / no                                  |
| `MSG_SELECT_OPTION`   | option index                                        |
| `MSG_SELECT_PLACE` / `POSITION` / `TRIBUTE` / `SUM` / `COUNTER` | zone / position / tribute set / sum subset / counter count |

Each message type owns a disjoint slice of the flat space; the legal mask zeroes
everything not currently offered. This keeps the network's policy head fixed-size
while the *meaning* of an action id depends on the pending decision.

## Plugging in ocgcore

`yugioh_ai/engine/ocgcore/` already implements:

1. **FFI** (`ffi.py`) — ctypes bindings for the `OCG_*` API and its structs.
2. **Card data** — a sqlite reader over `cards.cdb` feeding `OCG_CardData`.
3. **Scripts** — a reader that serves Lua files via `OCG_LoadScript`.
4. **Lifecycle** — `OCG_CreateDuel` → `OCG_DuelNewCard` (decks) → `OCG_StartDuel`.
5. **Message loop** — `OCG_DuelProcess` / `OCG_DuelGetMessage` /
   `OCG_DuelSetResponse`, with a per-message dispatch table.

What remains for full play (the single documented extension point in
`backend.py::_handle_message` / `_encode_response`):

* Decode each decision `MSG_SELECT_*` body into the legal subset of the flat
  action space (set `_pending_player` + a legal list).
* Encode a chosen action id back into the response byte layout the engine wants.
* Build the observation encoder from `OCG_DuelQueryField` / `OCG_DuelQueryLocation`
  (life points, zones, card ids/positions, counters) into model planes.

Verify all struct layouts and message ids against the headers you build against
(`third_party/ygopro-core/ocgapi.h`, `common.h`).

## Search (AlphaZero PUCT)

`agents/mcts.py`. Standard PUCT with Dirichlet root noise. The one non-standard
detail: turns do **not** strictly alternate (a player takes several actions per
turn), so value is propagated per-perspective (each edge accumulates value from
the perspective of the player who chose it) instead of naive negamax sign flips.

## Hidden information

Yu-Gi-Oh is imperfect-information. v0.1 search clones full state (fine for the
mock testbed and for training from a player's own observation). For the real
engine, faithful cloning + correct search require **determinization / ISMCTS**
(sample opponent hand/deck consistent with public info, search, average) — see
the roadmap.
