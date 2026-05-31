# Roadmap: from toy testbed to meta-solver

The vision is the "Stockfish of Yu-Gi-Oh": an engine that simulates games, plays
autonomously, searches decision trees, learns from self-play, and solves lines in
the live TCG meta. That is a long program. Here is the staged path.

### Phase 0 — Foundation ✅ (this release)
- [x] Engine-agnostic `Backend` interface.
- [x] Mock duel backend (deterministic, cloneable, skill matters).
- [x] Gymnasium-style env wrapper.
- [x] AlphaZero-style PUCT MCTS (perspective-correct for multi-action turns).
- [x] Self-play data generation + replay buffer.
- [x] Policy/value network + AlphaZero training loop (PyTorch, optional).
- [x] ocgcore integration skeleton (FFI, card/script readers, lifecycle, msg loop).
- [x] Tests + CLI + docs.

### Phase 1 — Real rules engine
- [ ] Decode every `MSG_SELECT_*` into the flat action space (`_handle_message`).
- [ ] Encode responses (`_encode_response`).
- [ ] Observation encoder from `OCG_DuelQuery*` (zones, ids, positions, counters).
- [ ] Verify FFI structs/message ids against built `ocgapi.h`.
- [ ] End-to-end random vs random duels through ocgcore in CI (with engine cached).

### Phase 2 — Strong play on real duels
- [ ] Determinization / ISMCTS for hidden hands & deck order.
- [ ] Faithful `OcgcoreBackend.clone` (duel replay or state snapshotting).
- [ ] Structured, field-aware encoder + residual/attention network.
- [ ] Card embedding table (learned per-card vectors keyed by card id).
- [ ] Action masking + legal-move-aware policy loss.

### Phase 3 — Scale & learn
- [ ] Distributed self-play (many actors, central learner).
- [ ] Replay prioritization, target networks, learning-rate schedules.
- [ ] Evaluation gating (new net must beat old by a margin to be promoted).
- [ ] Opening/handtrap knowledge surfacing; line/"combo" explorer that reports
      forcing sequences (the "solver" output).

### Phase 4 — Meta
- [ ] Deck representation + deck-vs-deck matchup matrices on current meta lists.
- [ ] Sideboard / tech-card evaluation.
- [ ] Benchmark vs existing bots (e.g. WindBot) and vs fixed meta decks.
- [ ] Human-readable analysis: win-rate by line, key decision points, mulligans.

## Known limitations today
- The `mock` backend is a stand-in, not Yu-Gi-Oh.
- `OcgcoreBackend` stops at the first decision message until Phase 1 lands.
- The network is a plain MLP placeholder; real play needs the Phase 2 encoder.
- MCTS searches full state (no determinization yet).
