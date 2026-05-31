"""A self-contained toy duel that mimics the *shape* of Yu-Gi-Oh decisions.

This is NOT Yu-Gi-Oh. It is a small, fully deterministic two-player board game
whose only job is to exercise the entire stack (env -> agents -> MCTS ->
self-play -> training) without needing the real engine, while being deep enough
that *skill matters* (so MCTS reliably beats random play -- a real testbed, not
a coin flip). It has:

* hidden information (you see your hand, only the *size* of the opponent's),
* persistent board state (summoned monsters), so positions have structure,
* genuine tactical tension (summon timing, whether to attack into a bigger
  monster and lose life, board trades),
* a clear win condition and guaranteed termination.

Rules
-----
* Two players, 20 life points each.
* A deck of monsters with ATK 1..4. Each turn the active player draws 1 (opening
  hand of 5).
* On your turn you may, then must END_TURN:
    - SUMMON hand slot i : place a monster on your field (max 3) -- **once per turn**.
    - ATTACK with your monster j (each monster once per turn):
        * if the opponent controls monsters, you battle their *strongest* one:
          higher ATK destroys lower and the loser's controller takes the ATK
          difference as LP damage; equal ATK destroys both.
        * if the opponent has no monsters, it's a direct attack for ATK damage.
    - The player who goes first may not attack on their first turn.
* You lose if your LP hits 0, or if you must draw with an empty deck and empty
  hand (deck-out). A turn cap forces termination (lower LP loses, else draw).

Determinism: deck order is fixed at :meth:`reset` from the seed; :meth:`clone`
is an exact copy, so search rollouts are faithful.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from yugioh_ai.engine.backend import ActionMeta, Backend, Observation, StepResult

HAND_CAP = 5
FIELD_CAP = 3
START_LP = 20
DECK_SIZE = 30
MAX_ATK = 4
MAX_TURNS = 100  # safety cap -> guaranteed termination

# Action layout (fixed-size global action space).
A_SUMMON0 = 0                       # SUMMON hand slot 0..HAND_CAP-1  -> 0..4
A_ATTACK0 = HAND_CAP               # ATTACK field monster 0..FIELD_CAP-1 -> 5..7
A_END_TURN = HAND_CAP + FIELD_CAP  # 8
ACTION_SPACE = HAND_CAP + FIELD_CAP + 1  # 9

# Observation vector layout (see _encode).
OBS_SIZE = 6 + FIELD_CAP * 2 + HAND_CAP


class MockBackend(Backend):
    action_space_size = ACTION_SPACE

    def __init__(self) -> None:
        self._reset_state(seed=0, decks=None)

    # -- lifecycle -------------------------------------------------------
    def reset(self, *, seed: int, decks: Sequence[Sequence[int]] | None = None) -> None:
        self._reset_state(seed=seed, decks=decks)

    def _reset_state(self, *, seed: int, decks) -> None:
        rng = np.random.default_rng(seed)
        if decks is None:
            decks = [
                list(int(x) for x in rng.integers(1, MAX_ATK + 1, size=DECK_SIZE)),
                list(int(x) for x in rng.integers(1, MAX_ATK + 1, size=DECK_SIZE)),
            ]
        self.lp = [START_LP, START_LP]
        self.deck = [list(d) for d in decks]
        self.hand = [[], []]
        self.field = [[], []]            # list of monster ATK values per player
        self._attacked = [set(), set()]  # field indices that already attacked this turn
        self.active = 0
        self.turn = 0                    # number of completed turns
        self.summoned_this_turn = False
        self._winner: int | None = None
        self._terminal = False
        for p in (0, 1):
            self._draw(p, 5)

    # -- internals -------------------------------------------------------
    def _draw(self, p: int, n: int) -> None:
        # Hand is capped at HAND_CAP so that summon action ids (0..HAND_CAP-1)
        # can never overflow into the attack id range (A_ATTACK0 = HAND_CAP).
        for _ in range(n):
            if self.deck[p] and len(self.hand[p]) < HAND_CAP:
                self.hand[p].append(self.deck[p].pop(0))

    def _first_turn_no_attack(self) -> bool:
        # The player who went first (player 0) cannot attack on turn 0.
        return self.turn == 0 and self.active == 0

    def _win(self, who: int) -> StepResult:
        self._terminal = True
        self._winner = who
        return self._result_for(self.active)

    def _result_for(self, mover: int) -> StepResult:
        if not self._terminal:
            return StepResult(False, None)
        reward = 0.0 if self._winner is None else (1.0 if self._winner == mover else -1.0)
        return StepResult(True, self._winner, reward)

    # -- Backend API -----------------------------------------------------
    def current_player(self) -> int:
        return self.active

    def legal_actions(self) -> list[int]:
        if self._terminal:
            return []
        p = self.active
        acts: list[int] = []
        if not self.summoned_this_turn and len(self.field[p]) < FIELD_CAP:
            for i in range(min(len(self.hand[p]), HAND_CAP)):
                acts.append(A_SUMMON0 + i)
        if not self._first_turn_no_attack():
            for j in range(len(self.field[p])):
                if j not in self._attacked[p]:
                    acts.append(A_ATTACK0 + j)
        acts.append(A_END_TURN)
        return acts

    def step(self, action: int) -> StepResult:
        if self._terminal:
            raise RuntimeError("step() called on a terminal duel")
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action {action}; legal={self.legal_actions()}")
        p, opp = self.active, 1 - self.active

        if action == A_END_TURN:
            return self._end_turn(p, opp)

        if A_SUMMON0 <= action < A_SUMMON0 + HAND_CAP:
            slot = action - A_SUMMON0
            self.field[p].append(self.hand[p].pop(slot))
            self.summoned_this_turn = True
            return StepResult(False, None)

        # ATTACK
        j = action - A_ATTACK0
        attacker = self.field[p][j]
        self._attacked[p].add(j)
        if not self.field[opp]:
            self.lp[opp] -= attacker
            if self.lp[opp] <= 0:
                return self._win(p)
            return StepResult(False, None)
        # battle the opponent's strongest monster
        d_idx = int(np.argmax(self.field[opp]))
        defender = self.field[opp][d_idx]
        if attacker > defender:
            self.field[opp].pop(d_idx)
            self.lp[opp] -= attacker - defender
        elif attacker < defender:
            self.field[p].pop(j)
            self._attacked[p] = {k - 1 if k > j else k for k in self._attacked[p] if k != j}
            self.lp[p] -= defender - attacker
        else:  # equal -> both destroyed
            self.field[opp].pop(d_idx)
            self.field[p].pop(j)
            self._attacked[p] = {k - 1 if k > j else k for k in self._attacked[p] if k != j}
        if self.lp[p] <= 0 and self.lp[opp] <= 0:
            return self._win(opp)  # active player loses on simultaneous
        if self.lp[opp] <= 0:
            return self._win(p)
        if self.lp[p] <= 0:
            return self._win(opp)
        return StepResult(False, None)

    def _end_turn(self, p: int, opp: int) -> StepResult:
        self.turn += 1
        if self.turn >= MAX_TURNS:
            self._terminal = True
            if self.lp[0] == self.lp[1]:
                self._winner = None
            else:
                self._winner = 0 if self.lp[0] > self.lp[1] else 1
            return self._result_for(p)
        self.active = opp
        self.summoned_this_turn = False
        self._attacked[opp] = set()
        self._draw(opp, 1)
        if not self.deck[opp] and not self.hand[opp] and not self.field[opp]:
            return self._win(p)  # opponent decked out with no board
        return StepResult(False, None)

    def observation(self, player: int) -> Observation:
        opp = 1 - player
        planes = self._encode(player)
        mask = np.zeros(self.action_space_size, dtype=bool)
        if player == self.active and not self._terminal:
            mask[self.legal_actions()] = True
        info = {
            "lp": list(self.lp),
            "hand": list(self.hand[player]),
            "field": [list(self.field[0]), list(self.field[1])],
            "opp_hand_size": len(self.hand[opp]),
            "deck": [len(self.deck[0]), len(self.deck[1])],
            "to_move": self.active,
            "turn": self.turn,
        }
        return Observation(planes=planes, legal_mask=mask, to_move=self.active, info=info)

    def _encode(self, player: int) -> np.ndarray:
        opp = 1 - player
        v = np.zeros(OBS_SIZE, dtype=np.float32)
        v[0] = self.lp[player] / START_LP
        v[1] = self.lp[opp] / START_LP
        v[2] = len(self.deck[player]) / DECK_SIZE
        v[3] = len(self.hand[opp]) / HAND_CAP
        v[4] = 1.0 if self.active == player else 0.0
        v[5] = 0.0 if self.summoned_this_turn else 1.0
        base = 6
        for i, atk in enumerate(sorted(self.field[player], reverse=True)[:FIELD_CAP]):
            v[base + i] = atk / MAX_ATK
        base += FIELD_CAP
        for i, atk in enumerate(sorted(self.field[opp], reverse=True)[:FIELD_CAP]):
            v[base + i] = atk / MAX_ATK
        base += FIELD_CAP
        for i, atk in enumerate(self.hand[player][:HAND_CAP]):
            v[base + i] = atk / MAX_ATK
        return v

    def is_terminal(self) -> bool:
        return self._terminal

    def winner(self) -> int | None:
        return self._winner

    def clone(self) -> "MockBackend":
        new = MockBackend.__new__(MockBackend)
        new.lp = list(self.lp)
        new.deck = [list(d) for d in self.deck]
        new.hand = [list(h) for h in self.hand]
        new.field = [list(f) for f in self.field]
        new._attacked = [set(self._attacked[0]), set(self._attacked[1])]
        new.active = self.active
        new.turn = self.turn
        new.summoned_this_turn = self.summoned_this_turn
        new._winner = self._winner
        new._terminal = self._terminal
        return new

    def describe_action(self, action: int) -> ActionMeta:
        if action == A_END_TURN:
            return ActionMeta(action, "END_TURN")
        if A_SUMMON0 <= action < A_SUMMON0 + HAND_CAP:
            return ActionMeta(action, f"SUMMON hand[{action - A_SUMMON0}]")
        return ActionMeta(action, f"ATTACK field[{action - A_ATTACK0}]")
