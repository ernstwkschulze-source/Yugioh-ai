#!/usr/bin/env python3
"""Play one game and print a full, human-readable duel transcript.

Shows both players' decks, opening hands, every draw, every summon, every
attack (with the battle result and damage), and the running life points -- so
you can read a match the way you'd read a duel replay, no programming needed.

Usage:
    python scripts/duel_log.py                 # AI (P0) vs Random bot (P1)
    python scripts/duel_log.py --p1 ai         # AI vs AI (a real duel)
    python scripts/duel_log.py --seed 7 --sims 100
"""

from __future__ import annotations

import argparse

from yugioh_ai.agents import MCTS, MCTSAgent, RandomAgent
from yugioh_ai.engine import make_backend
from yugioh_ai.engine.mock.backend import A_ATTACK0, A_END_TURN, A_SUMMON0, HAND_CAP


def snap(be):
    return {
        "lp": list(be.lp),
        "field": [list(be.field[0]), list(be.field[1])],
        "hand": [list(be.hand[0]), list(be.hand[1])],
        "deck": [list(be.deck[0]), list(be.deck[1])],
        "active": be.active,
        "turn": be.turn,
    }


def board(s, p):
    f = s["field"][p]
    return "[" + ", ".join(f"{v} ATK" for v in f) + "]" if f else "(empty)"


def describe(before, after, action, p):
    opp = 1 - p
    if action == A_END_TURN:
        np_ = after["active"]
        drew = len(before["deck"][np_]) - len(after["deck"][np_])
        line = f"P{p} ends the turn."
        if drew > 0:
            newcards = after["hand"][np_][len(before["hand"][np_]):]
            vals = ", ".join(f"{v} ATK" for v in newcards)
            line += f"  P{np_} draws: {vals}."
        return line

    if A_SUMMON0 <= action < A_SUMMON0 + HAND_CAP:
        added = [v for v in after["field"][p]]
        # the summoned monster is the one now on field that left the hand
        summoned = after["field"][p][-1]
        return f"P{p} Normal Summons a {summoned}-ATK monster.   P{p} board: {board(after, p)}"

    # ATTACK
    j = action - A_ATTACK0
    attacker = before["field"][p][j]
    if not before["field"][opp]:
        dmg = before["lp"][opp] - after["lp"][opp]
        return (f"P{p}'s {attacker}-ATK monster attacks DIRECTLY for {dmg}!   "
                f"P{opp} LP: {before['lp'][opp]} -> {after['lp'][opp]}")

    defender = max(before["field"][opp])
    opp_lost = len(before["field"][opp]) - len(after["field"][opp])
    self_lost = len(before["field"][p]) - len(after["field"][p])
    if opp_lost and self_lost:
        return (f"P{p}'s {attacker}-ATK attacks the {defender}-ATK monster -> "
                f"equal ATK, BOTH destroyed.")
    if opp_lost:
        dmg = before["lp"][opp] - after["lp"][opp]
        return (f"P{p}'s {attacker}-ATK attacks the {defender}-ATK monster -> "
                f"defender destroyed, P{opp} takes {dmg}.   "
                f"P{opp} LP: {before['lp'][opp]} -> {after['lp'][opp]}")
    # self lost
    dmg = before["lp"][p] - after["lp"][p]
    return (f"P{p}'s {attacker}-ATK crashes into the bigger {defender}-ATK monster -> "
            f"attacker destroyed, P{p} takes {dmg}.   "
            f"P{p} LP: {before['lp'][p]} -> {after['lp'][p]}")


def turn_banner(s, names):
    p = s["active"]
    print()
    print(f"========== Turn {s['turn'] + 1} — {names[p]} (Player {p}) ==========")
    print(f"   Life Points:   P0 {names[0]} = {s['lp'][0]}    |    P1 {names[1]} = {s['lp'][1]}")
    print(f"   Board:         P0: {board(s, 0)}    |    P1: {board(s, 1)}")
    print(f"   P{p} hand:       " + "[" + ", ".join(f"{v} ATK" for v in s['hand'][p]) + "]")
    print("   " + "-" * 40)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--sims", type=int, default=80)
    ap.add_argument("--p0", choices=["ai", "random"], default="ai")
    ap.add_argument("--p1", choices=["ai", "random"], default="random")
    args = ap.parse_args()

    be = make_backend("mock")
    be.reset(seed=args.seed)

    def mk(kind, seed):
        if kind == "ai":
            return MCTSAgent(MCTS(n_simulations=args.sims, seed=seed)), f"AI(thinks {args.sims})"
        return RandomAgent(seed=seed), "Random bot"

    a0, n0 = mk(args.p0, args.seed)
    a1, n1 = mk(args.p1, args.seed + 1)
    agents, names = [a0, a1], [n0, n1]

    print("############################################################")
    print("#  DUEL TRANSCRIPT")
    print(f"#  Player 0 = {n0}")
    print(f"#  Player 1 = {n1}")
    print("#  (A 'card' here is a monster with an Attack value 1-4.")
    print("#   Both start at 20 Life Points. First player can't attack turn 1.)")
    print("############################################################")
    for p in (0, 1):
        deck = " ".join(str(v) for v in be.deck[p])
        hand = ", ".join(f"{v} ATK" for v in be.hand[p])
        print(f"\nPlayer {p} ({names[p]}):")
        print(f"   Deck draw order (ATK):  {deck}")
        print(f"   Opening hand:           [{hand}]")

    last_active = None
    while not be.is_terminal():
        s_turn = snap(be)
        if s_turn["active"] != last_active:
            turn_banner(s_turn, names)
            last_active = s_turn["active"]
        p = be.current_player()
        action = agents[p].act(be)
        before = snap(be)
        be.step(action)
        after = snap(be)
        print("   " + describe(before, after, action, p))
        # if the turn passed, force a fresh banner next loop
        if action == A_END_TURN:
            last_active = None

    w = be.winner()
    print("\n############################################################")
    if w is None:
        print("#  RESULT: DRAW")
    else:
        print(f"#  RESULT: Player {w} ({names[w]}) WINS!")
    print(f"#  Final Life Points:  P0 = {be.lp[0]}   |   P1 = {be.lp[1]}")
    print("############################################################")


if __name__ == "__main__":
    main()
