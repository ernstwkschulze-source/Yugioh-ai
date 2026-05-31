"""Backend backed by the real ygopro-core / ocgcore engine.

Status: this is the real integration *skeleton*. It wires up the full duel
lifecycle against the ``OCG_*`` C API:

* loads ``libocgcore`` (see :mod:`yugioh_ai.engine.ocgcore.ffi`),
* serves card metadata from ``cards.cdb`` (sqlite) via the data-reader callback,
* serves Lua card scripts from the ``script/`` directory via the script-reader,
* creates a duel, loads both decks, starts it, and
* drives the ``OCG_DuelProcess`` / ``OCG_DuelGetMessage`` / ``OCG_DuelSetResponse``
  loop, dispatching messages to a handler registry.

What is intentionally left as an extension point: the binary parsing of each
``MSG_*`` decision message into our flat action space, and the mapping of a
chosen action id back into the response bytes. Those layouts are engine-version
specific; they are isolated in :meth:`_handle_message` / :meth:`_encode_response`
so they can be implemented and verified against the headers you build against,
without touching anything above the Backend interface. Until they are filled in,
this backend raises a clear ``NotImplementedError`` at the first decision point.

To build the engine + data:  ``bash scripts/fetch_engine.sh``
"""

from __future__ import annotations

import ctypes
import os
import sqlite3
from typing import Sequence

import numpy as np

from yugioh_ai.engine.backend import Backend, Observation, StepResult
from yugioh_ai.engine.ocgcore import ffi, messages as M

# Global action space for the real engine. Yu-Gi-Oh's branching factor is large
# but bounded; we reserve a flat space the policy head indexes into. The exact
# decomposition (idle commands x field slots, battle attacks, yes/no, option
# indices, card-select combinations) is documented in docs/ARCHITECTURE.md.
OCG_ACTION_SPACE = int(os.environ.get("YGO_ACTION_SPACE", "4096"))


class OcgcoreBackend(Backend):
    action_space_size = OCG_ACTION_SPACE

    def __init__(
        self,
        *,
        lib_path: str | None = None,
        database: str | None = None,
        script_dir: str | None = None,
    ) -> None:
        self.lib = ffi.load_library(lib_path)
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        self.database = database or os.environ.get(
            "YGO_CARDS_DB", os.path.join(root, "cards.cdb")
        )
        self.script_dir = script_dir or os.environ.get(
            "YGO_SCRIPT_DIR", os.path.join(root, "script")
        )
        if not os.path.exists(self.database):
            raise ffi.OcgError(
                f"card database not found at {self.database}. Run scripts/fetch_engine.sh "
                "or set YGO_CARDS_DB."
            )
        self._db = sqlite3.connect(self.database)
        self._duel: ctypes.c_void_p | None = None
        # Keep python refs to callbacks alive for the lifetime of the duel.
        self._cb_refs: list = []
        self._terminal = False
        self._winner: int | None = None
        self._pending_player = 0
        self._pending_msg: bytes | None = None

    # -- callbacks -------------------------------------------------------
    def _make_card_reader(self):
        def reader(payload, code, data_ptr):  # OCG_DataReader
            row = self._db.execute(
                "SELECT id,alias,type,level,attribute,race,atk,def,lscale,rscale,link "
                "FROM datas WHERE id=?",
                (code,),
            ).fetchone()
            if not row:
                return
            cd = ctypes.cast(data_ptr, ctypes.POINTER(ffi.OCG_CardData)).contents
            (cid, alias, ctype, level, attr, race, atk, dfn, ls, rs, link) = row
            cd.code = cid
            cd.alias = alias or 0
            cd.type = ctype or 0
            cd.level = (level or 0) & 0xFF
            cd.attribute = attr or 0
            cd.race = race or 0
            cd.attack = atk if atk is not None else 0
            cd.defense = dfn if dfn is not None else 0
            cd.lscale = ((level or 0) >> 24) & 0xFF
            cd.rscale = ((level or 0) >> 16) & 0xFF
            cd.link_marker = link or 0
            cd.setcodes = None

        return ffi.OCG_DataReader(reader)

    def _make_script_reader(self):
        def reader(payload, duel, name):  # OCG_ScriptReader
            fname = name.decode() if isinstance(name, bytes) else str(name)
            base = os.path.basename(fname)
            path = os.path.join(self.script_dir, base)
            if not os.path.exists(path):
                path = fname if os.path.exists(fname) else path
            try:
                with open(path, "rb") as fh:
                    buf = fh.read()
            except OSError:
                return 0
            return self.lib.OCG_LoadScript(duel, buf, len(buf), base.encode())

        return ffi.OCG_ScriptReader(reader)

    def _make_log_handler(self):
        def handler(payload, string, ltype):  # OCG_LogHandler
            return None

        return ffi.OCG_LogHandler(handler)

    # -- lifecycle -------------------------------------------------------
    def reset(self, *, seed: int, decks: Sequence[Sequence[int]] | None = None) -> None:
        if decks is None:
            raise ValueError("OcgcoreBackend requires explicit decks (list of card ids per player)")
        self._destroy()

        opts = ffi.OCG_DuelOptions()
        opts.seed = (ctypes.c_uint64 * 4)(seed & 0xFFFFFFFFFFFFFFFF, 0, 0, 0)
        opts.flags = 0
        opts.team1 = ffi.OCG_Player(8000, 5, 1)
        opts.team2 = ffi.OCG_Player(8000, 5, 1)

        card_reader = self._make_card_reader()
        script_reader = self._make_script_reader()
        log_handler = self._make_log_handler()
        self._cb_refs = [card_reader, script_reader, log_handler]
        opts.cardReader = card_reader
        opts.scriptReader = script_reader
        opts.logHandler = log_handler
        opts.cardReaderDone = ffi.OCG_DataReaderDone(0)
        opts.enableUnsafeLibraries = 1

        duel = ctypes.c_void_p()
        rc = self.lib.OCG_CreateDuel(ctypes.byref(duel), opts)
        if rc != 0:
            raise ffi.OcgError(f"OCG_CreateDuel failed with code {rc}")
        self._duel = duel

        for team, deck in enumerate(decks):
            for code in deck:
                info = ffi.OCG_NewCardInfo()
                info.team = team
                info.duelist = 0
                info.code = int(code)
                info.con = team
                info.loc = M.LOCATION_DECK
                info.seq = 0
                info.pos = 0
                self.lib.OCG_DuelNewCard(duel, info)

        self.lib.OCG_StartDuel(duel)
        self._terminal = False
        self._winner = None
        self._advance()

    def _destroy(self) -> None:
        if self._duel is not None:
            self.lib.OCG_DestroyDuel(self._duel)
            self._duel = None
            self._cb_refs = []

    # -- message loop ----------------------------------------------------
    def _advance(self) -> None:
        """Run the engine until the next decision point or terminal."""
        while True:
            status = self.lib.OCG_DuelProcess(self._duel)
            length = ctypes.c_uint32(0)
            ptr = self.lib.OCG_DuelGetMessage(self._duel, ctypes.byref(length))
            if ptr and length.value:
                blob = ctypes.string_at(ptr, length.value)
                if self._consume_messages(blob):
                    return
            if status == ffi.OCG_DUEL_STATUS_END:
                self._terminal = True
                return
            if status == ffi.OCG_DUEL_STATUS_AWAITING:
                return

    def _consume_messages(self, blob: bytes) -> bool:
        """Parse a batch of messages. Returns True if we hit a decision point."""
        r = M.BufferReader(blob)
        while r.remaining() > 0:
            msg_len = r.u32()
            start = r.pos
            msg_type = r.u8()
            hit = self._handle_message(msg_type, r)
            r.pos = start + msg_len  # skip to next message regardless
            if hit:
                return True
        return False

    def _handle_message(self, msg_type: int, r: M.BufferReader) -> bool:
        """Update mirrored state from an informational message, or expose a
        decision. Returns True iff this message is a decision point.

        Informational messages we understand are folded into state here. The
        binary layout of each ``MSG_*`` is engine-version specific -- fill these
        in against your built headers. Returning True stores the raw message so
        :meth:`legal_actions` / :meth:`step` can act on it.
        """
        if msg_type == M.MSG_WIN:
            player = r.u8()
            self._terminal = True
            self._winner = int(player)
            return False

        if msg_type in M.DECISION_MESSAGES:
            # NOTE: decode `player` and the option set, set self._pending_*,
            # then return True. The per-message decode + action mapping is the
            # remaining work for full play; see docs/ARCHITECTURE.md.
            raise NotImplementedError(
                f"ocgcore decision message {msg_type} decoding not implemented yet. "
                "This is the documented extension point: parse the message into the "
                "flat action space and implement _encode_response(). The lifecycle, "
                "card/script readers and process loop around it are complete."
            )

        # Informational message we don't track yet -> ignore (state mirrored
        # lazily via OCG_DuelQuery* when an observation is requested).
        return False

    def _encode_response(self, action: int) -> bytes:  # pragma: no cover
        """Map a flat action id into the response bytes for the pending message."""
        raise NotImplementedError

    # -- Backend API -----------------------------------------------------
    def current_player(self) -> int:
        return self._pending_player

    def legal_actions(self) -> list[int]:
        if self._terminal:
            return []
        raise NotImplementedError(
            "legal_actions for ocgcore requires decision-message decoding "
            "(see _handle_message)."
        )

    def step(self, action: int) -> StepResult:
        resp = self._encode_response(action)
        self.lib.OCG_DuelSetResponse(self._duel, resp, len(resp))
        self._advance()
        if self._terminal:
            mover = self._pending_player
            reward = 0.0 if self._winner is None else (1.0 if self._winner == mover else -1.0)
            return StepResult(True, self._winner, reward)
        return StepResult(False, None)

    def observation(self, player: int) -> Observation:
        # A full encoder reads OCG_DuelQueryField / OCG_DuelQueryLocation and
        # builds model planes; documented in docs/ARCHITECTURE.md. For now return
        # an empty-but-correctly-shaped observation so the interface is usable.
        planes = np.zeros((1,), dtype=np.float32)
        mask = np.zeros(self.action_space_size, dtype=bool)
        return Observation(planes=planes, legal_mask=mask, to_move=self._pending_player)

    def is_terminal(self) -> bool:
        return self._terminal

    def winner(self) -> int | None:
        return self._winner

    def clone(self) -> "OcgcoreBackend":
        # ocgcore has no native duel-copy; faithful cloning for search requires
        # replaying the message log into a fresh duel (determinization point for
        # imperfect information). Tracked in docs/ROADMAP.md.
        raise NotImplementedError(
            "OcgcoreBackend.clone requires duel replay/determinization (roadmap). "
            "Use MockBackend for MCTS development until this lands."
        )

    def __del__(self):
        try:
            self._destroy()
        except Exception:
            pass
