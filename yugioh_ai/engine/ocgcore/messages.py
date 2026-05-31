"""ocgcore message-type and location constants, and a little binary reader.

Message type ids and location bitflags below match ProjectIgnis ocgcore
(``common.h`` / ``ocgapi_types.h``). Verify against the headers you build
against; the values are stable across recent versions.
"""

from __future__ import annotations

import struct

# -- card locations (bitflags) ------------------------------------------
LOCATION_DECK = 0x01
LOCATION_HAND = 0x02
LOCATION_MZONE = 0x04
LOCATION_SZONE = 0x08
LOCATION_GRAVE = 0x10
LOCATION_REMOVED = 0x20
LOCATION_EXTRA = 0x40
LOCATION_OVERLAY = 0x80

# -- a subset of duel message types -------------------------------------
MSG_RETRY = 1
MSG_HINT = 2
MSG_WIN = 5
MSG_SELECT_BATTLECMD = 10
MSG_SELECT_IDLECMD = 11
MSG_SELECT_EFFECTYN = 12
MSG_SELECT_YESNO = 13
MSG_SELECT_OPTION = 14
MSG_SELECT_CARD = 15
MSG_SELECT_CHAIN = 16
MSG_SELECT_PLACE = 18
MSG_SELECT_POSITION = 19
MSG_SELECT_TRIBUTE = 20
MSG_SELECT_COUNTER = 22
MSG_SELECT_SUM = 23
MSG_SELECT_DISFIELD = 24
MSG_SHUFFLE_DECK = 32
MSG_SHUFFLE_HAND = 33
MSG_NEW_TURN = 40
MSG_NEW_PHASE = 41
MSG_MOVE = 50
MSG_DRAW = 90
MSG_DAMAGE = 91
MSG_RECOVER = 92
MSG_LPUPDATE = 94
MSG_PAY_LPCOST = 100

# Messages that hand control back to a player and therefore define a decision
# point we must expose as legal actions. The rest are "informational" and we
# fold their effects into our mirrored state.
DECISION_MESSAGES = frozenset(
    {
        MSG_SELECT_BATTLECMD,
        MSG_SELECT_IDLECMD,
        MSG_SELECT_EFFECTYN,
        MSG_SELECT_YESNO,
        MSG_SELECT_OPTION,
        MSG_SELECT_CARD,
        MSG_SELECT_CHAIN,
        MSG_SELECT_PLACE,
        MSG_SELECT_POSITION,
        MSG_SELECT_TRIBUTE,
        MSG_SELECT_COUNTER,
        MSG_SELECT_SUM,
        MSG_SELECT_DISFIELD,
    }
)


class BufferReader:
    """Little-endian reader for ocgcore message/query byte buffers."""

    def __init__(self, data: bytes):
        self.data = data
        self.pos = 0

    def remaining(self) -> int:
        return len(self.data) - self.pos

    def _take(self, n: int) -> bytes:
        b = self.data[self.pos : self.pos + n]
        if len(b) != n:
            raise EOFError("buffer underrun reading ocgcore message")
        self.pos += n
        return b

    def u8(self) -> int:
        return self._take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self._take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self._take(4))[0]

    def i32(self) -> int:
        return struct.unpack("<i", self._take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self._take(8))[0]
