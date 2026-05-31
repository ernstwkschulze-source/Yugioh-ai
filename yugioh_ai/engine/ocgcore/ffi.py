"""ctypes bindings for ygopro-core / ocgcore (the real Yu-Gi-Oh rules engine).

This binds the stable ``OCG_*`` C API exported by ``libocgcore`` (ProjectIgnis /
Fluorohydride). Use ``scripts/fetch_engine.sh`` to clone and build the core and
fetch the card database + Lua scripts.

.. important::
   The struct field layouts below mirror ``ocgapi.h`` as of the ProjectIgnis
   ``ocgcore`` API. Engine ABI can drift between versions. After fetching the
   sources, verify these structs against the actual ``ocgapi.h`` you build
   against (``third_party/ygopro-core/ocgapi.h``). The function *signatures*
   are the contract that matters most and are stable across recent versions.
"""

from __future__ import annotations

import ctypes
import os
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_char_p,
    c_int,
    c_uint8,
    c_uint32,
    c_uint64,
    c_void_p,
)

# -- callback function pointer types ------------------------------------
# int (*OCG_DataReader)(void* payload, uint32_t code, OCG_CardData* data)
# const char* (*OCG_ScriptReader)(void* payload, OCG_Duel duel, const char* name)
# void (*OCG_LogHandler)(void* payload, const char* string, int type)
# void (*OCG_DataReaderDone)(void* payload, OCG_CardData* data)
OCG_DataReader = CFUNCTYPE(None, c_void_p, c_uint32, c_void_p)
OCG_ScriptReader = CFUNCTYPE(c_int, c_void_p, c_void_p, c_char_p)
OCG_LogHandler = CFUNCTYPE(None, c_void_p, c_char_p, c_int)
OCG_DataReaderDone = CFUNCTYPE(None, c_void_p, c_void_p)


class OCG_CardData(Structure):
    """Mirrors ``OCG_CardData`` (card metadata handed back by the data reader)."""

    _fields_ = [
        ("code", c_uint32),
        ("alias", c_uint32),
        ("setcodes", POINTER(c_uint64)),  # 0-terminated list
        ("type", c_uint32),
        ("level", c_uint32),
        ("attribute", c_uint32),
        ("race", c_uint64),
        ("attack", c_int),
        ("defense", c_int),
        ("lscale", c_uint32),
        ("rscale", c_uint32),
        ("link_marker", c_uint32),
    ]


class OCG_Player(Structure):
    _fields_ = [
        ("startingLP", c_uint32),
        ("startingDrawCount", c_uint32),
        ("drawCountPerTurn", c_uint32),
    ]


class OCG_DuelOptions(Structure):
    """Mirrors ``OCG_DuelOptions``. ``seed`` is a 4-word xoshiro state in recent
    cores; verify against the header you build against."""

    _fields_ = [
        ("seed", c_uint64 * 4),
        ("flags", c_uint64),
        ("team1", OCG_Player),
        ("team2", OCG_Player),
        ("cardReader", OCG_DataReader),
        ("payload1", c_void_p),
        ("scriptReader", OCG_ScriptReader),
        ("payload2", c_void_p),
        ("logHandler", OCG_LogHandler),
        ("payload3", c_void_p),
        ("cardReaderDone", OCG_DataReaderDone),
        ("payload4", c_void_p),
        ("enableUnsafeLibraries", c_uint8),
    ]


class OCG_NewCardInfo(Structure):
    _fields_ = [
        ("team", c_uint8),
        ("duelist", c_uint8),
        ("code", c_uint32),
        ("con", c_uint8),
        ("loc", c_uint32),
        ("seq", c_uint32),
        ("pos", c_uint32),
    ]


class OCG_QueryInfo(Structure):
    _fields_ = [
        ("flags", c_uint32),
        ("con", c_uint8),
        ("loc", c_uint32),
        ("seq", c_uint32),
        ("overlay_seq", c_uint32),
    ]


# DuelProcess return flags.
OCG_DUEL_STATUS_END = 0
OCG_DUEL_STATUS_AWAITING = 1
OCG_DUEL_STATUS_CONTINUE = 2


def _candidate_lib_paths() -> list[str]:
    here = os.path.dirname(__file__)
    root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    names = ["libocgcore.so", "ocgcore.so", "libocgcore.dylib", "ocgcore.dll"]
    dirs = [
        os.environ.get("OCGCORE_LIB_DIR", ""),
        os.path.join(root, "engine_build"),
        os.path.join(root, "third_party", "ygopro-core"),
        root,
    ]
    out = []
    explicit = os.environ.get("OCGCORE_LIB")
    if explicit:
        out.append(explicit)
    for d in dirs:
        if not d:
            continue
        for n in names:
            out.append(os.path.join(d, n))
    return out


class OcgError(RuntimeError):
    pass


def load_library(path: str | None = None) -> ctypes.CDLL:
    """Load ``libocgcore`` and configure ``argtypes``/``restype``.

    Raises :class:`OcgError` with build instructions if not found.
    """
    candidates = [path] if path else _candidate_lib_paths()
    lib_path = next((p for p in candidates if p and os.path.exists(p)), None)
    if lib_path is None:
        raise OcgError(
            "libocgcore not found. Build the real engine first:\n"
            "  bash scripts/fetch_engine.sh\n"
            "or point OCGCORE_LIB at an existing shared library.\n"
            f"Searched: {[c for c in candidates if c]}"
        )
    lib = ctypes.CDLL(lib_path)
    _bind(lib)
    return lib


def _bind(lib: ctypes.CDLL) -> None:
    lib.OCG_GetVersion.restype = None
    lib.OCG_GetVersion.argtypes = [POINTER(c_int), POINTER(c_int)]

    lib.OCG_CreateDuel.restype = c_int
    lib.OCG_CreateDuel.argtypes = [POINTER(c_void_p), OCG_DuelOptions]

    lib.OCG_DestroyDuel.restype = None
    lib.OCG_DestroyDuel.argtypes = [c_void_p]

    lib.OCG_DuelNewCard.restype = None
    lib.OCG_DuelNewCard.argtypes = [c_void_p, OCG_NewCardInfo]

    lib.OCG_StartDuel.restype = None
    lib.OCG_StartDuel.argtypes = [c_void_p]

    lib.OCG_DuelProcess.restype = c_int
    lib.OCG_DuelProcess.argtypes = [c_void_p]

    lib.OCG_DuelGetMessage.restype = c_void_p
    lib.OCG_DuelGetMessage.argtypes = [c_void_p, POINTER(c_uint32)]

    lib.OCG_DuelSetResponse.restype = None
    lib.OCG_DuelSetResponse.argtypes = [c_void_p, c_void_p, c_uint32]

    lib.OCG_LoadScript.restype = c_int
    lib.OCG_LoadScript.argtypes = [c_void_p, c_char_p, c_uint32, c_char_p]

    lib.OCG_DuelQueryCount.restype = c_uint32
    lib.OCG_DuelQueryCount.argtypes = [c_void_p, c_uint8, c_uint32]

    lib.OCG_DuelQuery.restype = c_void_p
    lib.OCG_DuelQuery.argtypes = [c_void_p, POINTER(c_uint32), OCG_QueryInfo]

    lib.OCG_DuelQueryLocation.restype = c_void_p
    lib.OCG_DuelQueryLocation.argtypes = [c_void_p, POINTER(c_uint32), OCG_QueryInfo]

    lib.OCG_DuelQueryField.restype = c_void_p
    lib.OCG_DuelQueryField.argtypes = [c_void_p, POINTER(c_uint32)]


def get_version(lib: ctypes.CDLL) -> tuple[int, int]:
    major, minor = c_int(0), c_int(0)
    lib.OCG_GetVersion(ctypes.byref(major), ctypes.byref(minor))
    return major.value, minor.value
