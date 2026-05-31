#!/usr/bin/env bash
# Fetch and build the real Yu-Gi-Oh rules engine (ygopro-core / ocgcore) plus
# the card database and Lua card scripts, so the `ocgcore` backend can run.
#
# Requires network access to GitHub and a C++17 toolchain + cmake. Run this on a
# machine/CI with GitHub egress (the hosted Claude Code sandbox may block it).
#
# Outputs (git-ignored):
#   third_party/ygopro-core   ocgcore sources + ocgapi.h (verify FFI against this)
#   engine_build/libocgcore.so
#   cards.cdb                 sqlite card database
#   script/                   Lua card scripts
#
# Usage: bash scripts/fetch_engine.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TP="$ROOT/third_party"
BUILD="$ROOT/engine_build"
mkdir -p "$TP" "$BUILD"

clone() { # repo dir
  if [ -d "$2/.git" ]; then
    echo "== updating $2"; git -C "$2" pull --ff-only || true
  else
    echo "== cloning $1"; git clone --depth 1 "$1" "$2"
  fi
}

# 1. Engine core + script bindings library it depends on.
clone https://github.com/edo9300/ygopro-core.git "$TP/ygopro-core"
clone https://github.com/ProjectIgnis/CardScripts.git "$TP/CardScripts"
clone https://github.com/ProjectIgnis/BabelCDB.git    "$TP/BabelCDB"

# 2. Build libocgcore as a shared library.
echo "== building libocgcore"
cat > "$TP/ygopro-core/CMakeLists.txt.ygoai" <<'CMAKE'
cmake_minimum_required(VERSION 3.16)
project(ocgcore CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
file(GLOB SRC "*.cpp")
add_library(ocgcore SHARED ${SRC})
find_package(Lua REQUIRED)
target_include_directories(ocgcore PRIVATE ${LUA_INCLUDE_DIR})
target_link_libraries(ocgcore ${LUA_LIBRARIES})
CMAKE
cmake -S "$TP/ygopro-core" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -C /dev/null \
  -DCMAKE_PROJECT_INCLUDE="$TP/ygopro-core/CMakeLists.txt.ygoai" 2>/dev/null \
  || cmake -S "$TP/ygopro-core" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"
find "$BUILD" -name 'libocgcore.*' -exec cp {} "$BUILD/" \; 2>/dev/null || true

# 3. Assemble Lua scripts.
echo "== assembling scripts"
mkdir -p "$ROOT/script"
cp -f "$TP/CardScripts"/official/*.lua "$ROOT/script/" 2>/dev/null || true
cp -f "$TP/CardScripts"/*.lua          "$ROOT/script/" 2>/dev/null || true

# 4. Merge card databases into a single cards.cdb.
echo "== building cards.cdb"
python3 - "$TP/BabelCDB" "$ROOT/cards.cdb" <<'PY'
import glob, os, sqlite3, sys
src_dir, out = sys.argv[1], sys.argv[2]
if os.path.exists(out):
    os.remove(out)
con = sqlite3.connect(out)
cdbs = sorted(glob.glob(os.path.join(src_dir, "*.cdb")))
for i, cdb in enumerate(cdbs):
    con.execute("ATTACH ? AS src", (cdb,))
    if i == 0:
        for (sql,) in con.execute(
            "SELECT sql FROM src.sqlite_master WHERE type='table' AND sql NOT NULL"
        ):
            con.execute(sql)
    for (name,) in con.execute("SELECT name FROM src.sqlite_master WHERE type='table'"):
        try:
            con.execute(f"INSERT OR IGNORE INTO {name} SELECT * FROM src.{name}")
        except sqlite3.Error:
            pass
    con.execute("DETACH src")
con.commit()
n = con.execute("SELECT COUNT(*) FROM datas").fetchone()[0]
print(f"   merged {len(cdbs)} cdbs -> {n} cards")
con.close()
PY

echo
echo "Done. Verify yugioh_ai/engine/ocgcore/ffi.py structs against:"
echo "  $TP/ygopro-core/ocgapi.h"
echo "Then: python -c \"from yugioh_ai.engine import make_backend; make_backend('ocgcore')\""
