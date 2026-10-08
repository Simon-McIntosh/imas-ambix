#!/usr/bin/env bash
# imasdb.sh — install imas-alambic into a shared IMASDB folder on JT-60SA.
#
# One command, run from a development checkout on the building host.  It:
#   1. builds the engine and COCOS wheels with `uv build`,
#   2. pulls the private map bundle from GHCR with oras and the gh token,
#   3. relays the wheels and the bundle to the analysis server over ssh,
#   4. installs the engine as a uv tool on a uv-managed CPython 3.12 inside the
#      folder, and lays down a launcher that clears PYTHONPATH and points the
#      engine at the folder,
#   5. opens the folder to other facility users (read, no write) and checks it,
#   6. removes the older per-user install only once those checks pass.
#
# The folder it is given is the install-time setting.  The names of the layout
# under that folder are read from `imas-alambic config` rather than spelled
# here, so settings.py stays their one owner.
#
# Usage: packages/imas-alambic/install/imasdb.sh <IMASDB>
#
# Environment overrides:
#   IMASDB_SSH_HOST     ssh host alias of the server (default jt-60sa)
#   IMASDB_SSH_ARGS     extra ssh/scp arguments, word-split (e.g. "-F $HOME/.ssh/config")
#   IMASDB_MACHINE      machine whose bundle to install (default jt-60sa)
#   IMASDB_NOVA_COCOS   path to the nova-cocos package to build (defaults to the
#                       sibling `nova` checkout of this repository)
set -euo pipefail

PROG=$(basename "$0")

IMASDB=${1:-}
if [ -z "$IMASDB" ]; then
  echo "usage: $PROG <IMASDB>" >&2
  exit 2
fi
case $IMASDB in
  /*) ;;
  *) echo "$PROG: IMASDB must be an absolute path" >&2; exit 2 ;;
esac
IMASDB=${IMASDB%/}

SSH_HOST=${IMASDB_SSH_HOST:-jt-60sa}
SSH_ARGS=${IMASDB_SSH_ARGS:-}
MACHINE=${IMASDB_MACHINE:-jt-60sa}

REPO_ROOT=$(git -C "$(dirname -- "$0")" rev-parse --show-toplevel)
# nova-cocos is a sibling checkout of the repository, resolved from the main
# worktree so a build from a linked worktree still finds it.
MAIN_ROOT=$(git -C "$REPO_ROOT" worktree list --porcelain | awk 'NR==1{print $2}')
NOVA_COCOS=${IMASDB_NOVA_COCOS:-$(dirname "$MAIN_ROOT")/nova/packages/nova-cocos}

WORK=$(mktemp -d "${TMPDIR:-/tmp}/imasdb-install.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

say() { printf '\n== %s\n' "$*"; }

# --- 1. build the wheels ----------------------------------------------------
say "building wheels"
WHEELS=$WORK/wheels
mkdir -p "$WHEELS"
nice -n 19 uv build --out-dir "$WHEELS" "$REPO_ROOT/packages/imas-alambic"
nice -n 19 uv build --out-dir "$WHEELS" "$NOVA_COCOS"
ls -la "$WHEELS"

# --- 2. pull the map bundle -------------------------------------------------
# The registry, the authentication form and the artifact type have one owner,
# the maps CLI: it defaults to the package's latest stable tag, so this script
# names no version.
say "pulling the $MACHINE bundle"
BUNDLE=$WORK/bundle
mkdir -p "$BUNDLE"
export GH_TOKEN=${GH_TOKEN:-$(gh auth token)}
uv run --project "$MAIN_ROOT" imas-ambix maps pull "$MACHINE" --dest "$BUNDLE"
BUNDLE_VERSION=$(sed -n 's/.*"version"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$BUNDLE/bundle.json")
if [ -z "$BUNDLE_VERSION" ]; then
  echo "$PROG: $BUNDLE/bundle.json names no version" >&2
  exit 1
fi
echo "bundle version: $BUNDLE_VERSION"

# --- 3. relay the wheels and the bundle -------------------------------------
say "relaying to $SSH_HOST"
STAGE=/tmp/imasdb-stage.$$
# shellcheck disable=SC2086
# shellcheck disable=SC2086
ssh $SSH_ARGS "$SSH_HOST" "rm -rf '$STAGE' && mkdir -p '$STAGE'"
# shellcheck disable=SC2086
scp $SSH_ARGS -q "$WHEELS"/*.whl "$SSH_HOST:$STAGE/"
tar -C "$WORK" -cf "$WORK/bundle.tar" bundle
# shellcheck disable=SC2086
scp $SSH_ARGS -q "$WORK/bundle.tar" "$SSH_HOST:$STAGE/"
# shellcheck disable=SC2086
ssh $SSH_ARGS "$SSH_HOST" "nice -n 19 tar -C '$STAGE' -xf '$STAGE/bundle.tar'"

# --- 4-6. install on the server ---------------------------------------------
REMOTE=$WORK/remote.sh
cat > "$REMOTE" <<'REMOTE_SCRIPT'
#!/usr/bin/env bash
# Remote half of imasdb.sh, run on the analysis server under `nice -n 19`.
# argv: <IMASDB> <bundle-version> <stage-dir>
set -euo pipefail
IMASDB=$1
VERSION=$2
STAGE=$3

# The server's default CA path fails, so every outbound TLS client is pointed at
# the system bundle: SSL_CERT_FILE for Python and uv, GIT_SSL_CAINFO for git.
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export GIT_SSL_CAINFO=/etc/pki/tls/certs/ca-bundle.crt

TOOLS=$IMASDB/tools
BINDIR=$IMASDB/bin
export UV_TOOL_DIR=$TOOLS
export UV_TOOL_BIN_DIR=$TOOLS/bin
export UV_PYTHON_INSTALL_DIR=$TOOLS/python
IMASDB_PARENT=$(dirname -- "$IMASDB")

say() { printf '\n== %s\n' "$*"; }

say "laying down $IMASDB"
mkdir -p "$IMASDB" "$IMASDB_PARENT"
chmod 1755 "$IMASDB_PARENT"

say "installing a uv-managed CPython 3.12 into the folder"
# The interpreter has to live under the folder, so a 3.12 that happens to be
# reachable elsewhere must not be reused: it would tie the install to a path
# outside the shared folder that the next session or the node does not carry.
nice -n 19 uv python install --reinstall --no-cache --no-bin 3.12
PY312_BIN=$(ls -d "$TOOLS"/python/cpython-3.12.*/bin/python3.12 2>/dev/null | head -1)
if [ -z "$PY312_BIN" ]; then
  echo "imasdb.sh: no uv-managed CPython 3.12 under $TOOLS/python" >&2
  ls -la "$TOOLS/python" >&2 || true
  exit 1
fi
echo "managed interpreter: $PY312_BIN"

say "installing the tool environment"
IMAS_WHL=$(ls "$STAGE"/imas_alambic-*.whl)
COCOS_WHL=$(ls "$STAGE"/nova_cocos-*.whl)
rm -rf "$TOOLS/imas-alambic"
nice -n 19 uv tool install --force --python "$PY312_BIN" --no-cache "$IMAS_WHL" --with "$COCOS_WHL"

say "writing the launcher"
mkdir -p "$BINDIR"
cat > "$BINDIR/imas-alambic" <<'LAUNCHER'
#!/bin/sh
# imas-alambic facility launcher.
#
# Clears PYTHONPATH, because the login profile and `module load python/3.12` put
# foreign site-packages on it, and points the engine at this install's IMASDB
# folder, from which every setting resolves.
self=$(readlink -f -- "$0" 2>/dev/null || printf '%s' "$0")
here=$(CDPATH= cd -- "$(dirname -- "$self")" && pwd -P)
home=$(dirname -- "$here")
unset PYTHONPATH
IMAS_ALAMBIC_HOME=$home
export IMAS_ALAMBIC_HOME
# EDDB is on this machine, so the writer reads it in place rather than hopping
# back over ssh.  `local` selects the local transport in settings.py.
IMAS_ALAMBIC_EDDB_HOST=local
export IMAS_ALAMBIC_EDDB_HOST
exec "$home/tools/bin/imas-alambic" "$@"
LAUNCHER
chmod 0755 "$BINDIR/imas-alambic"
LAUNCHER=$BINDIR/imas-alambic

say "reading the layout names from imas-alambic config"
# `config` derives the map search path and the IDS root from IMAS_ALAMBIC_HOME,
# and reports every setting even when no bundle is reachable: an entry on the
# search path with no descriptor becomes the machine's and the cache's source
# rather than raising, so the layout names are readable before the layout
# exists.  The bundle is placed under its version directory just below.
CONF=$(env -u PYTHONPATH "$LAUNCHER" config)
printf '%s\n' "$CONF"
MAPS=$(printf '%s\n' "$CONF" | awk '/^maps:/{print $2}')
IDS=$(printf '%s\n' "$CONF" | awk '/^ids_root:/{print $2}')
MAPS_REL=${MAPS#"$IMASDB"/}
IDS_REL=${IDS#"$IMASDB"/}
MAPS_DIR=$(dirname -- "$MAPS_REL")
CURRENT=$(basename -- "$MAPS_REL")
echo "layout: maps=$MAPS_REL ids=$IDS_REL current=$CURRENT"

say "placing the bundle under the version directory"
mkdir -p "$IMASDB/$MAPS_DIR" "$IMASDB/$IDS_REL"
DEST=$IMASDB/$MAPS_DIR/$VERSION
rm -rf "$DEST"
mkdir -p "$DEST"
cp -a "$STAGE/bundle/." "$DEST/"
ln -sfn "$VERSION" "$IMASDB/$MAPS_DIR/$CURRENT"

say "opening the folder to other users"
# Others read, never write, and execute only what the owner may execute -- this
# is `chmod -R o=rX`.  Directories get others r-x so the tree is walkable;
# executable files, the launcher and the managed interpreter among them, get
# others r-x so a user outside the account's group can run the tool; every other
# file gets others r-- exactly.  Spelled with find so a symlink is left alone:
# its own mode is 0777 and carries no permissions, the target's govern.
find "$IMASDB" -type d -exec chmod o=rx {} +
find "$IMASDB" -type f -perm -u+x -exec chmod o=rx {} +
find "$IMASDB" -type f ! -perm -u+x -exec chmod o=r {} +

say "checks"
fail() { echo "FAIL: $*" >&2; exit 1; }

echo "-- account directory mode (want 1755)"
stat -c '%a %n' "$IMASDB_PARENT"
[ "$(stat -c %a "$IMASDB_PARENT")" = 1755 ] || fail "account directory is not 1755"

echo "-- directories under IMASDB lacking others r-x (want none)"
find "$IMASDB" -type d ! -perm -o+rx -print
[ -z "$(find "$IMASDB" -type d ! -perm -o+rx -print -quit)" ] || fail "a directory lacks others r-x"

echo "-- files under IMASDB lacking others r (want none)"
find "$IMASDB" -type f ! -perm -o+r -print
[ -z "$(find "$IMASDB" -type f ! -perm -o+r -print -quit)" ] || fail "a file lacks others r"

echo "-- executable files under IMASDB lacking others r-x (want none)"
find "$IMASDB" -type f -perm -u+x ! -perm -o+rx -print
[ -z "$(find "$IMASDB" -type f -perm -u+x ! -perm -o+rx -print -quit)" ] \
  || fail "an executable lacks others r-x"
echo "-- the launcher and the managed interpreter (want others r-x)"
stat -c '%A %a %n' "$BINDIR/imas-alambic" "$PY312_BIN"

echo "-- files and directories under IMASDB granting others write (want none)"
# A symlink's own mode is always 0777 and carries no permissions; the target's
# govern.  maps/current is such a link, so the write check is over real files
# and directories.
WRITABLE=$(find "$IMASDB" -not -type l -perm -o+w)
printf '%s\n' "$WRITABLE"
[ -z "$WRITABLE" ] || fail "a path grants others write"
echo "(maps/current is a symlink; its mode is 0777 and the target governs)"

echo "-- the tool environment's imas-alambic install source (want not editable)"
TOOL_PY=$TOOLS/imas-alambic/bin/python
env -u PYTHONPATH uv pip show imas-alambic --python "$TOOL_PY"
env -u PYTHONPATH uv pip show imas-alambic --python "$TOOL_PY" | grep -qi '^editable' \
  && fail "imas-alambic is an editable install"
test -d "$TOOLS/imas-alambic/lib/python3.12/site-packages/imas_alambic" \
  || fail "imas_alambic is not a wheel-installed package in the tool environment"

echo "-- imas-alambic --version, PYTHONPATH naming the python/3.12 module site-packages"
PY312=/home/ap/python/sys/3.12/lib64/python3.12/site-packages:/home/ap/python/sys/3.12/lib/python3.12/site-packages
PYTHONPATH="$PY312" "$LAUNCHER" --version

echo "-- imas-alambic --version, PYTHONPATH unset"
env -u PYTHONPATH "$LAUNCHER" --version

echo "-- uv pip check against the tool environment"
env -u PYTHONPATH uv pip check --python "$TOOL_PY"

echo "-- imas-alambic config with no IMAS_ALAMBIC_* variable set"
CONF=$(env -u PYTHONPATH "$LAUNCHER" config)
printf '%s\n' "$CONF"
MAPS=$(printf '%s\n' "$CONF" | awk '/^maps:/{print $2}')
IDS=$(printf '%s\n' "$CONF" | awk '/^ids_root:/{print $2}')
MACHINE=$(printf '%s\n' "$CONF" | awk '/^machine:/{print $2}')
[ "$MACHINE" = jt-60sa ] || fail "machine is '$MACHINE', not jt-60sa"
[ "$MAPS" = "$IMASDB/$MAPS_REL" ] || fail "maps did not resolve under IMAS_ALAMBIC_HOME"
[ "$IDS" = "$IMASDB/$IDS_REL" ] || fail "ids_root did not resolve under IMAS_ALAMBIC_HOME"

say "removing the older per-user install"
OLD=$HOME/imas-alambic
if [ -e "$OLD" ]; then
  mapfile -t OLD_ENTRIES < <(ls -A "$OLD")
  printf 'old install top-level: %s\n' "${OLD_ENTRIES[*]}"
  rm -rf -- "$OLD"
  for entry in "${OLD_ENTRIES[@]:-}"; do echo "removed $OLD/$entry"; done
  [ ! -e "$OLD" ] || fail "old install still present at $OLD"
  echo "old install gone"
else
  echo "no old install at $OLD"
fi

say "all checks passed"
REMOTE_SCRIPT
# shellcheck disable=SC2086
scp $SSH_ARGS -q "$REMOTE" "$SSH_HOST:$STAGE/imasdb-remote.sh"
# shellcheck disable=SC2086
ssh $SSH_ARGS "$SSH_HOST" "nice -n 19 bash '$STAGE/imasdb-remote.sh' '$IMASDB' '$BUNDLE_VERSION' '$STAGE'"
# shellcheck disable=SC2086
ssh $SSH_ARGS "$SSH_HOST" "rm -rf '$STAGE'"

say "done"