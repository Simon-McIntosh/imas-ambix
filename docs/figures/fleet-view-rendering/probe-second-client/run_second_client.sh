set -u
# Throwaway-session probe on the fleet node: a second client attaching at a
# different width to (1) a session with four tabs, (2) a one-tab control, and
# (3) a session holding a live claude pane resized between two widths.
#
# Every session is private to a socket directory under /tmp, created and
# destroyed by name. The fleet sessions are never touched.

WT=/home/ITER/mcintos/Code/.reckon-worktrees/imas-ambix-0405601943d2/ambix-fleet-view/fleet-view-unseen-surfaces
PROBE=$WT/docs/figures/fleet-view-rendering/probe-second-client
GEN=$WT/docs/figures/fleet-view-rendering/probe/gen_stream.py
OUT=$WT/docs/figures/fleet-view-rendering/evidence-second-client
RUN=/tmp/zj-second-client
SOCK=$RUN/sock
ZJ=/home/ITER/mcintos/.local/bin/zellij
CONF=/home/ITER/mcintos/.config/zellij/config.kdl
LOGF=/tmp/zellij-$(id -u)/zellij-log/zellij.log
LOCAL_ORIGIN=http://98dci4-gpu-0003:18802

mkdir -p "$RUN" "$SOCK" "$OUT"
export ZELLIJ_CONFIG_FILE="$CONF" ZELLIJ_SOCKET_DIR="$SOCK"

stallcount() { grep -c 'did not complete within' "$LOGF" 2>/dev/null || echo 0; }

if [ ! -s "$RUN/cc-60000.bin" ]; then
  timeout 120 script -q -c \
    "ANTHROPIC_BASE_URL=$LOCAL_ORIGIN ANTHROPIC_API_KEY=x /home/ITER/mcintos/.local/bin/claude -p 'list twelve short facts about the number seven, one per line, numbered'" \
    "$RUN/cc.raw" < /dev/null >/dev/null 2>&1
  /usr/bin/python3 "$GEN" 60000 "$RUN/cc.raw" > "$RUN/cc-60000.bin" 2> "$RUN/gen.log"
fi

cat > "$RUN/fill_cc.sh" <<'EOF'
#!/bin/bash
cat /tmp/zj-second-client/cc-60000.bin
exec sleep 900
EOF

cat > "$RUN/fill_shell.sh" <<'EOF'
#!/bin/bash
printf 'plain shell pane, no scrollback payload\n'
exec sleep 900
EOF

chmod +x "$RUN/fill_cc.sh" "$RUN/fill_shell.sh"

# The pane runs the local lane with no custom API-key env var: a key value in
# the environment makes claude show its "use this API key?" prompt, which then
# sits over the TUI the resize is meant to measure.
cat > "$RUN/live_app.sh" <<EOF
#!/bin/bash
cd /home/ITER/mcintos/Code/imas-codex
exec env -u ANTHROPIC_API_KEY ANTHROPIC_BASE_URL=$LOCAL_ORIGIN \\
  ANTHROPIC_MODEL=deepseek-v4.1-flash \\
  CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1 \\
  /home/ITER/mcintos/.local/bin/claude
EOF
chmod +x "$RUN/live_app.sh"

scenario_multi() {
  local tabs="$1" ordering="$2"
  local sess="zj-second-${tabs}tab-${ordering}"
  local before after
  "$ZJ" attach --create-background "$sess" >/dev/null 2>&1
  sleep 2
  ZELLIJ_SESSION_NAME="$sess" "$ZJ" run --name filler -- /usr/bin/env bash "$RUN/fill_cc.sh" >/dev/null 2>&1
  sleep 3
  echo "### session=$sess tabs=$tabs ordering=$ordering"
  before=$(stallcount)
  ZELLIJ_SESSION_NAME= /usr/bin/python3 "$PROBE/multitab_client.py" "$SOCK" "$sess" multi \
      "$OUT/$sess.bin" "$OUT/$sess.marks.json" "$tabs" "$ordering" 2>&1
  after=$(stallcount)
  echo "RESULT tabs=$tabs ordering=$ordering stall_before=$before stall_after=$after delta=$((after - before))"
  "$ZJ" kill-session "$sess" >/dev/null 2>&1
  "$ZJ" delete-session "$sess" >/dev/null 2>&1
  sleep 1
}

scenario_live() {
  local sess="zj-second-live"
  local before after
  "$ZJ" attach --create-background "$sess" >/dev/null 2>&1
  sleep 2
  ZELLIJ_SESSION_NAME="$sess" "$ZJ" run --name cli -- /usr/bin/env bash "$RUN/live_app.sh" >/dev/null 2>&1
  sleep 12
  echo "### session=$sess live claude pane"
  before=$(stallcount)
  ZELLIJ_SESSION_NAME= /usr/bin/python3 "$PROBE/multitab_client.py" "$SOCK" "$sess" live \
      "$OUT/$sess.bin" "$OUT/$sess.marks.json" 2>&1
  after=$(stallcount)
  echo "RESULT live stall_before=$before stall_after=$after delta=$((after - before))"
  "$ZJ" kill-session "$sess" >/dev/null 2>&1
  "$ZJ" delete-session "$sess" >/dev/null 2>&1
  sleep 1
}

WANT="${1:-all}"
case "$WANT" in
  live) scenario_live ;;
  multi) scenario_multi 4 pre; scenario_multi 4 attached; scenario_multi 1 attached ;;
  *) scenario_multi 4 pre; scenario_multi 4 attached; scenario_multi 1 attached; scenario_live ;;
esac

echo "### cleanup"
"$ZJ" ls 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
echo "PROBE_DONE"