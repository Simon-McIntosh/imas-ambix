#!/bin/bash
# Throwaway probe on the fleet node. Never touches the live fleet socket dir.
set -u
RUN=/tmp/zjprobe-run
ZJ=/home/ITER/mcintos/.local/bin/zellij
FLEET_CONFIG=/home/ITER/mcintos/.config/zellij/config.kdl
SOCK=$RUN/sock
export PATH=/home/ITER/mcintos/.local/bin:$PATH

mkdir -p "$RUN" "$SOCK"

# --- pane payloads -----------------------------------------------------------
/usr/bin/python3 "$RUN/gen_stream.py" 60000 "$RUN/cc.raw" > "$RUN/cc-60000.bin" 2> "$RUN/gen.log"

cat > "$RUN/fill_cc.sh" <<'EOF'
#!/bin/bash
cat /tmp/zjprobe-run/cc-60000.bin
exec sleep 900
EOF
cat > "$RUN/fill_shell.sh" <<'EOF'
#!/bin/bash
printf 'plain shell pane, no scrollback payload\n'
exec sleep 900
EOF
chmod +x "$RUN/fill_cc.sh" "$RUN/fill_shell.sh"

layout_for() {
  cat > "$RUN/layout-$1.kdl" <<EOF
layout {
    pane command="$RUN/fill_$1.sh"
}
EOF
  echo "$RUN/layout-$1.kdl"
}

# --- per-scrollback config ---------------------------------------------------
mk_config() {
  sed -e "s/^scroll_buffer_size .*/scroll_buffer_size $1/" \
      -e "s/^scrollback_lines_to_serialize .*/scrollback_lines_to_serialize $1/" \
      "$FLEET_CONFIG" > "$RUN/config-$1.kdl"
  echo "$RUN/config-$1.kdl"
}

scenario() {  # $1 scroll_buffer  $2 pane-kind
  local sb="$1" kind="$2"
  local sess="zjprobe-${kind}-${sb}"
  local cfg; cfg=$(mk_config "$sb")
  local lay; lay=$(layout_for "$kind")
  local sdir="$SOCK/$sb-$kind"
  mkdir -p "$sdir"
  rm -f "$RUN/log-$kind-$sb.log"

  ZELLIJ_SOCKET_DIR="$sdir" ZELLIJ_CONFIG_FILE="$cfg" ZELLIJ_LOG_PATH="$RUN/zjlog-$kind-$sb.log" \
    "$ZJ" -s "$sess" -l "$lay" attach --create-background > "$RUN/create-$kind-$sb.out" 2>&1
  sleep 4     # let the pane fill its 60000-line scrollback
  echo "### scenario pane=$kind scroll_buffer=$sb" | tee -a "$RUN/log-$kind-$sb.log"

  for spec in "120 40 attach" "100 40 attach" "100 40 attach_resize"; do
    set -- $spec
    ZELLIJ=1 ZELLIJ_SOCKET_DIR="$sdir" ZELLIJ_CONFIG_FILE="$cfg" ZELLIJ_LOG_PATH="$RUN/zjlog-$kind-$sb.log" \
      /usr/bin/python3 "$RUN/probe.py" "$sdir" "$sess" "$1" "$2" "$3" \
      2>>"$RUN/log-$kind-$sb.log" | tee -a "$RUN/log-$kind-$sb.log"
    sleep 1
  done

  ZELLIJ_SOCKET_DIR="$sdir" "$ZJ" kill-session "$sess" >/dev/null 2>&1
  ZELLIJ_SOCKET_DIR="$sdir" "$ZJ" delete-session "$sess" >/dev/null 2>&1
  sleep 1
  rm -rf "$sdir"
}

scenario 10000 cc
scenario 50000 cc
scenario 50000 shell

echo "=== DONE ==="
echo "--- server-side stall errors per scenario ---"
for f in "$RUN"/zjlog-*.log; do echo "$f: $(grep -c 'did not complete' "$f" 2>/dev/null)"; done