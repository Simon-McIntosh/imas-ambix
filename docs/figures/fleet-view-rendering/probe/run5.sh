set -u
RUN=/tmp/zjprobe-run
ZJ=/home/ITER/mcintos/.local/bin/zellij
SOCK=$RUN/sock5
LOGF=/tmp/zellij-$(id -u)/zellij-log/zellij.log
mkdir -p "$RUN" "$SOCK" "$RUN/evidence"

/usr/bin/python3 "$RUN/gen_stream.py" 60000 "$RUN/cc.raw" > "$RUN/cc-60000.bin" 2>"$RUN/gen.log"

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

scen() {  # $1 scroll_buffer  $2 pane-kind
  local sb="$1"
  local kind="$2"
  local sess="zj5-${kind}-${sb}"
  local sdir="$SOCK/${sb}-${kind}"
  mkdir -p "$sdir"
  sed -e "s/^scroll_buffer_size .*/scroll_buffer_size $sb/" \
      -e "s/^scrollback_lines_to_serialize .*/scrollback_lines_to_serialize $sb/" \
      /home/ITER/mcintos/.config/zellij/config.kdl > "$RUN/cfg5-$sb.kdl"
  export ZELLIJ_CONFIG_FILE="$RUN/cfg5-$sb.kdl" ZELLIJ_SOCKET_DIR="$sdir"
  "$ZJ" attach --create-background "$sess" > "$RUN/evidence/log-$kind-$sb.create" 2>&1
  sleep 2
  ZELLIJ_SESSION_NAME="$sess" "$ZJ" run --name filler -- /usr/bin/env bash "$RUN/fill_$kind.sh" >/dev/null 2>&1
  sleep 10
  echo "### pane=$kind scroll_buffer=$sb"
  "$ZJ" ls 2>&1 | sed "s/\x1b\[[0-9;]*m//g" | grep -v EXITED

  local before after
  before=$(grep -c 'did not complete' "$LOGF" 2>/dev/null || echo 0)
  ZELLIJ_SESSION_NAME= /usr/bin/python3 "$RUN/probe4.py" "$sdir" "$sess" 100 40 timed \
      > "$RUN/evidence/log-$kind-$sb.timed" 2>&1
  cat "$RUN/evidence/log-$kind-$sb.timed"
  after=$(grep -c 'did not complete' "$LOGF" 2>/dev/null || echo 0)
  echo "stall_log_delta=$((after - before))"

  if [ "$kind" = cc ] && [ "$sb" = 50000 ]; then
    ZELLIJ_SESSION_NAME= /usr/bin/python3 "$RUN/probe4.py" "$sdir" "$sess" 100 40 twoclient \
        "$RUN/evidence/first-client-diffsize.bin" > "$RUN/evidence/log-cc-50000.twoclient" 2>&1
    cat "$RUN/evidence/log-cc-50000.twoclient"
    after=$(grep -c 'did not complete' "$LOGF" 2>/dev/null || echo 0)
    echo "stall_log_delta_after_twoclient=$((after - before))"
  fi

  "$ZJ" kill-session "$sess" >/dev/null 2>&1
  "$ZJ" delete-session "$sess" >/dev/null 2>&1
  sleep 1
  unset ZELLIJ_CONFIG_FILE ZELLIJ_SOCKET_DIR
  rm -rf "$sdir"
}

scen 10000 cc
scen 50000 cc
scen 50000 shell
echo DONE