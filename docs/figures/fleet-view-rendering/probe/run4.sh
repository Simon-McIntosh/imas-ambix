set -u
RUN=/tmp/zjprobe-run; ZJ=/home/ITER/mcintos/.local/bin/zellij
SOCK=$RUN/sock4; mkdir -p $SOCK
scen() {
  local sb=$1 kind=$2 sess="zj4-$2-$1" sdir="$SOCK/$1-$2"
  mkdir -p $sdir
  sed -e "s/^scroll_buffer_size .*/scroll_buffer_size $sb/" -e "s/^scrollback_lines_to_serialize .*/scrollback_lines_to_serialize $sb/" \
      /home/ITER/mcintos/.config/zellij/config.kdl > $RUN/cfg4-$sb.kdl
  export ZELLIJ_CONFIG_FILE=$RUN/cfg4-$sb.kdl ZELLIJ_SOCKET_DIR=$sdir
  $ZJ attach --create-background $sess >/dev/null 2>&1
  sleep 2
  ZELLIJ_SESSION_NAME=$sess $ZJ run --name filler -- /usr/bin/env bash $RUN/fill_$kind.sh >/dev/null 2>&1
  sleep 8
  echo "### $kind sb=$sb  ls:"; $ZJ ls 2>&1 | sed "s/\x1b\[[0-9;]*m//g"
  ZELLIJ_SESSION_NAME= $ZJ --session $sess action list-panes >/dev/null 2>&1
  /usr/bin/python3 $RUN/probe3.py $sdir $sess 2>&1
  $ZJ kill-session $sess >/dev/null 2>&1; $ZJ delete-session $sess >/dev/null 2>&1
  sleep 1; rm -rf $sdir; unset ZELLIJ_CONFIG_FILE ZELLIJ_SOCKET_DIR
}
scen 10000 cc
scen 50000 cc
scen 50000 shell
echo DONE
