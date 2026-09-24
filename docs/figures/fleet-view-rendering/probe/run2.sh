set -u
RUN=/tmp/zjprobe-run; ZJ=/home/ITER/mcintos/.local/bin/zellij
SOCK=$RUN/sock2; mkdir -p $SOCK
scen() {  # sb kind
  local sb=$1 kind=$2 sess="zj2-${2}-${1}" sdir="$SOCK/$1-$2"
  mkdir -p $sdir
  sed -e "s/^scroll_buffer_size .*/scroll_buffer_size $sb/" -e "s/^scrollback_lines_to_serialize .*/scrollback_lines_to_serialize $sb/" \
      /home/ITER/mcintos/.config/zellij/config.kdl > $RUN/cfg-$sb.kdl
  cat > $RUN/lay-$kind.kdl <<EOF
layout {
    pane command="$RUN/fill_$kind.sh"
}
EOF
  ZELLIJ_SOCKET_DIR=$sdir $ZJ -s $sess -l $RUN/lay-$kind.kdl attach --create-background >/dev/null 2>&1
  sleep 8
  echo "### $kind sb=$sb"
  ZELLIJ_SOCKET_DIR=$sdir /usr/bin/python3 $RUN/probe2.py $sdir $sess 120 40 attach 2>&1
  ZELLIJ_SOCKET_DIR=$sdir /usr/bin/python3 $RUN/probe2.py $sdir $sess 100 40 attach_resize 2>&1
  ZELLIJ_SOCKET_DIR=$sdir $ZJ kill-session $sess >/dev/null 2>&1
  ZELLIJ_SOCKET_DIR=$sdir $ZJ delete-session $sess >/dev/null 2>&1
  sleep 1; rm -rf $sdir
}
scen 10000 cc
scen 50000 cc
scen 50000 shell
echo DONE
