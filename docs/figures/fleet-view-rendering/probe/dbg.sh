set -u
RUN=/tmp/zjprobe-run; ZJ=/home/ITER/mcintos/.local/bin/zellij
sdir=$RUN/dbg-sock; mkdir -p $sdir
cat > $RUN/layout-dbg.kdl <<EOF
layout {
    pane command="$RUN/fill_cc.sh"
}
EOF
ZELLIJ_SOCKET_DIR=$sdir $ZJ -s zjdbg -l $RUN/layout-dbg.kdl attach --create-background >/dev/null 2>&1
sleep 5
ZELLIJ_SOCKET_DIR=$sdir $ZJ ls 2>&1
ZELLIJ_SOCKET_DIR=$sdir /usr/bin/python3 $RUN/dbg.py $sdir zjdbg 2>&1
ZELLIJ_SOCKET_DIR=$sdir /usr/bin/python3 $RUN/dbg.py $sdir zjdbg 2>&1
echo "=== server log tail ==="
tail -25 /tmp/zellij-39486/zellij-log/zellij.log 2>&1
ZELLIJ_SOCKET_DIR=$sdir $ZJ kill-session zjdbg 2>&1
ZELLIJ_SOCKET_DIR=$sdir $ZJ delete-session zjdbg 2>&1
rm -rf $sdir
