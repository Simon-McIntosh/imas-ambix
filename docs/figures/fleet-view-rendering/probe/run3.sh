set -u
RUN=/tmp/zjprobe-run; ZJ=/home/ITER/mcintos/.local/bin/zellij
sdir=$RUN/sock3; mkdir -p $sdir
sed -e "s/^scroll_buffer_size .*/scroll_buffer_size 50000/" /home/ITER/mcintos/.config/zellij/config.kdl > $RUN/cfg3.kdl
printf "layout {\n    pane command=\"%s/fill_cc.sh\"\n}\n" $RUN > $RUN/lay3.kdl
ZELLIJ_SOCKET_DIR=$sdir $ZJ -s zj3 -l $RUN/lay3.kdl attach --create-background >/dev/null 2>&1
sleep 8
ZELLIJ_SOCKET_DIR=$sdir /usr/bin/python3 $RUN/probe3.py $sdir zj3 2>&1
ZELLIJ_SOCKET_DIR=$sdir $ZJ kill-session zj3 >/dev/null 2>&1
ZELLIJ_SOCKET_DIR=$sdir $ZJ delete-session zj3 >/dev/null 2>&1
rm -rf $sdir
