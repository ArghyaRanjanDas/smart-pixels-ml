#!/usr/bin/env bash
# Resolve the LIVE pid of each O25 trainer by seed.
#
# Why this exists: the launcher backgrounded `env ... setsid nohup python ...`,
# so $! captured the env WRAPPER, which exits as soon as setsid forks. Every pid
# recorded that way is dead within a second while the trainer runs on with
# ppid=1. Killing a stale list kills nothing and looks like "already gone".
# Always resolve pids here, at the moment you need them -- never from a note.
#
# Prints "seed pid rss_gb etime" per line. Kill by the EXACT pid printed here;
# never pkill -f the script name (it also matches your own calling shell).
ps -eo pid,etime,rss,args | grep "[r]un_simplerouter_mdmm_discovery" | grep -v grep |
awk '{
  seed="?"; for (i=1;i<=NF;i++) if ($i=="--seeds") seed=$(i+1);
  if (seed ~ /^[0-9]+$/) printf "%s %s %.2f %s\n", seed, $1, $3/1048576, $2
}' | sort -n
