#!/usr/bin/env bash
# O25 watchdog: restart a chain whose shell died, until every seed reaches TOTAL.
#
# Why: each seed is driven by a chain shell (ppid 1) that survives chunk exits.
# Nothing supervises THAT. If it dies -- OOM kill, node blip, a stray signal --
# the seed just stops, and on an overnight run nobody notices until morning.
#
# Safety: never starts a second chain for a seed that already has one (that would
# put two processes on one output dir and corrupt history.csv / the checkpoints).
# Bounded restarts per seed so a seed that dies instantly cannot spin forever.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
TOTAL=${1:-10000}; CHUNK=${2:-100}; MAXR=${3:-20}; PERIOD=${4:-120}
SEEDS="50142 50242 50342 50442 50542 50642 50742 50842"
LOG=$REPO/runs/o25_qconv2d/watchdog.log
declare -A R; for s in $SEEDS; do R[$s]=0; done

ep_of () { local h; h=$(ls $REPO/runs/o25_qconv2d/s$1/seed_*/history.csv 2>/dev/null|head -1)
           [ -n "$h" ] && echo $(( $(wc -l < "$h") - 1 )) || echo 0; }
chain_of () { ps -eo pid,args | awk -v s="$1" '/[o]25_chunk_run.sh/ && $0 ~ (" " s " ") {print $1; exit}'; }

echo "[wd $(date +%F\ %T)] up: total=$TOTAL chunk=$CHUNK maxrestart=$MAXR period=${PERIOD}s" >> "$LOG"
while true; do
  done_n=0
  for s in $SEEDS; do
    ep=$(ep_of $s)
    if [ "$ep" -ge "$TOTAL" ]; then done_n=$((done_n+1)); continue; fi
    if [ -z "$(chain_of $s)" ]; then
      if [ "${R[$s]}" -ge "$MAXR" ]; then
        echo "[wd $(date +%T)] s$s chain gone at ep $ep -- restart budget ${R[$s]}/$MAXR EXHAUSTED, leaving stopped" >> "$LOG"
        done_n=$((done_n+1)); continue
      fi
      R[$s]=$(( ${R[$s]} + 1 ))
      echo "[wd $(date +%T)] s$s chain gone at ep $ep -- restarting (${R[$s]}/$MAXR)" >> "$LOG"
      setsid nohup bash $REPO/scripts/o25_chunk_run.sh $s $TOTAL $CHUNK \
        >> $REPO/runs/o25_qconv2d/s$s/chain_outer.log 2>&1 &
      sleep 10
    fi
  done
  [ "$done_n" -ge 8 ] && { echo "[wd $(date +%T)] all seeds finished or exhausted -- exiting" >> "$LOG"; exit 0; }
  sleep "$PERIOD"
done
