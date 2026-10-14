#!/usr/bin/env bash
# O25 status: chain shells, live trainers, epochs, memory. Session-independent.
#
# Two process layers: a chain shell per seed (ppid 1, survives every chunk) and
# the python trainer it is currently running (a NEW pid each chunk).  Resolve
# both here at the moment you need them -- pids recorded at launch are wrong:
# $! captures the env/setsid wrapper, which exits immediately.
#
# NEVER pkill -f these patterns: your own calling shell matches them.
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
printf "%-8s %-9s %-9s %-8s %-7s %s\n" SEED CHAIN_PID TRAIN_PID RSS_GB EPOCHS STATE
for S in 50142 50242 50342 50442 50542 50642 50742 50842; do
  cp=$(ps -eo pid,args | awk -v s="$S" '/[o]25_chunk_run.sh/ && $0 ~ (" " s " ") {print $1; exit}')
  tp=$(ps -eo pid,args | awk -v s="$S" '/[r]un_simplerouter_mdmm_discovery/ && /--seeds/ {for(i=1;i<=NF;i++) if($i=="--seeds" && $(i+1)==s) {print $1; exit}}')
  rss=$([ -n "$tp" ] && ps -o rss= -p "$tp" 2>/dev/null | awk '{printf "%.2f",$1/1048576}' || echo "-")
  h=$(ls $REPO/runs/o25_qconv2d/s$S/seed_*/history.csv 2>/dev/null | head -1)
  ep=$([ -n "$h" ] && echo $(( $(wc -l < "$h") - 1 )) || echo 0)
  st="RUNNING"; [ -z "$cp" ] && st="CHAIN-GONE"; [ -n "$cp" ] && [ -z "$tp" ] && st="between-chunks"
  grep -q "DONE at epoch" $REPO/runs/o25_qconv2d/s$S/chain.log 2>/dev/null && st="DONE"
  printf "%-8s %-9s %-9s %-8s %-7s %s\n" "$S" "${cp:--}" "${tp:--}" "$rss" "$ep" "$st"
done
echo
awk '/^(cache|rss) /{printf "cgroup %-6s %7.1f GB\n",$1,$2/1073741824}' /sys/fs/cgroup/memory/memory.stat
echo "mine total: $(ps -eo rss,args | grep '[r]un_simplerouter' | awk '{t+=$1} END{printf "%.1f GB",t/1048576}')"
