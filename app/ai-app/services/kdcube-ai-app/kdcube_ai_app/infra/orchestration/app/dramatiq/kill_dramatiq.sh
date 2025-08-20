#!/usr/bin/env bash
# kill_dramatiq.sh
# Kill IntelliJ-launched Dramatiq runs + all their children.

set -euo pipefail

# DEFAULT PATTERN — no quotes inside the variable!
# Matches:
#  - .../pydevd.py --file /.../dramatiq_app_cli.py
#  - python -c '... "file": "/.../dramatiq_app_cli.py" ...'
PATTERN="${PATTERN:-dramatiq_app_cli\.py|pydevd\.py.*--file .*dramatiq_app_cli\.py}"
TIMEOUT="${TIMEOUT:-5}"
SIGNAL="${SIGNAL:-TERM}"   # TERM or INT
DRY_RUN="${DRY_RUN:-0}"    # DRY_RUN=1 to just list

get_children() {
  local pid="$1"
  local kids
  kids="$(pgrep -P "$pid" || true)"
  for c in $kids; do
    echo "$c"
    get_children "$c"
  done
}

kill_tree() {
  local root="$1"
  local all=()
  while IFS= read -r c; do all+=("$c"); done < <(get_children "$root")

  echo "Targets (root first shown last): ${all[*]} $root"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi

  # Gentle stop
  if ((${#all[@]})); then kill -s "$SIGNAL" "${all[@]}" 2>/dev/null || true; fi
  kill -s "$SIGNAL" "$root" 2>/dev/null || true

  local end=$((SECONDS + TIMEOUT))
  while [ $SECONDS -lt $end ]; do
    sleep 1
    local alive=""
    for p in "${all[@]}" "$root"; do
      if kill -0 "$p" 2>/dev/null; then alive=1; break; fi
    done
    [ -z "$alive" ] && return 0
  done

  echo "Force killing with SIGKILL…"
  for p in "${all[@]}" "$root"; do kill -9 "$p" 2>/dev/null || true; done
}

echo "Searching with pgrep -f for: $PATTERN"
PIDS="$(pgrep -f "$PATTERN" || true)"

# Fallback if pgrep misses (macOS sometimes truncates for pgrep but not ps)
if [ -z "$PIDS" ]; then
  echo "pgrep found nothing; trying ps fallback…"
  PIDS="$(ps auxww | grep -E "$PATTERN" | grep -v grep | awk '{print $2}')"
fi

if [ -z "$PIDS" ]; then
  echo "No matching processes found."
  exit 0
fi

echo "Matched processes:"
for p in $PIDS; do
  ps -o pid,ppid,pgid,command -p "$p"
done

for p in $PIDS; do
  echo "----"
  kill_tree "$p"
done

echo "Verifying…"
LEFT="$(pgrep -f "$PATTERN" || true)"
if [ -z "$LEFT" ]; then
  echo "All dramatiq debug runs stopped."
else
  echo "Still running (check manually): $LEFT"
  ps -o pid,ppid,pgid,command -p $LEFT
  exit 1
fi
