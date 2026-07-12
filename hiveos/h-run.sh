#!/usr/bin/env bash
# HiveOS runs this in screen — do NOT nest another screen.
# Must exec a differently-named supervisor so restarts are not blocked by
# "already running" grep of h-run.sh.

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f "$SCRIPT_PATH/h-manifest.conf" ]] && source "$SCRIPT_PATH/h-manifest.conf"

LOG="${CUSTOM_LOG_BASENAME:-/var/log/miner/custom/alpha}.log"
mkdir -p "$(dirname "$LOG")" 2>/dev/null

exec > >(exec tee -a "$LOG") 2>&1

SUP="$SCRIPT_PATH/alpha-supervise.sh"
if [[ ! -f "$SUP" ]]; then
  echo "FATAL: supervisor not found: $SUP"
  exit 1
fi
exec bash "$SUP"
echo "FATAL: failed to exec supervisor"
exit 1
