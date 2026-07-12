#!/usr/bin/env bash
# Failover supervisor for alpha-miner (HiveOS).
# Long-lived process — argv must NOT contain h-run.sh.

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f "$SCRIPT_PATH/h-manifest.conf" ]] && source "$SCRIPT_PATH/h-manifest.conf"
cd "$SCRIPT_PATH"

MINER_BIN="$SCRIPT_PATH/alpha"
LOG="${CUSTOM_LOG_BASENAME:-/var/log/miner/custom/alpha}.log"

if [[ ! -x "$MINER_BIN" ]]; then
  echo "ERROR: Binary not found or not executable: $MINER_BIN"
  exit 1
fi

[[ -f "$SCRIPT_PATH/miner.conf" ]] && source "$SCRIPT_PATH/miner.conf"

if [[ ${#POOLS[@]} -eq 0 || ${#ALPHA_BASE_ARGS[@]} -eq 0 ]]; then
  echo "ERROR: miner.conf missing POOLS[] or ALPHA_BASE_ARGS[] (run h-config.sh)"
  exit 1
fi

: "${FAILOVER_GRACE_SEC:=120}"
: "${FAILOVER_DEAD_SEC:=300}"
: "${FAILOVER_RETURN_SEC:=1800}"
: "${FAILOVER_POLL_SEC:=15}"

echo "========================================"
echo "${CUSTOM_NAME:-alpha} v${CUSTOM_VERSION:-?}  (pid $$)"
echo "Pools (${#POOLS[@]}): ${POOLS[*]}"
echo "Base args: ${ALPHA_BASE_ARGS[*]}"
echo "========================================"

miner_pid=""

stop_miner() {
  [[ -z "$miner_pid" ]] && return
  kill -INT "$miner_pid" 2>/dev/null
  for _ in $(seq 1 12); do
    kill -0 "$miner_pid" 2>/dev/null || break
    sleep 1
  done
  kill -KILL "$miner_pid" 2>/dev/null
  wait "$miner_pid" 2>/dev/null
  miner_pid=""
}

on_stop() {
  trap - INT TERM
  echo "$(date -u +%FT%TZ) [wrapper] stop"
  stop_miner
  exit 0
}
trap on_stop INT TERM

now() { date +%s; }

# Last share / submit timestamp from our log format.
last_share_epoch() {
  local ts
  ts=$(tail -n "${FAILOVER_SHARE_SCAN_LINES:-4000}" "$LOG" 2>/dev/null \
    | grep -aE '\[submit\]|\[submit/fee\]|share nonce=' \
    | tail -n 1 \
    | grep -oE '^[0-9]{4}-[0-9]{2}-[0-9]{2}[T ][0-9]{2}:[0-9]{2}:[0-9]{2}' || true)
  if [[ -n "$ts" ]]; then
    date -d "${ts/ /T}Z" +%s 2>/dev/null || date -d "$ts" +%s 2>/dev/null || echo 0
  else
    # Fallback: file mtime if recent activity lines without ISO prefix
    if grep -aqE '\[submit\]|share nonce=' "$LOG" 2>/dev/null; then
      now
    else
      echo 0
    fi
  fi
}

n=${#POOLS[@]}
idx=0
primary_retry_at=$(( $(now) + FAILOVER_RETURN_SEC ))

while :; do
  pool="${POOLS[$idx]}"
  echo "$(date -u +%FT%TZ) [wrapper] launching on pool[$idx]=$pool"

  # alpha-miner expects -o host:port
  pool_arg="$pool"
  pool_arg="${pool_arg#stratum+tcp://}"
  pool_arg="${pool_arg#stratum+tcps://}"
  pool_arg="${pool_arg#tcp://}"

  "$MINER_BIN" -o "$pool_arg" "${ALPHA_BASE_ARGS[@]}" &
  miner_pid=$!
  started=$(now)

  while kill -0 "$miner_pid" 2>/dev/null; do
    sleep "$FAILOVER_POLL_SEC"
    elapsed=$(( $(now) - started ))
    if (( elapsed < FAILOVER_GRACE_SEC )); then
      continue
    fi
    last=$(last_share_epoch)
    if (( last > 0 )); then
      silent=$(( $(now) - last ))
    else
      silent=$elapsed
    fi
    if (( silent >= FAILOVER_DEAD_SEC )); then
      echo "$(date -u +%FT%TZ) [wrapper] pool[$idx] silent ${silent}s — failover"
      stop_miner
      break
    fi
    # Prefer primary after return interval
    if (( idx != 0 && $(now) >= primary_retry_at )); then
      echo "$(date -u +%FT%TZ) [wrapper] returning to primary pool"
      stop_miner
      idx=0
      primary_retry_at=$(( $(now) + FAILOVER_RETURN_SEC ))
      break
    fi
  done

  # Miner exited on its own
  wait "$miner_pid" 2>/dev/null || true
  miner_pid=""

  if (( n > 1 )); then
    idx=$(( (idx + 1) % n ))
    if (( idx == 0 )); then
      primary_retry_at=$(( $(now) + FAILOVER_RETURN_SEC ))
    fi
  fi
  sleep 3
done
