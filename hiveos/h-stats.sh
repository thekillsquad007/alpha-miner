#!/usr/bin/env bash
# Emit HiveOS miner stats JSON for alpha-miner logs.
# Parses [stats] hashrate lines and [submit] share counts.

[[ -z $SCRIPT_PATH ]] && SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[[ -f $SCRIPT_PATH/h-manifest.conf ]] && source "$SCRIPT_PATH/h-manifest.conf"
[[ -f $SCRIPT_PATH/miner.conf ]] && source "$SCRIPT_PATH/miner.conf"

log_file="${CUSTOM_LOG_BASENAME:-/var/log/miner/custom/alpha}.log"
: "${HSTATS_RAW_LINES:=3000}"

khs=0
hs_units="khs"
acc=0
rej=0
uptime_sec=0
ver="${CUSTOM_VERSION:-0}"
algo="blake3-an"

# Latest [stats] line: "[stats] 4.80 GH/s connected=yes"
if [[ -f "$log_file" ]]; then
  stats_line=$(tail -n "$HSTATS_RAW_LINES" "$log_file" 2>/dev/null | grep -aE '\[stats\]' | tail -n 1 || true)
  if [[ "$stats_line" =~ ([0-9.]+)[[:space:]]*([kKmMgGtT]?[Hh])/s ]]; then
    val="${BASH_REMATCH[1]}"
    unit="${BASH_REMATCH[2]}"
    # Convert to kH/s
    case "${unit^^}" in
      H|H/S) khs=$(awk -v v="$val" 'BEGIN{printf "%.3f", v/1000}') ;;
      KH|KHS|KH/S) khs="$val" ;;
      MH|MHS|MH/S) khs=$(awk -v v="$val" 'BEGIN{printf "%.3f", v*1000}') ;;
      GH|GHS|GH/S) khs=$(awk -v v="$val" 'BEGIN{printf "%.3f", v*1000000}') ;;
      TH|THS|TH/S) khs=$(awk -v v="$val" 'BEGIN{printf "%.3f", v*1000000000}') ;;
      *) khs="$val" ;;
    esac
  fi
  acc=$(grep -acE '\[submit\]|\[submit/fee\]' "$log_file" 2>/dev/null || echo 0)
  rej=$(grep -acE 'rejected|invalid share' "$log_file" 2>/dev/null || echo 0)
fi

# Uptime from miner process
miner_pid=$(pgrep -f "$SCRIPT_PATH/alpha " | head -1 || true)
if [[ -n "$miner_pid" ]]; then
  uptime_sec=$(ps -o etimes= -p "$miner_pid" 2>/dev/null | tr -d ' ' || echo 0)
fi
uptime_sec=${uptime_sec:-0}
acc=${acc:-0}
rej=${rej:-0}
khs=${khs:-0}

# Single total hashrate (no per-GPU breakdown from OpenCL miner logs yet)
hs_json="[$khs]"
temp_json="[]"
fan_json="[]"
bus_json="[]"

# Optional AMD/NVIDIA temps via sensors/rocm-smi/nvidia-smi (best-effort)
if command -v nvidia-smi >/dev/null 2>&1; then
  temps=()
  fans=()
  buses=()
  while IFS=',' read -r bus temp fan; do
    bus=$(echo "$bus" | awk -F: '{print $(NF-1)}')
    buses+=("$(printf '%d' "0x${bus}" 2>/dev/null || echo 0)")
    temps+=("$(echo "$temp" | tr -d ' ')")
    fans+=("$(echo "$fan" | tr -d ' ')")
  done < <(nvidia-smi --query-gpu=pci.bus_id,temperature.gpu,fan.speed --format=csv,noheader,nounits 2>/dev/null || true)
  if ((${#temps[@]} > 0)); then
    temp_json="[$(IFS=,; echo "${temps[*]}")]"
    fan_json="[$(IFS=,; echo "${fans[*]}")]"
    bus_json="[$(IFS=,; echo "${buses[*]}")]"
    # Split total hashrate evenly if multi-GPU (placeholder)
    n=${#temps[@]}
    if (( n > 1 )); then
      per=$(awk -v t="$khs" -v n="$n" 'BEGIN{printf "%.3f", t/n}')
      hs_json="["
      for ((i=0;i<n;i++)); do
        [[ $i -gt 0 ]] && hs_json+=","
        hs_json+="$per"
      done
      hs_json+="]"
    fi
  fi
fi

cat <<EOF
{"hs":$hs_json,"hs_units":"$hs_units","temp":$temp_json,"fan":$fan_json,"uptime":$uptime_sec,"ver":"$ver","ar":[$acc,$rej],"algo":"$algo","bus_numbers":$bus_json,"total_khs":$khs}
EOF
