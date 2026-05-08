#!/bin/bash
# Pure-I/O Peformance: Random read bandwidth — BaM vs CoPilotIO
# Sweeps SM count (12, 24, 48, 96) at 4KB random read.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BIN_BAM="$SCRIPT_DIR/build/nvm-bam-read-bw"
BIN_COP="$SCRIPT_DIR/build/nvm-copilotio-read-bw"
LIB_BAM="$SCRIPT_DIR/../bam/build/lib"
LIB_COP="$SCRIPT_DIR/../copilot-io/build/lib"
GDRCOPY_LIB="$SCRIPT_DIR/../gdrcopy/src"

SM_LIST="12 24 48 96"
IO=4096
CTRL="/dev/libnvm0"
QPS=16
BENCH_IOS="--bench-ios 10"

for bin in "$BIN_BAM" "$BIN_COP"; do
    if [ ! -f "$bin" ]; then
        echo "Error: $bin not found. Build first:"
        echo "  cd microbenchmark && mkdir -p build && cd build && cmake .. && make"
        exit 1
    fi
done

get_async_batch() {
    local sms="$1"
    case "$sms" in
        24) echo 8 ;;
        *)  echo 1 ;;
    esac
}

echo "=== SSD: $CTRL, $QPS QPs ==="
echo ""

# --- BaM baseline (SQ GPU + CQ GPU) ---
OUT_BAM="$SCRIPT_DIR/results_bam_read_bw.csv"
echo "sms,bandwidth_gibs,iops_m" > "$OUT_BAM"
echo "=== BaM Baseline (32 warps/SM, $QPS QPs) ==="
for sms in $SM_LIST; do
    echo ">>> BaM SMs=$sms"
    out=$(LD_LIBRARY_PATH="$LIB_BAM${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
          "$BIN_BAM" --ctrl "$CTRL" --sms "$sms" --warps 32 --qps "$QPS" --qd 1024 --io-size "$IO" $BENCH_IOS 2>&1)
    bw=$(echo "$out" | grep "Bandwidth:" | awk '{print $2}')
    iops=$(echo "$out" | grep "IOPS:" | awk '{print $2}')
    echo "$sms,$bw,$iops" | tee -a "$OUT_BAM"
done

# --- CoPilotIO async (SQ CPU + CQ CPU + CPU polling + GDRCopy notify + async drain) ---
OUT_COP="$SCRIPT_DIR/results_copilotio_read_bw.csv"
echo "sms,bandwidth_gibs,iops_m" > "$OUT_COP"
echo "=== CoPilotIO (1 warp/SM, $QPS QPs, async) ==="
for sms in $SM_LIST; do
    batch=$(get_async_batch "$sms")
    echo ">>> CoPilotIO SMs=$sms batch=$batch"
    out=$(LD_LIBRARY_PATH="$LIB_COP:$GDRCOPY_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
          "$BIN_COP" --ctrl "$CTRL" --sms "$sms" --warps 1 --copilot --async --batch "$batch" --qd 4096 --io-size "$IO" $BENCH_IOS 2>&1)
    bw=$(echo "$out" | grep "Bandwidth:" | awk '{print $2}')
    iops=$(echo "$out" | grep "IOPS:" | awk '{print $2}')
    echo "$sms,$bw,$iops" | tee -a "$OUT_COP"
done

echo ""
echo "=== Results ==="
echo "BaM:       $OUT_BAM"
echo "CoPilotIO: $OUT_COP"

echo ""
echo "=== Plotting ==="
python3 "$SCRIPT_DIR/plot.py"
