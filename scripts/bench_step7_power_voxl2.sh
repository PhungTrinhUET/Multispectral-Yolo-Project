#!/bin/bash
# Step 7 — Power measurement on VOXL2 Sentinel
#
# Run ON VOXL2. Two methods:
#   Method A (preferred): External USB power meter on 5V rail → record manually
#   Method B: px4-listener battery_status → whole-board estimate via PX4
#   Method C: sysfs INA sensor (if available on VOXL2 carrier board)
#
# Usage:
#   bash bench_step7_power_voxl2.sh fp16 GPU
#   bash bench_step7_power_voxl2.sh int8 NPU

VARIANT=${1:-fp16}
DELEGATE=${2:-GPU}
BENCH_DIR=${BENCH_DIR:-/data/gmd_bench}
LOG_FILE="$BENCH_DIR/power_${VARIANT}_${DELEGATE}.log"
DURATION=180   # seconds — must match n_sustained_s in step 5

echo "[POWER] variant=$VARIANT  delegate=$DELEGATE"
echo "[POWER] Logging ${DURATION}s → $LOG_FILE"

# ── Method B: px4-listener battery_status ────────────────────────────────────
if command -v px4-listener &>/dev/null; then
    echo "[POWER] Method B: px4-listener battery_status"
    echo "# timestamp_s  current_a  voltage_v  power_w" > "$LOG_FILE"
    END=$((SECONDS + DURATION))
    while [ $SECONDS -lt $END ]; do
        SAMPLE=$(px4-listener battery_status -n 1 2>/dev/null \
                 | grep -E 'current_a|voltage_v' \
                 | awk '{print $NF}' \
                 | paste - -)
        CURR=$(echo "$SAMPLE" | cut -f1)
        VOLT=$(echo "$SAMPLE" | cut -f2)
        if [[ -n "$CURR" && -n "$VOLT" ]]; then
            PWR=$(echo "$CURR * $VOLT" | bc -l 2>/dev/null || echo "N/A")
            echo "$(date +%s)  $CURR  $VOLT  $PWR" | tee -a "$LOG_FILE"
        fi
        sleep 1
    done

# ── Method C: sysfs INA3221 sensor ───────────────────────────────────────────
elif ls /sys/bus/i2c/drivers/ina3221/ &>/dev/null 2>&1; then
    echo "[POWER] Method C: INA3221 sysfs"
    INA_DIR=$(find /sys/bus/i2c/drivers/ina3221/ -name "in_power*_input" \
              | head -1 | xargs dirname 2>/dev/null)
    echo "# timestamp_s  power_mW" > "$LOG_FILE"
    END=$((SECONDS + DURATION))
    while [ $SECONDS -lt $END ]; do
        PWR_MW=$(cat "$INA_DIR/in_power1_input" 2>/dev/null || echo "N/A")
        echo "$(date +%s)  $PWR_MW" | tee -a "$LOG_FILE"
        sleep 0.5
    done

else
    echo "[WARN] No automated power measurement available."
    echo "       Use external USB meter on VOXL2 5V power input."
    echo "       Record idle_W (board idle) and inference_W (during bench run)."
    echo "       Formula: energy_mJ = (inference_W / sustained_FPS) * 1000"
    echo ""
    echo "       When ready, enter measurements:"
    read -p "  Idle power (W)      : " IDLE_W
    read -p "  Inference power (W) : " INFER_W
    read -p "  Sustained FPS       : " SUS_FPS

    ENERGY=$(echo "scale=2; ($INFER_W / $SUS_FPS) * 1000" | bc 2>/dev/null || echo "N/A")
    RESULT="{
  \"device\": \"VOXL2_Sentinel\",
  \"variant\": \"$VARIANT\",
  \"delegate\": \"$DELEGATE\",
  \"measurement_method\": \"external_meter_manual\",
  \"note\": \"WHOLE-BOARD power, not NPU-isolated\",
  \"idle_w\": $IDLE_W,
  \"inference_w\": $INFER_W,
  \"sustained_fps\": $SUS_FPS,
  \"energy_per_inference_mJ\": $ENERGY
}"
    echo "$RESULT" | tee "$BENCH_DIR/power_${VARIANT}_${DELEGATE}.json"
    exit 0
fi

# ── Parse log → summary JSON ──────────────────────────────────────────────────
python3 - "$LOG_FILE" "$VARIANT" "$DELEGATE" "$BENCH_DIR" <<'EOF'
import sys, json, re
import numpy as np

log_file, variant, delegate, bench_dir = sys.argv[1:]
powers = []
try:
    with open(log_file) as f:
        for line in f:
            if line.startswith('#'): continue
            cols = line.strip().split()
            if len(cols) >= 4:
                try: powers.append(float(cols[3]))
                except: pass
except Exception as e:
    print(f"[WARN] {e}")

if not powers:
    print("[WARN] No power data parsed")
    sys.exit(0)

arr = np.array(powers)
print(f"Power (W)  mean={arr.mean():.2f}  max={arr.max():.2f}  min={arr.min():.2f}")

result_file = f"{bench_dir}/power_{variant}_{delegate}.json"
with open(result_file, 'w') as f:
    json.dump({
        "device": "VOXL2_Sentinel",
        "variant": variant, "delegate": delegate,
        "measurement_method": "px4_battery_status_or_INA3221",
        "note": "WHOLE-BOARD power, not NPU-isolated — state this in paper",
        "inference_power_mean_w": round(float(arr.mean()), 2),
        "inference_power_max_w":  round(float(arr.max()),  2),
        "n_samples": len(powers),
        "log_file": log_file,
    }, f, indent=2)
print(f"Saved: {result_file}")
EOF
