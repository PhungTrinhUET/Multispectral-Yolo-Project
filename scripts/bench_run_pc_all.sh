#!/bin/bash
# ─── Master pipeline: PC-side benchmark (Steps 1–4 + Pre-flight) ──────────────
#
# Run from ultralytics/ directory:
#   bash scripts/bench_run_pc_all.sh
#
# Prerequisites:
#   pip install ptflops onnx onnxsim onnxruntime onnx2tf tensorflow

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "============================================================"
echo " GMD-YOLO Edge Benchmark — PC Phase"
echo " Model: Exp09_CMX_Redux (FusionRectify / CMX-YOLO)"
echo " Input: 1×4×640×640  |  $(date)"
echo "============================================================"

echo ""
echo "── Step 1: Profile (params, GFLOPs, FP32 accuracy, GPU FPS) ──"
python scripts/bench_step1_profile_pc.py

echo ""
echo "── Step 2: Export ONNX ──────────────────────────────────────"
python scripts/bench_step2_export_onnx.py

echo ""
echo "── Step 3a: Build calibration dataset ───────────────────────"
python scripts/bench_step3_make_calib.py

echo ""
echo "── Step 3b: Convert TFLite (FP32 / FP16 / INT8) + Gate 2 ──"
python scripts/bench_step3_convert_tflite.py

echo ""
echo "── Step 4: Accuracy check across variants ───────────────────"
python scripts/bench_step4_accuracy_variants.py

echo ""
echo "── PRE-FLIGHT: Gate 2 full report + Gate 3a reference save ─"
python scripts/bench_preflight_pc.py

echo ""
echo "============================================================"
echo " PC phase complete.  Results in: bench_results/"
echo ""
echo " ─── PRE-FLIGHT GATE 1 (run ON VOXL2 FIRST, before full sweep) ───"
echo " adb push bench_results/ /data/gmd_bench/"
echo " adb push scripts/bench_step5_voxl2_harness.py /data/gmd_bench/"
echo " adb push scripts/bench_step7_power_voxl2.sh   /data/gmd_bench/"
echo ""
echo " # Gate 1+3: MUST PASS before full benchmark"
echo " adb shell 'cd /data/gmd_bench && python3 bench_step5_voxl2_harness.py --preflight'"
echo ""
echo " # Only if preflight passes — 3 VALID pairings only:"
echo " adb shell 'cd /data/gmd_bench && python3 bench_step5_voxl2_harness.py --variant fp32 --delegate CPU'"
echo " adb shell 'cd /data/gmd_bench && python3 bench_step5_voxl2_harness.py --variant fp16 --delegate GPU'"
echo " adb shell 'cd /data/gmd_bench && python3 bench_step5_voxl2_harness.py --variant int8 --delegate NPU'"
echo ""
echo " # Power (run in parallel with each benchmark):"
echo " adb shell 'cd /data/gmd_bench && bash bench_step7_power_voxl2.sh <variant> <delegate>'"
echo ""
echo " ─── JETSON Orin Nano ────────────────────────────────────────────"
echo " scp -r bench_results/ jetson:/data/gmd_bench/"
echo " scp scripts/bench_step6_jetson_trt.py jetson:/data/gmd_bench/"
echo ""
echo " ssh jetson 'python3 /data/gmd_bench/bench_step6_jetson_trt.py --build_only'"
echo " # Then for each variant:"
echo " ssh jetson 'tegrastats --interval 500 --logfile /data/gmd_bench/tgs_fp16.log &"
echo "              python3 /data/gmd_bench/bench_step6_jetson_trt.py --variant fp16 &&"
echo "              kill %1'"
echo " ssh jetson 'python3 /data/gmd_bench/bench_step6_jetson_trt.py --parse_power /data/gmd_bench/tgs_fp16.log'"
echo ""
echo " ─── Collect all results → Paper Tables ──────────────────────────"
echo " adb pull /data/gmd_bench/voxl2_*.json  bench_results/"
echo " adb pull /data/gmd_bench/power_*.json  bench_results/"
echo " scp jetson:/data/gmd_bench/jetson_*.json bench_results/"
echo " python scripts/bench_step8_collect_tables.py"
echo "============================================================"
