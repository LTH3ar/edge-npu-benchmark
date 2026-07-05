# Edge NPU Benchmark — RPi5 CPU vs RPi5 + Hailo-8

Seminar 1 (RCE2: Inference Efficiency with Pruning/Quantization). This bundle contains everything for the project: the report draft, the plan, all scripts, all raw logs, the processed results, and the seed paper source.

**Research question:** Does adding a low-power 26-TOPS Hailo-8 NPU to a Raspberry Pi 5 overturn the seed paper's conclusion that the RPi5 cannot do real-time YOLO — and at what accuracy and energy cost?

**Seed paper:** Rey et al., "A Performance Analysis of YOLO Models for Deployment on Constrained Computational Edge Devices in Drone Applications," Electronics 2025 (arXiv:2502.15737). Source in `seed_paper/`.

---

## Folder guide

```
edge-npu-benchmark/
├── README.md                        <- this file (overview + full method + how to reproduce)
├── scripts/                         <- run order is the file-number order
│   ├── 1_ncnn_cpu_bench.py          <- CPU speed (MIS/IPM) + power, NCNN FP32
│   ├── 2_make_local_zoo.py          <- builds a local DeGirum zoo from your HEFs
│   └── 3_eval_hailo_map.py          <- NPU INT8 accuracy (mAP) via DeGirum local zoo
├── results/
│   ├── benchmark_results.xlsx       <- MASTER results table (live formulas) + Notes sheet
│   ├── hailo8_official_accuracy.md  <- Hailo Model Zoo reference numbers + sources
│   ├── cpu_ncnn_results.csv         <- raw output of script 1 (CPU speed/power/EPI)
│   └── npu_accuracy.csv             <- raw output of script 3 (NPU INT8 mAP)
├── data/
│   ├── npu_speed/
│   │   └── hailo8_benchmark.log     <- hailortcli benchmark output (NPU FPS/latency)
│   ├── cpu_accuracy_ncnn/           <- yolo val logs, NCNN FP32, full COCO val (final)
│   │   └── *_ncnn_val.log           <- 6 models
└── └── cpu_accuracy_pytorch_archive/<- original PyTorch CPU val logs (SUPERSEDED by NCNN)
        └── *_test.log

```

---

## Hardware / software

- Raspberry Pi 5 (8 GB), active cooling, Pi OS 64-bit.
- Hailo-8 on the Raspberry Pi AI HAT+ (26 TOPS), PCIe Gen3 x1.
- CPU path: Ultralytics YOLO exported to NCNN (FP32) — multi-threaded, ARM-optimized; matches the seed paper's RPi5 setup.
- NPU path: HailoRT, INT8, Hailo Model Zoo HEFs.
- Models: YOLOv8n/s/m, YOLOv11n/s/m at 640x640.
- Dataset: COCO val2017 (5000 images). Accuracy = COCO mAP via pycocotools.

Two separate Python venvs were used and should be kept separate (DeGirum bundles its own HailoRT and can clash with the hailo-apps install):
- `venv_hailo_apps` — for `hailortcli` benchmarks (NPU speed).
- a CPU venv with `ultralytics ncnn pycocotools` — for the CPU path.
- a DeGirum venv with `degirum degirum_tools pycocotools` — for NPU accuracy.

---

## What we did, step by step

### Step 1 — NPU speed (already had this)
Ran `hailortcli benchmark <model>.hef` for all six models. Output in `data/npu_speed/hailo8_benchmark.log`. Gives FPS (hw_only) and HW latency. Note: the AI HAT+ does NOT report power, so the "Power on supported platforms" line is empty — this is expected and is why NPU power is taken from the datasheet, not measured.

### Step 2 — NPU accuracy (INT8 mAP), fully local
The Hailo path is INT8-only and the compiler quantizes during compile, so we measure the deployed INT8 model's accuracy directly.
1. `python scripts/2_make_local_zoo.py --hef-dir <dir with your .hef> --zoo-dir ./hailo_local_zoo`
   Writes a DeGirum model `.json` per HEF + a shared `labels_coco.json`, symlinks the HEFs. Uses the `DetectionYoloHailo` postprocessor (correct for the on-chip-NMS HEFs) and a minimal preprocess block (over-specifying it triggers a DG_FLT/DG_UINT8 error).
2. Install a free DeGirum token (required even for LOCAL inference on this build):
   `degirum token install <dg_token>`   (get it free from hub.degirum.com)
3. `python scripts/3_eval_hailo_map.py --zoo ./hailo_local_zoo --images <val2017> --ann <instances_val2017.json>`
   Runs each HEF over COCO val, scores with pycocotools (80->91 COCO classmap to avoid the "first-20-classes" bug). Output: `results/npu_accuracy.csv`.

Validation: our measured INT8 mAP reproduces Hailo's published "hardware mAP" within ~0.4 pt (see `results/hailo8_official_accuracy.md`).

### Step 3 — CPU speed + power, NCNN FP32 (the fair baseline)
The first CPU attempt used stock single-threaded PyTorch (~3 FPS), which is unfair and not comparable to the seed paper. We switched to NCNN FP32 to match the paper and give the CPU its best effort.
- `python scripts/1_ncnn_cpu_bench.py --images <val2017>`
  For each model: exports to NCNN if needed, warms up, times MIS (inference-only FPS over N images) while logging `vcgencmd pmic_read_adc` power in a background thread, then runs a 60 s IPM count. EPI = mean_power * 60 / IPM. Output: `results/cpu_ncnn_results.csv`.
- Power note: the parser sums V*I across all metered rails (VDD_CORE, 1V8, DDR x2, 3V3 x2). The 5V rail has no current sensor and is excluded by design.

### Step 4 — CPU accuracy, NCNN FP32 (full COCO val)
Ran `yolo val task=detect model=<model>_ncnn_model imgsz=640 data=coco.yaml` for all six.
Logs in `data/cpu_accuracy_ncnn/`. We use the COCOeval (pycocotools) AP numbers to stay consistent with the NPU eval. NCNN FP32 mAP matched the earlier PyTorch run within 0.001, confirming the export is lossless. (The PyTorch logs are archived in `data/cpu_accuracy_pytorch_archive/` for provenance but are superseded.)

### Step 5 — Consolidation
All numbers merged into `results/benchmark_results.xlsx` (live formulas; Speedup, Quant-drop, EPI, Energy-ratio, and validation deltas compute automatically). The Notes sheet documents sources and caveats.

---

## Key results (see benchmark_results.xlsx for the full table)

- NPU speedup over the fair NCNN CPU baseline: 7.5x–82x. Every model moves from < 14 FPS (CPU) into 24–491 FPS (NPU) — overturns the seed paper's "RPi5 not real-time" verdict.
- INT8 accuracy cost (FP32 -> INT8): 0.7–1.3 mAP points across all six. Far below the seed paper's 6+ point INT8 loss -> their large loss is a calibration/single-class-dataset artifact, not intrinsic to INT8.
- Energy per inference: NPU is 12x–110x more efficient than CPU.
- YOLOv11 is 4–9x slower than YOLOv8 on the same NPU, and quantizes slightly worse — newer != better on fixed silicon.
- Pi reaches ~0.4–0.6x the Hailo-8's rated FPS = the AI HAT+'s single PCIe Gen3 lane.
- Measured NPU mAP reproduces Hailo's published numbers within ~0.4 pt (independent validation).

---

## Power measurement decision (important for the methods section)

- CPU power: MEASURED via `vcgencmd pmic_read_adc`.
- NPU power: DATASHEET (Hailo-8 typical ~2.5 W), labeled manufacturer-reported. The AI HAT+ exposes no on-board power telemetry, and the Pi's PMIC cannot read the 5V rail the HAT draws through, so there is no software path to NPU power. An external wall/USB meter was considered and dropped (a cheap AC meter can't resolve a ~10 W load accurately). The energy conclusion is robust to this because the throughput advantage is so large.
- Asymmetry to disclose: CPU EPI uses IPM (full loop); NPU EPI uses datasheet power / hw FPS.

---

## Status & what's left

- Data collection: COMPLETE (all six models: CPU speed/power/accuracy, NPU speed/accuracy).
- Report: first draft done (`report/report_draft.md`). TODO: author names, citation formatting + in-text cites, trim to 4–6 double-column pages, add figures, fact-check the two seed-paper specifics (their dataset image count and exact INT8 mAP numbers) against the PDF.
- Not done (scoped as future work): pruning experiments, QAT, harder/aerial dataset, newest YOLO versions.

## Reproducing from scratch (quick reference)

```bash
# NPU speed
hailortcli benchmark <model>.hef

# NPU accuracy
python scripts/2_make_local_zoo.py --hef-dir <hef_dir> --zoo-dir ./hailo_local_zoo
degirum token install <dg_token>
python scripts/3_eval_hailo_map.py --zoo ./hailo_local_zoo --images <val2017> --ann <instances_val2017.json>

# CPU speed + power
python scripts/1_ncnn_cpu_bench.py --images <val2017>

# CPU accuracy
for m in yolov8n yolov8s yolov8m yolo11n yolo11s yolo11m; do
  yolo val task=detect model=${m}_ncnn_model imgsz=640 data=coco.yaml | tee ${m}_ncnn_val.log
done
```
