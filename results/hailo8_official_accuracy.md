# Hailo-8 official accuracy reference (NPU / INT8)

**Source:** Hailo Model Zoo — `docs/public_models/HAILO8/HAILO8_object_detection.rst` (COCO table). All models compiled with Hailo Dataflow Compiler v2.18.0. The ⭐ models in Hailo's table are the ones used by Hailo-apps — i.e. the exact HEFs in your `resources/models/hailo8/`, so these numbers apply to what you benchmarked.

**Column meaning:**
- **float mAP** = the FP32 model, COCO mAP@[.50:.95] ×100 (Hailo's reference).
- **hardware mAP** = the **quantized INT8 model running on the Hailo-8** = your NPU accuracy reference.
- **Quant drop** = float − hardware = the accuracy cost of Hailo's INT8 PTQ.

## Your six models

| Model | float mAP | hardware mAP (INT8) | Quant drop | Input | Params (M) |
|---|---|---|---|---|---|
| YOLOv8n  | 37.0 | 36.4 | −0.6 | 640 | 3.2 |
| YOLOv8s  | 44.6 | 44.0 | −0.6 | 640 | 11.2 |
| YOLOv8m  | 49.9 | 49.2 | −0.7 | 640 | 25.9 |
| YOLOv11n | 39.0 | 37.8 | −1.2 | 640 | 2.6 |
| YOLOv11s | 46.3 | 45.5 | −0.8 | 640 | 9.4 |
| YOLOv11m | 51.1 | 49.8 | −1.3 | 640 | 20.1 |

(mAP50-95 ×100. These are the standard COCO AP numbers; divide by 100 to match your CPU values like 0.374.)

## Cross-check vs your measured CPU (FP32) numbers — validates your pipeline

| Model | Your CPU mAP (measured) | Hailo float mAP | Match? |
|---|---|---|---|
| YOLOv8n | 37.4 | 37.0 | ✅ +0.4 |
| YOLOv8s | 45.0 | 44.6 | ✅ +0.4 |
| YOLOv8m | 50.3 | 49.9 | ✅ +0.4 |

Your full-COCO-val CPU results land within ~0.4 of Hailo's FP32 reference across all three — confirming your CPU pipeline is correct and comparable. (The tiny offset is preprocessing/eval differences, well within noise.)

## FPS: your Pi+Hailo vs the full-chip spec (the PCIe bottleneck)

| Model | Your NPU FPS (measured) | Hailo spec FPS (bs=1) | Ratio |
|---|---|---|---|
| YOLOv8n  | 431.5 | 1036 | 0.42 |
| YOLOv8s  | 491.3 | 491  | 1.00 |
| YOLOv8m  | 31.5  | 67.3 | 0.47 |
| YOLOv11n | 104.9 | 185  | 0.57 |
| YOLOv11s | 52.0  | 111  | 0.47 |
| YOLOv11m | 24.3  | 50.2 | 0.48 |

Your numbers run ~0.4–0.6× the full-chip spec (except v8s, which matches). This is the **PCIe Gen3 ×1** single lane on the Pi AI HAT+ plus host overhead — the lightest, fastest models are most data-transfer-bound, which is why v8n (spec 1036) loses the most relative throughput. Report your measured numbers and cite this as the reason for the gap.

## Three findings you can use directly

1. **Hailo INT8 is nearly lossless** — every model drops < 1.3 mAP points from FP32. This *contradicts* the seed paper's 6+ point INT8 losses, and supports the argument that large INT8 degradation comes from weak calibration/method, not from INT8 itself. Strong critical-analysis point.
2. **YOLOv11 quantizes slightly worse than YOLOv8** (−1.2/−1.3 for v11n/m vs −0.6/−0.7 for v8) *and* runs far slower on the NPU (v11n 105 vs v8n 431 FPS). On fixed edge silicon, newer ≠ better.
3. **The Pi's PCIe lane caps NPU throughput** at roughly half the chip's rated FPS for most models — a Pi-AI-HAT-specific result the spec sheets don't show.

## How to use these numbers in the paper

- **Fastest path (cite):** use the **hardware mAP** column as your NPU accuracy, labeled "Hailo Model Zoo reported (INT8, DFC v2.18.0)." Defensible and immediate.
- **Stronger path (measure):** reproduce these with `eval_hailo_map.py` (DeGirum) on your own COCO val and report your measured INT8 mAP next to Hailo's. Note: community reports DeGirum-measured mAP can land slightly *below* the Model Zoo figure — that's expected (eval/postprocess differences), and the gap itself is worth a sentence.
