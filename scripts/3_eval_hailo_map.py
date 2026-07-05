#!/usr/bin/env python3
"""
eval_hailo_map.py  (LOCAL ZOO version)
--------------------------------------
Measure COCO mAP for YOLO models running INT8 on a Hailo-8, using a LOCAL DeGirum
zoo built by make_local_zoo.py. Fully offline: no cloud, no token.

Prereqs (on the Pi, in a venv SEPARATE from your hailo-apps env):
    pip install degirum degirum_tools pycocotools

Build the local zoo first:
    python make_local_zoo.py --hef-dir resources/models/hailo8 --zoo-dir ./hailo_local_zoo

Then:
    # smoke test (100 images, one model):
    python eval_hailo_map.py --zoo ./hailo_local_zoo \
        --images /path/to/val2017 --ann /path/to/instances_val2017.json \
        --max-images 100 --only yolov8n

    # full run over all six models -> npu_accuracy.csv:
    python eval_hailo_map.py --zoo ./hailo_local_zoo \
        --images /path/to/val2017 --ann /path/to/instances_val2017.json
"""

import argparse
import csv
import os
import sys

import degirum as dg

# The evaluator's import path moved between degirum_tools versions:
try:
    from degirum_tools import ObjectDetectionModelEvaluator          # newer top-level
except ImportError:
    try:
        from degirum_tools.detection_eval import ObjectDetectionModelEvaluator
    except ImportError:
        from degirum_tools.eval import ObjectDetectionModelEvaluator  # alt path

HOST = "@local"
MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov11n", "yolov11s", "yolov11m"]

# Standard COCO 80 -> 91 category-id map (model output index -> COCO category id).
# Prevents the "only detects first N classes" mismatch when scoring vs instances_val2017.
COCO_CLASSMAP = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]

# pycocotools stats order in results[0]
STAT_COLS = [
    ("mAP50_95", 0), ("mAP50", 1), ("mAP75", 2),
    ("mAP_small", 3), ("mAP_medium", 4), ("mAP_large", 5),
    ("AR_1", 6), ("AR_10", 7), ("AR_100", 8),
    ("AR_small", 9), ("AR_medium", 10), ("AR_large", 11),
]


def eval_one(short, zoo, image_dir, coco_json, max_images, token):
    model = dg.load_model(
        model_name=short,
        inference_host_address=HOST,
        zoo_url=zoo,          # LOCAL directory -> no cloud
        token=token,          # free DeGirum token: licenses the local HailoRT runtime
    )
    # standard COCO-eval settings (low conf, high max-dets)
    model.output_confidence_threshold = 0.001
    try:
        model.output_nms_threshold = 0.7
        model.output_max_detections = 300
        model.output_max_detections_per_class = 300
    except Exception:
        pass  # some are baked into the on-chip NMS; ignore if not settable

    evaluator = ObjectDetectionModelEvaluator(model, classmap=COCO_CLASSMAP)
    results = evaluator.evaluate(image_dir, coco_json, max_images=max_images)
    return results[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zoo", required=True, help="local zoo dir from make_local_zoo.py")
    ap.add_argument("--images", required=True, help="COCO val2017 image dir")
    ap.add_argument("--ann", required=True, help="instances_val2017.json")
    ap.add_argument("--out", default="npu_accuracy.csv")
    ap.add_argument("--max-images", type=int, default=0, help="0 = all images")
    ap.add_argument("--only", help="single short name, e.g. yolov8n")
    ap.add_argument("--token", default=os.environ.get("DEGIRUM_CLOUD_TOKEN", ""),
                    help="free DeGirum token (dg_...). Or set DEGIRUM_CLOUD_TOKEN, "
                         "or install once with `degirum token` so it's automatic.")
    args = ap.parse_args()

    if not args.token:
        print("[warn] no --token given and DEGIRUM_CLOUD_TOKEN not set. "
              "Local Hailo inference needs a free token from hub.degirum.com.")

    targets = [args.only] if args.only else MODELS
    rows = []
    for short in targets:
        print(f"[run] {short}")
        try:
            stats = eval_one(short, args.zoo, args.images, args.ann, args.max_images, args.token)
        except Exception as e:
            print(f"[error] {short}: {e}")
            continue
        row = {"model": short}
        row.update({col: round(float(stats[i]), 4) for col, i in STAT_COLS})
        rows.append(row)
        print(f"      mAP50-95={row['mAP50_95']}  mAP50={row['mAP50']}")

    if not rows:
        print("No results — check --zoo path and that the smoke test runs.")
        sys.exit(1)

    fields = ["model"] + [c for c, _ in STAT_COLS]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
