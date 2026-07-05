#!/usr/bin/env python3
"""
make_local_zoo.py
-----------------
Build a LOCAL DeGirum model zoo from the Hailo HEFs you already have, so you can
run accuracy evaluation fully offline (no cloud, no token).

For each model it writes <zoo>/<model>.json, symlinks the .hef next to it, and
writes one shared labels_coco.json. Your HEFs use Hailo on-chip NMS, so the
built-in 'DetectionYoloHailo' postprocessor handles all of them.

Usage:
    python make_local_zoo.py --hef-dir resources/models/hailo8 --zoo-dir ./hailo_local_zoo
    # then point eval_hailo_map.py --zoo ./hailo_local_zoo
"""

import argparse
import json
import os

MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov11n", "yolov11s", "yolov11m"]

COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
]


def model_json(hef_filename, device):
    # Minimal preprocess on purpose: over-specifying it triggers a
    # "DG_FLT vs DG_UINT8" mismatch with Hailo HEFs. InputQuantEn=true is the key.
    return {
        "ConfigVersion": 11,
        "Checksum": "local",  # required field; any non-empty value is fine for local models
        "DEVICE": [
            {
                "DeviceType": device,
                "RuntimeAgent": "HAILORT",
                "SupportedDeviceTypes": f"HAILORT/{device}",
            }
        ],
        "PRE_PROCESS": [
            {
                "InputType": "Image",
                "InputN": 1,
                "InputH": 640,
                "InputW": 640,
                "InputC": 3,
                "InputPadMethod": "letterbox",
                "InputResizeMethod": "bilinear",
                "InputQuantEn": True,
            }
        ],
        "MODEL_PARAMETERS": [{"ModelPath": hef_filename}],
        "POST_PROCESS": [
            {
                "OutputPostprocessType": "DetectionYoloHailo",
                "OutputNumClasses": 80,
                "LabelsPath": "labels_coco.json",
                "OutputConfThreshold": 0.001,  # low for eval; full PR curve
            }
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef-dir", default="resources/models/hailo8",
                    help="dir containing yolov8n.hef etc.")
    ap.add_argument("--zoo-dir", default="./hailo_local_zoo")
    ap.add_argument("--device", default="HAILO8", choices=["HAILO8", "HAILO8L"])
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    os.makedirs(args.zoo_dir, exist_ok=True)

    # shared labels file: {"0": "person", ...}
    labels = {str(i): name for i, name in enumerate(COCO80)}
    with open(os.path.join(args.zoo_dir, "labels_coco.json"), "w") as f:
        json.dump(labels, f, indent=2)

    made = 0
    for short in args.models:
        hef_src = os.path.abspath(os.path.join(args.hef_dir, f"{short}.hef"))
        if not os.path.exists(hef_src):
            print(f"[skip] {short}: not found at {hef_src}")
            continue

        hef_name = f"{short}.hef"
        hef_link = os.path.join(args.zoo_dir, hef_name)
        model_path_value = hef_name
        # symlink the HEF into the zoo so ModelPath can stay relative; fall back to abs path
        try:
            if os.path.islink(hef_link) or os.path.exists(hef_link):
                os.remove(hef_link)
            os.symlink(hef_src, hef_link)
        except OSError:
            model_path_value = hef_src  # use absolute path in JSON instead

        cfg = model_json(model_path_value, args.device)
        with open(os.path.join(args.zoo_dir, f"{short}.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[ok]   {short}: {short}.json + {hef_name}")
        made += 1

    print(f"\nBuilt {made} models in {os.path.abspath(args.zoo_dir)}")
    print("Next: python eval_hailo_map.py --zoo", args.zoo_dir,
          "--images <val2017> --ann <instances_val2017.json> --max-images 100 --only yolov8n")


if __name__ == "__main__":
    main()
