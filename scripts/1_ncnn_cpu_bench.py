#!/usr/bin/env python3
"""
ncnn_cpu_bench.py
-----------------
CPU benchmark on Raspberry Pi 5 following the seed paper's methodology:
  - NCNN, FP32 (exports automatically if needed)
  - MIS  = 1000 / mean inference-only time (ms), over N images at 640x640
  - IPM  = full-loop inferences counted over a 60 s continuous run
  - Power = vcgencmd pmic_read_adc, sampled during the MIS run, averaged
  - EPI  = mean_power * 60 / IPM   (paper's formula)

Accuracy is NOT recomputed here: NCNN FP32 mAP == your existing FP32 mAP
(export doesn't change weights/precision), so keep the accuracy numbers you have.

Run on the Pi, in the venv that has ultralytics:
    pip install ultralytics ncnn        # ncnn usually pulled by the export step
    python ncnn_cpu_bench.py --images /path/to/coco/images/val2017

Output: cpu_ncnn_results.csv  (model, MIS, mean_infer_ms, IPM, mean/max power, EPI)
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import threading
import time

from ultralytics import YOLO

MODELS = ["yolov8n", "yolov8s", "yolov8m", "yolov11n", "yolov11s", "yolov11m"]


def read_pmic_power():
    """Sum V*I across all measured Pi rails. (5V rail has no current sensor -> excluded.)"""
    try:
        out = subprocess.check_output(["vcgencmd", "pmic_read_adc"], text=True)
    except Exception:
        return None
    amps, volts = {}, {}
    for line in out.splitlines():
        ma = re.search(r"(\S+)_A current\(\d+\)=([\d.]+)A", line)
        if ma:
            amps[ma.group(1)] = float(ma.group(2))
        mv = re.search(r"(\S+)_V volt\(\d+\)=([\d.]+)V", line)
        if mv:
            volts[mv.group(1)] = float(mv.group(2))
    return sum(volts[r] * amps[r] for r in amps if r in volts)


class PowerLogger(threading.Thread):
    def __init__(self, interval=1.0):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            p = read_pmic_power()
            if p is not None:
                self.samples.append(p)
            self._stop.wait(self.interval)

    def stop(self):
        self._stop.set()

    def stats(self):
        s = self.samples
        if not s:
            return 0.0, 0.0, 0
        return sum(s) / len(s), max(s), len(s)


def ensure_ncnn(name):
    d = f"{name}_ncnn_model"
    if not os.path.isdir(d):
        print(f"  exporting {name} -> NCNN (FP32)...")
        YOLO(f"{name}.pt").export(format="ncnn", imgsz=640)
    return d


def bench(name, images, mis_n, ipm_secs, warmup):
    d = ensure_ncnn(name)
    model = YOLO(d, task="detect")

    for i in range(warmup):
        model(images[i % len(images)], verbose=False, imgsz=640)

    # ---- MIS: inference-only time, with power logging in parallel ----
    pw = PowerLogger()
    pw.start()
    inf_ms = []
    for i in range(mis_n):
        r = model(images[i % len(images)], verbose=False, imgsz=640)
        inf_ms.append(r[0].speed["inference"])
    pw.stop()
    pw.join()
    mean_ms = sum(inf_ms) / len(inf_ms)
    mis = 1000.0 / mean_ms
    mean_w, max_w, nsamp = pw.stats()

    # ---- IPM: full-loop inferences in ipm_secs (incl. pre/post) ----
    n = 0
    i = 0
    t0 = time.time()
    while time.time() - t0 < ipm_secs:
        model(images[i % len(images)], verbose=False, imgsz=640)
        n += 1
        i += 1
    ipm = n * (60.0 / ipm_secs)

    epi = (mean_w * 60.0 / ipm) if ipm else 0.0
    return dict(
        model=name, mis_fps=round(mis, 2), mean_infer_ms=round(mean_ms, 2),
        ipm=round(ipm, 1), mean_power_w=round(mean_w, 3), max_power_w=round(max_w, 3),
        epi_j=round(epi, 4), power_samples=nsamp,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="dir of .jpg images (e.g. COCO val2017)")
    ap.add_argument("--out", default="cpu_ncnn_results.csv")
    ap.add_argument("--mis-images", type=int, default=500,
                    help="images for MIS timing (paper used 5000; 500 gives a stable mean faster)")
    ap.add_argument("--ipm-secs", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--only", help="single model name, e.g. yolov8n")
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()

    need = max(args.mis_images, 64)
    images = sorted(glob.glob(os.path.join(args.images, "*.jpg")))[:need]
    if not images:
        print("No .jpg images found in", args.images)
        return

    targets = [args.only] if args.only else args.models
    rows = []
    for name in targets:
        print(f"[bench] {name}")
        try:
            row = bench(name, images, args.mis_images, args.ipm_secs, args.warmup)
        except Exception as e:
            print(f"  [error] {name}: {e}")
            continue
        rows.append(row)
        print(f"  MIS={row['mis_fps']} FPS | IPM={row['ipm']} | "
              f"power={row['mean_power_w']} W (n={row['power_samples']}) | EPI={row['epi_j']} J")

    if not rows:
        print("No results.")
        return
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nWrote", args.out)


if __name__ == "__main__":
    main()
