# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""
This script performs NPU-based inference with timing support and textual detection logs.
All disk I/O for images and progress bars have been removed.
"""
import numpy as np
import argparse
import os
import sys
import time
import platform

# Determine the directory containing the current script
script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, script_dir)

from utils.tools import get_file_list
from utils.image_process import preprocess_object_detect_method1
from utils.object_detect_postprocess import postprocess_yolo, xywh2xyxy
from utils.NOE_Engine import EngineInfer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_path",
        default="palace.jpg",
        help="Path to the image file or directory of images",
    )
    parser.add_argument(
        "--model_path",
        default="model/yolov8_l.cix",
        help="Path to the quantized model file",
    )
    parser.add_argument(
        "--conf_thr",
        type=float,
        default=0.3,
        help="Score threshold for filtering detections",
    )
    parser.add_argument(
        "--nms_thr",
        type=float,
        default=0.45,
        help="IoU threshold for non-max suppression",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of timed runs per image (first run is warm-up)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # Print system information
    sys_info = platform.uname()
    print(f"=== System info: {sys_info.system} {sys_info.node} {sys_info.release} {sys_info.version} {sys_info.machine} {sys_info.processor} ===\n")

    # Build the list of images
    image_list = get_file_list(args.image_path)

    # Initialize the NPU inference engine
    model = EngineInfer(args.model_path)

    # Collect timing statistics
    all_times = []

    for img_name in image_list:
        # Preprocess image
        src_shape, new_shape, show_image, data = preprocess_object_detect_method1(
            img_name, target_size=(640, 640), mode="BGR"
        )
        data = data.astype(np.float32)

        # Warm-up run
        _ = model.forward(data)[0]

        # Timed runs
        for _ in range(args.runs):
            t0 = time.perf_counter()
            out = model.forward(data)[0]
            dt = (time.perf_counter() - t0) * 1000.0  # milliseconds
            all_times.append(dt)

        # Post-process the last output
        pred = out.reshape(84, 8400).transpose(1, 0)
        results = postprocess_yolo(pred, args.conf_thr, args.nms_thr)

        # Textual output of detections
        print(f"\n=== Detections for {os.path.basename(img_name)} ===")
        if len(results) == 0:
            print("  No objects detected.")
        else:
            # Convert bboxes back to original image scale
            bbox_xywh = results[:, :4]
            bbox_xyxy = xywh2xyxy(bbox_xywh)
            x_scale = src_shape[1] / new_shape[1]
            y_scale = src_shape[0] / new_shape[0]
            bbox_xyxy *= (x_scale, y_scale, x_scale, y_scale)

            for idx, row in enumerate(results):
                cls_id = int(row[5])
                conf = float(row[4])
                x1, y1, x2, y2 = bbox_xyxy[idx]
                print(
                    f"  [{idx}] class={cls_id}, conf={conf:.3f}, "
                    f"bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})"
                )

    # Print timing summary
    if all_times:
        avg = sum(all_times) / len(all_times)
        mn = min(all_times)
        mx = max(all_times)
        print(f"\nInference over {len(all_times)} runs:")
        print(f"  avg = {avg:.2f} ms   min = {mn:.2f} ms   max = {mx:.2f} ms")

    # Clean up the model
    model.clean()
