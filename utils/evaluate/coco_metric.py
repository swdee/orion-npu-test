# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import json
import cv2
import numpy as np


COCO_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)

_COLORS = (
    np.array(
        [
            0.000,
            0.447,
            0.741,
            0.850,
            0.325,
            0.098,
            0.929,
            0.694,
            0.125,
            0.494,
            0.184,
            0.556,
            0.466,
            0.674,
            0.188,
            0.301,
            0.745,
            0.933,
            0.635,
            0.078,
            0.184,
            0.300,
            0.300,
            0.300,
            0.600,
            0.600,
            0.600,
            1.000,
            0.000,
            0.000,
            1.000,
            0.500,
            0.000,
            0.749,
            0.749,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.333,
            0.333,
            0.000,
            0.333,
            0.667,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            0.333,
            0.000,
            0.667,
            0.667,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.667,
            0.000,
            1.000,
            1.000,
            0.000,
            0.000,
            0.333,
            0.500,
            0.000,
            0.667,
            0.500,
            0.000,
            1.000,
            0.500,
            0.333,
            0.000,
            0.500,
            0.333,
            0.333,
            0.500,
            0.333,
            0.667,
            0.500,
            0.333,
            1.000,
            0.500,
            0.667,
            0.000,
            0.500,
            0.667,
            0.333,
            0.500,
            0.667,
            0.667,
            0.500,
            0.667,
            1.000,
            0.500,
            1.000,
            0.000,
            0.500,
            1.000,
            0.333,
            0.500,
            1.000,
            0.667,
            0.500,
            1.000,
            1.000,
            0.500,
            0.000,
            0.333,
            1.000,
            0.000,
            0.667,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            0.000,
            1.000,
            0.333,
            0.333,
            1.000,
            0.333,
            0.667,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            0.000,
            1.000,
            0.667,
            0.333,
            1.000,
            0.667,
            0.667,
            1.000,
            0.667,
            1.000,
            1.000,
            1.000,
            0.000,
            1.000,
            1.000,
            0.333,
            1.000,
            1.000,
            0.667,
            1.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.167,
            0.000,
            0.000,
            0.333,
            0.000,
            0.000,
            0.500,
            0.000,
            0.000,
            0.667,
            0.000,
            0.000,
            0.833,
            0.000,
            0.000,
            1.000,
            0.000,
            0.000,
            0.000,
            0.143,
            0.143,
            0.143,
            0.286,
            0.286,
            0.286,
            0.429,
            0.429,
            0.429,
            0.571,
            0.571,
            0.571,
            0.714,
            0.714,
            0.714,
            0.857,
            0.857,
            0.857,
            0.000,
            0.447,
            0.741,
            0.314,
            0.717,
            0.741,
            0.50,
            0.5,
            0,
        ]
    )
    .astype(np.float32)
    .reshape(-1, 3)
)


def coco80_to_coco91_class():
    """
    Converts COCO 80-class index to COCO 91-class index used in the paper.

    Reference: https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    """
    # a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    # b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    # x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]  # darknet to coco
    # x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]  # coco to darknet
    return [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]


class COCO_Metric:
    def __init__(
        self,
        saved_json_path: str,
        annotation_path: str = None,
        coco_images_path: str = None,
    ) -> None:
        if annotation_path is None:
            self.coco = COCO(
                os.path.join(
                    os.path.dirname(__file__),
                    "../../datasets/COCO2017/annotations/instances_val2017.json",
                )
            )
        else:
            self.coco = COCO(annotation_path)

        if coco_images_path is None:
            self.coco_images_path = os.path.join(
                os.path.dirname(__file__),
                "../../datasets/COCO2017/images/val2017",
            )
        else:
            self.coco_images_path = coco_images_path

        self.class_map = coco80_to_coco91_class()
        self.results = []
        self.saved_json_path = saved_json_path

    def get_image_ids(self):
        return self.coco.getImgIds()

    def get_image_info(self, img_id):
        return self.coco.loadImgs(img_id)[0]

    def get_image_path(self, img_id):
        image_path = os.path.join(
            self.coco_images_path, self.get_image_info(img_id)["file_name"]
        )
        return image_path

    def saved_json(self):
        with open(self.saved_json_path, "w") as f:
            json.dump(self.results, f)

    def _append(self, img_id, class_id, bbox, score):
        self.results.append(
            {
                "image_id": img_id,
                "category_id": int(self.class_map[int(class_id)]),
                "bbox": [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2] - bbox[0]),
                    float(bbox[3] - bbox[1]),
                ],  # convert coco format
                "score": float(score),
            }
        )

    def append_bboxes(self, img_id, bboxes, classes, confs):

        for i in range(bboxes.shape[0]):
            score = confs[i]
            class_id = classes[i]
            bbox = bboxes[i]
            self._append(img_id, class_id, bbox, score)

    def _load_res(self, load_path):
        coco_dt = self.coco.loadRes(load_path)
        self.coco_eval = COCOeval(self.coco, coco_dt, iouType="bbox")

    def evaluate(self):
        self._load_res(self.saved_json_path)
        self.coco_eval.evaluate()
        self.coco_eval.accumulate()
        self.coco_eval.summarize()

    @staticmethod
    def draw(image, bboxes, classes, confidences):
        if len(classes) == 0:
            return image
        for i in range(bboxes.shape[0]):
            label_id = int(classes[i])
            score = confidences[i]
            x1, y1, x2, y2 = bboxes[i]
            det = [int(x1), int(y1), int(x2), int(y2)]
            classname = COCO_CLASSES[label_id]
            color = (_COLORS[int(label_id)] * 255).astype(np.uint8).tolist()
            text = "{}:{:.1f}%".format(classname, float(score) * 100)
            txt_color = (
                (0, 0, 0) if np.mean(_COLORS[label_id]) > 0.5 else (255, 255, 255)
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            img_ = cv2.rectangle(image, (det[0], det[1]), (det[2], det[3]), color, 1)

            txt_bk_color = (_COLORS[label_id] * 255 * 0.7).astype(np.uint8).tolist()
            cv2.rectangle(
                img_,
                (det[0], det[1] + 1),
                (det[0] + txt_size[0] + 1, det[1] + int(1.5 * txt_size[1])),
                txt_bk_color,
                -1,
            )

            # Put the text label on the image
            cv2.putText(
                img_,
                text,
                (det[0], det[1] + txt_size[1]),
                font,
                0.4,
                txt_color,
                thickness=1,
            )

        return img_
