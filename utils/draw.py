# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""
This is the script of cix draw tools.
"""
from .label.coco_classes import COCO_CLASSES, _COLORS
from .label.voc_classes import VOC_CLASSES
import cv2
import numpy as np
import colorsys


def draw_coco(
    image: np.ndarray,
    bboxes: np.ndarray,
    classes: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    """
    Draws COCO bounding boxes and labels on an image.

    Args:
        image (np.ndarray):
            The input image on which the bounding boxes will be drawn.
            Must be a NumPy array in BGR format.
        bboxes (np.ndarray):
            Array of bounding boxes with shape (N, 4), where N is the number of detections.
            Each bounding box is represented as [x1, y1, x2, y2].
        classes (np.ndarray):
            Array of class indices for each detection. Shape: (N,).
        confidences (np.ndarray):
            Array of confidence scores for each detection. Shape: (N,).

    Returns:
        np.ndarray:
            The input image with drawn bounding boxes and labels.

    Example:
        >>> image = cv2.imread("image.jpg")
        >>> bboxes = np.array([[50, 50, 150, 150], [200, 200, 300, 300]])
        >>> classes = np.array([0, 1])
        >>> confidences = np.array([0.95, 0.85])
        >>> result_image = draw_coco(image, bboxes, classes, confidences)
    """
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
        txt_color = (0, 0, 0) if np.mean(_COLORS[label_id]) > 0.5 else (255, 255, 255)

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


def draw_keypoints(
    img : np.ndarray,
    current_poses : list
    ) -> np.ndarray:
    """
    Draw keypoints and bounding boxes on the given image.

    Args:
        img (numpy.ndarray):
            The input image on which keypoints will be drawn.
        current_poses (list):
            A list of pose objects, each with a draw method and a bounding box attribute.

    Returns:
        numpy.ndarray
            The image with keypoints and bounding boxes drawn on it.
    """
    orig_img = np.copy(img)
    for pose in current_poses:
        pose.draw(img)
    img = cv2.addWeighted(orig_img, 0.6, img, 0.4, 0)
    for pose in current_poses:
        cv2.rectangle(
            img,
            (pose.bbox[0], pose.bbox[1]),
            (pose.bbox[0] + pose.bbox[2], pose.bbox[1] + pose.bbox[3]),
            (255, 255, 0),
        )
    return img


class Colors:
    def __init__(self):
        hexs = (
            "FF3838",
            "FF9D97",
            "FF701F",
            "FFB21D",
            "CFD231",
            "48F90A",
            "92CC17",
            "3DDB86",
            "1A9334",
            "00D4BB",
            "2C99A8",
            "00C2FF",
            "344593",
            "6473FF",
            "0018EC",
            "8438FF",
            "520085",
            "CB38FF",
            "FF95C8",
            "FF37C7",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))


# Draw the inference result on the image
def draw_yolov8seg(
    im : np.ndarray,
    bboxes : list,
    segments : list,
    color_palette : callable = Colors()
    ) -> np.ndarray:
    """
    Draw bounding boxes and segmentation masks on the input image.

    Args:
        im (numpy.ndarray):
            The input image on which to draw the segments and bounding boxes.
        bboxes (list):
            A list of bounding boxes where each bounding box is represented as 
            [x_min, y_min, x_max, y_max, confidence, class_id].
        segments (list):
            A list of segmentation polygons corresponding to each bounding box.
        color_palette (callable, optional):
            A function to retrieve colors based on class IDs (default is a Colors() instance).

    Returns:
        numpy.ndarray
            The modified image with drawn bounding boxes and segmentation masks.
    """
    # Draw rectangles and polygons
    im_canvas = im.copy()
    for (*box, conf, cls_), segment in zip(bboxes, segments):
        # draw contour and fill mask
        cv2.polylines(
            im, np.int32([segment]), True, (255, 255, 255), 2
        )  # white borderline
        cv2.fillPoly(
            im_canvas, np.int32([segment]), color_palette(int(cls_), bgr=True)
        )
        # draw bbox rectangle
        cv2.rectangle(
            im,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color_palette(int(cls_), bgr=True),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            im,
            f"{COCO_CLASSES[int(cls_)]}: {conf:.3f}",
            (int(box[0]), int(box[1] - 9)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color_palette(int(cls_), bgr=True),
            2,
            cv2.LINE_AA,
        )
    # Mix image
    im = cv2.addWeighted(im_canvas, 0.3, im, 0.7, 0)
    return im


def draw_voc(image : np.ndarray, results : list) -> np.ndarray:
    """
    Draws VOC bounding boxes and labels on an image.

    Args:
        image (np.ndarray):
            The input image on which the bounding boxes will be drawn.
            Must be a NumPy array in BGR format.
        results (list):
            A list of detection results, where each result is a NumPy array of shape (N, 6),
            where N is the number of detections. Each detection is represented as
            [xmin, ymin, xmax, ymax, confidence, class_id].

    Returns:
        np.ndarray:
            The input image with drawn bounding boxes and labels.
    """
    hsv_tuples = [(x / 20, 1., 1.) for x in range(20)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    top_label   = np.array(results[0][:, 5], dtype = 'int32')
    top_conf    = results[0][:, 4]
    top_boxes   = results[0][:, :4]
    image_shape = image.shape[:2]
    thickness = max((image_shape[0] + image_shape[1]) // 512, 1)
    for i, c in list(enumerate(top_label)):
        predicted_class = VOC_CLASSES[int(c)]
        box             = top_boxes[i]
        score           = top_conf[i]
        top, left, bottom, right = box
        top     = max(0, np.floor(top).astype('int32'))
        left    = max(0, np.floor(left).astype('int32'))
        bottom  = min(image_shape[1], np.floor(bottom).astype('int32'))
        right   = min(image_shape[0], np.floor(right).astype('int32'))
        label = '{} {:.2f}'.format(predicted_class, score)
        font = cv2.FONT_HERSHEY_SIMPLEX
        label_size = cv2.getTextSize(label, font, 0.4, 1)[0]
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        for i in range(thickness):
            img_ = cv2.rectangle(image, (left + i, top + i), (right - i, bottom - i), colors[c], 1)

        cv2.rectangle(
        img_,
        (text_origin[0], text_origin[1]),
        (text_origin[0] + label_size[0] + 1, text_origin[1] + int(label_size[1])),
        colors[c],
        -1,
    )
        # Put the text label on the image
        cv2.putText(
            img_,
            label,
            (text_origin[0], text_origin[1] + label_size[1]),
            font,
            0.4,
            (0,0,0),
            thickness=1,
        )
    return img_


def draw_handpose(img0 : np.ndarray,output : np.ndarray) -> np.ndarray:
    """
    Draws handpose keypoints and lines on an image.

    Args:
        img0 (np.ndarray):
            The input image on which the keypoints will be drawn.
            Must be a NumPy array in BGR format.
        output (np.ndarray):
            A NumPy array of shape (1, 42).
            Each keypoint is represented as [x, y].

    Returns:
        np.ndarray:
            The input image with drawn keypoints and lines.
    """
    img_width = img0.shape[1]
    img_height = img0.shape[0]
    pts_hand = {}
    for i in range(int(output.shape[0]/2)):
        x = (output[i*2+0]*float(img_width))
        y = (output[i*2+1]*float(img_height))

        pts_hand[str(i)] = {}
        pts_hand[str(i)] = {
            "x":x,
            "y":y,
            }
    thick = 2
    colors = [(0,215,255),(255,115,55),(5,255,55),(25,15,255),(225,15,55)]
    cv2.line(img0 , (int(pts_hand['0']['x']), int(pts_hand['0']['y'])),(int(pts_hand['1']['x']), int(pts_hand['1']['y'])), colors[0], thick)
    cv2.line(img0 , (int(pts_hand['1']['x']), int(pts_hand['1']['y'])),(int(pts_hand['2']['x']), int(pts_hand['2']['y'])), colors[0], thick)
    cv2.line(img0 , (int(pts_hand['2']['x']), int(pts_hand['2']['y'])),(int(pts_hand['3']['x']), int(pts_hand['3']['y'])), colors[0], thick)
    cv2.line(img0 , (int(pts_hand['3']['x']), int(pts_hand['3']['y'])),(int(pts_hand['4']['x']), int(pts_hand['4']['y'])), colors[0], thick)

    cv2.line(img0 , (int(pts_hand['0']['x']), int(pts_hand['0']['y'])),(int(pts_hand['5']['x']), int(pts_hand['5']['y'])), colors[1], thick)
    cv2.line(img0 , (int(pts_hand['5']['x']), int(pts_hand['5']['y'])),(int(pts_hand['6']['x']), int(pts_hand['6']['y'])), colors[1], thick)
    cv2.line(img0 , (int(pts_hand['6']['x']), int(pts_hand['6']['y'])),(int(pts_hand['7']['x']), int(pts_hand['7']['y'])), colors[1], thick)
    cv2.line(img0 , (int(pts_hand['7']['x']), int(pts_hand['7']['y'])),(int(pts_hand['8']['x']), int(pts_hand['8']['y'])), colors[1], thick)

    cv2.line(img0 , (int(pts_hand['0']['x']), int(pts_hand['0']['y'])),(int(pts_hand['9']['x']), int(pts_hand['9']['y'])), colors[2], thick)
    cv2.line(img0 , (int(pts_hand['9']['x']), int(pts_hand['9']['y'])),(int(pts_hand['10']['x']), int(pts_hand['10']['y'])), colors[2], thick)
    cv2.line(img0 , (int(pts_hand['10']['x']), int(pts_hand['10']['y'])),(int(pts_hand['11']['x']), int(pts_hand['11']['y'])), colors[2], thick)
    cv2.line(img0 , (int(pts_hand['11']['x']), int(pts_hand['11']['y'])),(int(pts_hand['12']['x']), int(pts_hand['12']['y'])), colors[2], thick)

    cv2.line(img0 , (int(pts_hand['0']['x']), int(pts_hand['0']['y'])),(int(pts_hand['13']['x']), int(pts_hand['13']['y'])), colors[3], thick)
    cv2.line(img0 , (int(pts_hand['13']['x']), int(pts_hand['13']['y'])),(int(pts_hand['14']['x']), int(pts_hand['14']['y'])), colors[3], thick)
    cv2.line(img0 , (int(pts_hand['14']['x']), int(pts_hand['14']['y'])),(int(pts_hand['15']['x']), int(pts_hand['15']['y'])), colors[3], thick)
    cv2.line(img0 , (int(pts_hand['15']['x']), int(pts_hand['15']['y'])),(int(pts_hand['16']['x']), int(pts_hand['16']['y'])), colors[3], thick)

    cv2.line(img0 , (int(pts_hand['0']['x']), int(pts_hand['0']['y'])),(int(pts_hand['17']['x']), int(pts_hand['17']['y'])), colors[4], thick)
    cv2.line(img0 , (int(pts_hand['17']['x']), int(pts_hand['17']['y'])),(int(pts_hand['18']['x']), int(pts_hand['18']['y'])), colors[4], thick)
    cv2.line(img0 , (int(pts_hand['18']['x']), int(pts_hand['18']['y'])),(int(pts_hand['19']['x']), int(pts_hand['19']['y'])), colors[4], thick)
    cv2.line(img0 , (int(pts_hand['19']['x']), int(pts_hand['19']['y'])),(int(pts_hand['20']['x']), int(pts_hand['20']['y'])), colors[4], thick)
    for i in range(int(output.shape[0]/2)):
        x = (output[i*2+0]*float(img_width))
        y = (output[i*2+1]*float(img_height))

        cv2.circle(img0 , (int(x),int(y)), 3, (255,50,60),-1)
        cv2.circle(img0 , (int(x),int(y)), 1, (255,150,180),-1)

def draw_yolo_word(image : np.ndarray, boxes : list, scores : list, class_ids : list, names : list)-> np.ndarray:
    """
    Draws YOLOv8l-wordv2 bounding boxes and labels on an image.

    Args:
        image (np.ndarray):
            The input image on which the bounding boxes will be drawn.
            Must be a NumPy array in BGR format.
        boxes (list):
            A list of bounding boxes where each bounding box is represented as
            [x_min, y_min, x_max, y_max].
        scores (list):
            A list of confidence scores for each bounding box.
        class_ids (list):
            A list of class IDs for each bounding box.
        names (list):
            A list of class names for each bounding box.

    Returns:
        np.ndarray:
            The input image with drawn bounding boxes and labels.
    """
    for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box
            x1, y1 = int(x - w / 2), int(y - h / 2)
            x2, y2 = int(x + w / 2), int(y + h / 2)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = names[class_id]
            cv2.putText(image, f"{class_name}: {score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image


def draw_masks(image: np.ndarray, masks, alpha: float = 0.5, draw_border: bool = True) -> np.ndarray:
    """
    Draws masks on the image.
    Args:
        image: The image to draw on.
        masks: A dictionary of label_id to mask.
        alpha: The transparency of the masks.
        draw_border: Whether to draw the border of the masks.
    Returns:
        The image with the masks drawn on it.
    """
    rng = np.random.default_rng(2)
    colors = rng.uniform(0, 255, size=(100, 3))
    mask_image = image.copy()
    for label_id, label_masks in masks.items():
        if label_masks is None:
            continue
        color = colors[label_id]
        mask_image = draw_mask(mask_image, label_masks, (color[0], color[1], color[2]), alpha, draw_border)

    return mask_image

def draw_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (0, 255, 0), alpha: float = 0.5, draw_border: bool = True) -> np.ndarray:
    """
    Draws a mask on the image.
    Args:
        image: The image to draw on.
        mask: The mask to draw.
        color: The color of the mask.
        alpha: The transparency of the mask.
        draw_border: Whether to draw the border of the mask.
    Returns:
        The image with the mask drawn on it.
    """
    mask_image = image.copy()
    mask_image[mask > 0.01] = color
    mask_image = cv2.addWeighted(image, 1-alpha, mask_image, alpha, 0)

    if draw_border:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask_image = cv2.drawContours(mask_image, contours, -1, color, thickness=2)

    return mask_image


def draw_Ultra_Fast_Lane_Detection(img : np.ndarray,out_j : np.ndarray, col_sample_w : int) -> np.ndarray:
    """
    Draws Ultra-Fast-Lane-Detection results on an image.

    Args:
        img (np.ndarray):
            The input image on which the results will be drawn.
            Must be a NumPy array in BGR format.
        out_j (np.ndarray):
            A NumPy array of shape (1, 56).
        col_sample_w (int):
            The column sample width of the input image.

    Returns:
        np.ndarray:
            The input image with drawn results.
    """
    cls_num_per_lane = 56
    row_anchor = [ 64,  68,  72,  76,  80,  84,  88,  92,  96, 100, 104, 108, 112,
            116, 120, 124, 128, 132, 136, 140, 144, 148, 152, 156, 160, 164,
            168, 172, 176, 180, 184, 188, 192, 196, 200, 204, 208, 212, 216,
            220, 224, 228, 232, 236, 240, 244, 248, 252, 256, 260, 264, 268,
            272, 276, 280, 284]
    img_h, img_w = img.shape[:2]
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                    cv2.circle(img,ppp,5,(0,255,0),-1)
    return img