# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""
This is the script of cix object detection post process.
"""
import numpy as np
import torch
from torchvision.ops import nms
import torchvision
from .label.coco_classes import COCO_CLASSES
from typing import List, Union, Tuple
import scipy.special

names = COCO_CLASSES


# Yolo_v8 postprocess
# Calculate the union area
def get_iou(
    box1 : np.ndarray,
    box2 : np.ndarray,
    inter_area : float
    ) -> float:
    """
    Calculate the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (np.ndarray): The first bounding box defined as (x, y, width, height).
        box2 (np.ndarray): The second bounding box defined as (x, y, width, height).
        inter_area (float): The area of the intersection between the two bounding boxes.

    Returns:
        float: The IoU score, which is the ratio of the intersection area 
            to the union area of the two bounding boxes.
    """
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou


# Calculate the intersection area
def get_inter(
    box1 : np.ndarray,
    box2 : np.ndarray
    ) -> float:
    """
    Calculate the intersection area between two bounding boxes.

    Args:
        box1 (np.ndarray): The first bounding box defined as (x, y, width, height).
        box2 (np.ndarray): The second bounding box defined as (x, y, width, height).

    Returns:
        float: The area of the intersection between the two bounding boxes. 
            Returns 0 if the boxes do not intersect.
    """
    box1_x1, box1_y1, box1_x2, box1_y2 = (
        box1[0] - box1[2] / 2,
        box1[1] - box1[3] / 2,
        box1[0] + box1[2] / 2,
        box1[1] + box1[3] / 2,
    )
    box2_x1, box2_y1, box2_x2, box2_y2 = (
        box2[0] - box2[2] / 2,
        box2[1] - box1[3] / 2,
        box2[0] + box2[2] / 2,
        box2[1] + box2[3] / 2,
    )
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter


# NMS the inference result
def nms_yolo(pred: np.ndarray, conf_thres: float, iou_thres: float) -> List[np.ndarray]:
    """
    Perform Non-Maximum Suppression (NMS) for YOLO predictions.
    Args:
        pred (np.ndarray): Predictions array of shape (N, D), where D >= 6.
                           Each prediction includes bounding box coordinates (x, y, w, h),
                           object confidence, and class probabilities.
                           Example structure: [x, y, w, h, obj_conf, class_prob1, class_prob2, ...].
        conf_thres (float): Confidence threshold for filtering predictions.
                            Only predictions with object confidence > conf_thres will be considered.
        iou_thres (float): Intersection over Union (IoU) threshold for suppressing overlapping boxes.
    Returns:
        List[np.ndarray]: List of filtered bounding boxes after NMS. Each bounding box is represented as:
                          [x, y, w, h, confidence, class_id], where:
                          - `x, y, w, h`: Bounding box coordinates.
                          - `confidence`: Confidence score of the box.
                          - `class_id`: Predicted class ID.
    """
    # Filter out predictions with confidence below the threshold
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]
    cls_conf = box[..., 5:]
    cls = []
    # extract the class of each box
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    total_cls = list(set(cls))
    output_box = []
    # Process each class
    for i in range(len(total_cls)):
        clss = total_cls[i]
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]
        box_conf_sort = np.argsort(box_conf)
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)
        cls_box = np.delete(cls_box, 0, 0)
        # Remove the highest confidence box and apply NMS
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]
                interArea = get_inter(max_conf_box, current_box)
                iou = get_iou(max_conf_box, current_box, interArea)
                if iou > iou_thres:
                    del_index.append(j)
            # Remove boxes that are too similar
            cls_box = np.delete(cls_box, del_index, 0)
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box


def postprocess_yolox(
    outputs : np.ndarray,
    img_size : tuple,
    conf_thr : float,
    nms_thr : float,
    p6 : bool = False
    ) -> List[np.ndarray]:
    """
    Post-process the raw outputs of the YOLOX model to extract the bounding boxes 
    and their corresponding scores.

    Parameters:
        outputs (numpy.ndarray): The raw model outputs from YOLOX, 
                                expected shape (num_boxes, 85) where 85 includes x, y, width, height, and class scores.
        img_size (tuple): The size of the input image as (height, width).
        conf_thr (float): Confidence threshold for filtering out low-confidence predictions.
        nms_thr (float): Non-maximum suppression threshold for overlapping bounding boxes.
        p6 (bool): Flag indicating whether to use a P6 scale. If True, adds an additional scale.

    Returns:
        list: A list of processed results for each input image, containing the bounding boxes and their scores after 
            applying confidence thresholding and non-maximum suppression.
    """
    grids = []
    expanded_strides = []
    strides = [8, 16, 32] if not p6 else [8, 16, 32, 64]

    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]

    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = np.meshgrid(np.arange(wsize), np.arange(hsize))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(np.full((*shape, 1), stride))

    grids = np.concatenate(grids, 1)
    expanded_strides = np.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * expanded_strides

    results = []
    for i in range(outputs.shape[0]):
        boxes = outputs[i][:, :4]
        scores = outputs[i][:, 4:5] * outputs[i][:, 5:]
        predictions = np.concatenate([boxes, scores], axis=-1)
        result = postprocess_yolo(predictions, conf_thr=conf_thr, nms_thr=nms_thr)
        results.append(result)
    return results


# Postprocess the inference result
def postprocess_yolo(
    pred : np.ndarray,
    conf_thr : float,
    nms_thr : float
    ) -> np.ndarray:
    """
    Post-process the predictions from the YOLO model to filter out low-confidence 
    detections and apply non-maximum suppression (NMS) to eliminate redundant bounding boxes.

    Args:
        pred (numpy.ndarray): The predictions array containing bounding box coordinates, 
                            object confidence, and class scores. Shape should be (num_boxes, 85).
        conf_thr (float): Confidence threshold for filtering out weak predictions.
        nms_thr (float): Non-maximum suppression threshold for overlapping bounding boxes.

    Returns:
        numpy.ndarray: The filtered and processed results after applying confidence thresholding 
                    and non-maximum suppression, containing bounding boxes, confidence scores, and class IDs.
                    Returns an empty array if no results pass the thresholds.
    """
    pred_class = pred[..., 4:]
    pred_conf = np.max(pred_class, axis=-1)
    pred = np.insert(pred, 4, pred_conf, axis=-1)
    # boxes, confs, class_ids
    results = nms_yolo(pred, conf_thr, nms_thr)
    if len(results) == 0:
        return results

    results = np.concatenate([np.expand_dims(res, axis=0) for res in results])
    return results


# Yolo_v3 postprocess
def xywh2xyxy(
    x : Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert bounding box format from [center_x, center_y, width, height] 
    to [x1, y1, x2, y2] format, which represents the top-left and bottom-right 
    corners of the bounding box.

    Args:
        x (Union[torch.Tensor, np.ndarray]): The input bounding boxes in the format 
                                        [center_x, center_y, width, height]. 
                                        Expected shape is (num_boxes, 4).

    Returns:
        Union[torch.Tensor, np.ndarray]: The converted bounding boxes in the format 
                                    [x1, y1, x2, y2], with the same type as the input.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def xyxy2xywh(
    x : Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
    """
    Convert bounding box format from [x1, y1, x2, y2]
    to [center_x, center_y, width, height],
    supporting both torch.Tensor and numpy.ndarray.

    Args:
        x (Union[torch.Tensor, np.ndarray]): The input bounding boxes in the format
                                        [x1, y1, x2, y2]. Expected shape is
                                        (num_boxes, 4).

    Returns:
        Union[torch.Tensor, np.ndarray]: The converted bounding boxes in the format
                                    [center_x, center_y, width, height], with the
                                    same type as the input.
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.

    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.

    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])

    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def nms_yolov3(
    prediction : np.ndarray,
    conf_thres : float = 0.25,
    iou_thres : float = 0.45,
    classes : List[int] = None,
    agnostic : bool = False,
    multi_label : bool = False,
    labels=(),
    max_det : int = 300,
    nm : int = 0,  # number of masks
) -> List[np.ndarray]:
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections.

    Args:
        prediction (numpy.ndarray): The predictions array containing bounding box coordinates,
                                    object confidence, and class scores. Format should be (num_boxes, 85).
        conf_thres (float): Confidence threshold for filtering out low-confidence detections.
                            Default is 0.25.
        iou_thres (float): IoU threshold for determining whether two boxes overlap.
                            Default is 0.45.
        classes (list or None): A list of class indices to keep. If None, all classes are kept.
        agnostic (bool): If True, performs agnostic NMS and does not distinguish between classes.
        multi_label (bool): If True, allows multiple labels per box.
        labels (tuple): Tuple of tuples containing ground truth label coordinates for evaluation.
        max_det (int): Maximum number of detections to keep for each image. Default is 300.
        nm (int): The number of masks; optional parameter for handling masks if applicable.

    Returns:
        list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv3 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    # t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4]
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[
            x[:, 4].argsort(descending=True)[:max_nms]
        ]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)

    return output


def clip_boxes(
    boxes : Union[torch.Tensor, np.ndarray],
    shape : tuple
    ):
    """
    Clip bounding boxes to ensure they are within the specified image shape.

        Args:
            boxes (torch.Tensor or numpy.ndarray): The bounding boxes to be clipped, expected shape (num_boxes, 4),
                                                where each box is defined as (x1, y1, x2, y2).
            shape (tuple): The shape of the image as (height, width), used to constrain bounding box coordinates.
    """
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[..., 0].clamp_(0, shape[1])  # x1
        boxes[..., 1].clamp_(0, shape[0])  # y1
        boxes[..., 2].clamp_(0, shape[1])  # x2
        boxes[..., 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2


def scale_boxes(
    img1_shape : tuple,
    boxes : np.ndarray,
    img0_shape : tuple,
    ratio_pad : tuple = None
    ) -> np.ndarray:
    """
    Rescale bounding boxes from one image size to another, with optional ratio and padding adjustments.

    Args:
        img1_shape (tuple): The shape of the target image as (height, width).
        boxes (numpy.ndarray): The bounding boxes to be rescaled, expected shape (num_boxes, 4),
                            where each box is defined as (x1, y1, x2, y2).
        img0_shape (tuple): The shape of the original image as (height, width).
        ratio_pad (tuple, optional): A tuple containing the scaling ratio and padding to adjust the boxes.
                                    If None, it calculates from img0_shape.

    Returns:
        numpy.ndarray: The rescaled bounding boxes adjusted for the new image size, clipped to the dimensions of img0_shape.
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    boxes[..., [0, 2]] -= pad[0]  # x padding
    boxes[..., [1, 3]] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


# Postprocess the inference result
def postprocess_yolo_v3(
    old_shape : tuple,
    new_shape : tuple,
    outputs : np.ndarray,
    conf_thr : float,
    nms_thr : float
    ) -> list:
    """
    Post-process the outputs from the YOLOv3 model to adjust bounding box coordinates, 
    apply non-maximum suppression (NMS), and scale the boxes to the original image size.

    Args:
        old_shape (tuple): The original shape of the input image as (height, width).
        new_shape (tuple): The shape of the input image the model processed as (height, width).
        outputs (numpy.ndarray): The raw model outputs, expected to be in a specific format 
                                for YOLOv3 (num_boxes, 85).
        conf_thr (float): Confidence threshold for filtering out low-confidence detections.
        nms_thr (float): Non-maximum suppression threshold for overlapping bounding boxes.

    Returns:
        list: A list of detected bounding boxes after applying NMS and scaling, 
            where each entry contains the coordinates of the bounding boxes.
    """
    outputs = torch.from_numpy(outputs[0])
    pred = nms_yolov3(outputs, conf_thr, nms_thr)
    h, w = old_shape
    for i, det in enumerate(pred):
        det[:, :4] = scale_boxes(new_shape, det[:, :4], new_shape)
        det[:, :4] = det[:, :4] * torch.tensor([w / 640, h / 640, w / 640, h / 640])
    return pred


def pool_nms(heat : torch.Tensor, kernel : int = 3) -> torch.Tensor:
    """
    Performs non-maximum suppression (NMS) on the heatmap using max pooling.

    Args:
        heat (torch.Tensor): The heatmap to apply NMS on, expected shape (batch_size, num_classes, height, width).
        kernel (int): The size of the max pooling kernel.

    Returns:
        torch.Tensor: The heatmap with NMS applied, same shape as input.
    """
    pad = (kernel - 1) // 2

    hmax = torch.nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep

def decode_bbox(pred_hms : torch.Tensor, pred_whs : torch.Tensor, pred_offsets : torch.Tensor, confidence : float = 0.3) -> List[torch.Tensor]:
    """
    Decode the predicted heatmap, width and height to obtain the final bounding boxes.

    Args:
        pred_hms (torch.Tensor): The predicted heatmap, expected shape (batch_size, num_classes, height, width).
        pred_whs (torch.Tensor): The predicted width and height, expected shape (batch_size, 2, height, width).
        pred_offsets (torch.Tensor): The predicted offsets, expected shape (batch_size, 2, height, width).
        confidence (float): The confidence threshold for filtering out low-confidence detections.

    Returns:
        List[torch.Tensor]: A list of detected bounding boxes, where each entry contains the coordinates of the bounding boxes.
    """
    pred_hms = pool_nms(pred_hms)
    b, c, output_h, output_w = pred_hms.shape
    detects = []
    for batch in range(b):
        heat_map    = pred_hms[batch].permute(1, 2, 0).view([-1, c])
        pred_wh     = pred_whs[batch].permute(1, 2, 0).view([-1, 2])
        pred_offset = pred_offsets[batch].permute(1, 2, 0).view([-1, 2])
        yv, xv      = torch.meshgrid(torch.arange(0, output_h), torch.arange(0, output_w),indexing='ij')
        xv, yv      = xv.flatten().float(), yv.flatten().float()

        class_conf, class_pred  = torch.max(heat_map, dim = -1)
        mask                    = class_conf > confidence

        pred_wh_mask        = pred_wh[mask]
        pred_offset_mask    = pred_offset[mask]
        if len(pred_wh_mask) == 0:
            detects.append([])
            continue
        xv_mask = torch.unsqueeze(xv[mask] + pred_offset_mask[..., 0], -1)
        yv_mask = torch.unsqueeze(yv[mask] + pred_offset_mask[..., 1], -1)
        half_w, half_h = pred_wh_mask[..., 0:1] / 2, pred_wh_mask[..., 1:2] / 2
        bboxes = torch.cat([xv_mask - half_w, yv_mask - half_h, xv_mask + half_w, yv_mask + half_h], dim=1)
        bboxes[:, [0, 2]] /= output_w
        bboxes[:, [1, 3]] /= output_h
        detect = torch.cat([bboxes, torch.unsqueeze(class_conf[mask],-1), torch.unsqueeze(class_pred[mask],-1).float()], dim=-1)
        detects.append(detect)

    return detects

def centernet_correct_boxes(box_xy : torch.Tensor, box_wh : torch.Tensor, input_shape : List[int], image_shape : List[int]) -> torch.Tensor:
    """
    Adjust the predicted bounding boxes for the original image shape.

    Args:
        box_xy (torch.Tensor): The predicted bounding box center points, expected shape (batch_size, num_boxes, 2).
        box_wh (torch.Tensor): The predicted bounding box width and height, expected shape (batch_size, num_boxes, 2).
        input_shape (List[int]): The shape of the input image as (height, width).
        image_shape (List[int]): The shape of the original image as (height, width).

    Returns:
        torch.Tensor: The adjusted bounding boxes, same shape as input.
    """
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape)
    image_shape = np.array(image_shape)

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=-1)
    return boxes

def postprocess(prediction : List[torch.Tensor], image_shape : List[int], nms_thres : float = 0.4, input_shape : List[int] = [512, 512], need_nms : bool = True) -> List[np.ndarray]:
    """
    Post-process the predicted bounding boxes for the original image shape.

    Args:
        prediction (List[torch.Tensor]): The predicted bounding boxes, expected shape (batch_size, num_boxes, 6).
        image_shape (List[int]): The shape of the original image as (height, width).
        nms_thres (float): The non-maximum suppression threshold for overlapping bounding boxes.
        input_shape (List[int]): The shape of the input image as (height, width).
        need_nms (bool): Whether to apply non-maximum suppression (NMS) to the predicted bounding boxes.

    Returns:
        List[np.ndarray]: A list of detected bounding boxes, where each entry contains the coordinates of the bounding boxes.
    """
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        detections      = prediction[i]
        if len(detections) == 0:
            continue
        unique_labels   = detections[:, -1].cpu().unique()

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            if need_nms:
                keep = nms(
                    detections_class[:, :4],
                    detections_class[:, 4],
                    nms_thres
                )
                max_detections = detections_class[keep]
            else:
                max_detections  = detections_class

            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i].detach().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    return output


def postprocess_yolov8l_worldv2(output : np.ndarray, conf : float, imgsz : int, iou : float, img_width : int, img_height : int):
    """
    Process the output from the YOLOv8l-world model to obtain the final bounding boxes.

    Args:
        output (numpy.ndarray): The raw model outputs.
        conf (float): The confidence threshold for filtering out low-confidence detections.
        imgsz (int): The input image size for the model.
        iou (float): The non-maximum suppression threshold for overlapping bounding boxes.
        img_width (int): The width of the original image.
        img_height (int): The height of the original image.

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]: A tuple containing the detected bounding boxes, scores, and class IDs.
    """
    predictions = np.squeeze(output[0]).T
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores >= conf, :]
    scores = scores[scores > conf]
    class_ids = np.argmax(predictions[:, 4:], axis=1)
    boxes = extract_boxes(predictions, imgsz,img_width,img_height)
    detections = [(class_id, x, y, w, h, score)
                    for class_id, (x, y, w, h), score in zip(class_ids, boxes, scores)]
    nms_detections = apply_nms(detections, iou)
    boxes = []
    scores = []
    class_ids = []
    for det in nms_detections:
        class_id, x_nms, y_nms, w_nms, h_nms, score = det
        boxes.append([x_nms, y_nms, w_nms, h_nms])
        scores.append(score)
        class_ids.append(class_id)
    return boxes, scores, class_ids

def extract_boxes(predictions : np.ndarray, imgsz : int , img_width : int, img_height : int):
    """
    Extract the bounding boxes from the predicted output.

    Args:
        predictions (numpy.ndarray): The predicted output.
        imgsz (int): The input image size for the model.
        img_width (int): The width of the original image.
        img_height (int): The height of the original image.

    Returns:
        numpy.ndarray: The extracted bounding boxes, expected shape (num_predictions, 4).
    """
    boxes = predictions[:, :4]
    boxes[:, 0] /= imgsz
    boxes[:, 1] /= imgsz
    boxes[:, 2] /= imgsz
    boxes[:, 3] /= imgsz
    boxes[:, 0] *= img_width
    boxes[:, 1] *= img_height
    boxes[:, 2] *= img_width
    boxes[:, 3] *= img_height
    return boxes

def apply_nms(detections : List[Tuple[int, float, float]], iou_threshold : float) -> List[Tuple[int, float, float]]:
    """
    Apply non-maximum suppression (NMS) to the predicted bounding boxes.

    Args:
        detections (List[Tuple[int, float, float]]): The predicted bounding boxes, where each entry contains the class ID, x, y, w, h, and confidence.
        iou_threshold (float): The non-maximum suppression threshold for overlapping bounding boxes.

    Returns:
        List[Tuple[int, float, float]]: The filtered bounding boxes, where each entry contains the class ID, x, y, w, h, and confidence.
    """
    boxes = []
    for det in detections:
        (cls_id, x, y, w, h, confidence) = det
        boxes.append([x, y, w, h, cls_id, confidence])
    sorted_boxes = sorted(boxes, key=lambda x: x[5], reverse=True)
    selected_boxes = []
    while len(sorted_boxes) > 0:
        selected_boxes.append(sorted_boxes[0])
        remaining_boxes = []
        for box in sorted_boxes[1:]:
            x1_a, y1_a, w1_a, h1_a, _, _ = selected_boxes[-1]
            x1_b, y1_b, w1_b, h1_b, _, _ = box
            x2_a = x1_a + w1_a
            y2_a = y1_a + h1_a
            x2_b = x1_b + w1_b
            y2_b = y1_b + h1_b
            intersection = max(0, min(x2_a, x2_b) - max(x1_a, x1_b)) * max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
            union = w1_a * h1_a + w1_b * h1_b - intersection
            if union == 0:
                iou = 0
            else:
                iou = intersection / union
            if iou < iou_threshold:
                remaining_boxes.append(box)
        sorted_boxes = remaining_boxes
    nms_detections = []
    for box in selected_boxes:
        x, y, w, h, cls_id, confidence = box
        nms_detections.append((cls_id, x, y, w, h, confidence))
    return nms_detections


def decodebox_retinanet(regression : torch.Tensor, anchors : torch.Tensor, input_shape : List[int]) -> torch.Tensor:
    """
    Decode the predicted regression values to obtain the final bounding boxes.

    Args:
        regression (torch.Tensor): The predicted regression values, expected shape (batch_size, num_anchors, 4).
        anchors (torch.Tensor): The anchor boxes, expected shape (num_anchors, 4).
        input_shape (List[int]): The shape of the input image as (height, width).

    Returns:
        torch.Tensor: The decoded bounding boxes, same shape as input.
    """
    dtype   = regression.dtype
    anchors = anchors.to(dtype)
    y_centers_a = (anchors[..., 0] + anchors[..., 2]) / 2
    x_centers_a = (anchors[..., 1] + anchors[..., 3]) / 2
    ha = anchors[..., 2] - anchors[..., 0]
    wa = anchors[..., 3] - anchors[..., 1]
    w = regression[..., 3].exp() * wa
    h = regression[..., 2].exp() * ha
    y_centers = regression[..., 0] * ha + y_centers_a
    x_centers = regression[..., 1] * wa + x_centers_a
    ymin = y_centers - h / 2.
    xmin = x_centers - w / 2.
    ymax = y_centers + h / 2.
    xmax = x_centers + w / 2.
    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=2)
    boxes[:, :, [0, 2]] = boxes[:, :, [0, 2]] / input_shape[1]
    boxes[:, :, [1, 3]] = boxes[:, :, [1, 3]] / input_shape[0]

    boxes = torch.clamp(boxes, min = 0, max = 1)
    return boxes


def non_max_suppression(prediction : List[torch.Tensor], input_shape : List[int], image_shape : List[int], conf_thres : float = 0.5, nms_thres : float = 0.4) -> List[np.ndarray]:
    """
    Apply non-maximum suppression (NMS) to the predicted bounding boxes.

    Args:
        prediction (List[torch.Tensor]): The predicted bounding boxes, expected shape (batch_size, num_boxes, 6).
        input_shape (List[int]): The shape of the input image as (height, width).
        image_shape (List[int]): The shape of the original image as (height, width).
        conf_thres (float): The confidence threshold for filtering out low-confidence detections.
        nms_thres (float): The non-maximum suppression threshold for overlapping bounding boxes.

    Returns:
        List[np.ndarray]: A list of detected bounding boxes, where each entry contains the coordinates of the bounding boxes.
    """
    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        class_conf, class_pred = torch.max(image_pred[:, 4:], 1, keepdim=True)
        conf_mask = (class_conf[:, 0] >= conf_thres).squeeze()
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]
        if not image_pred.size(0):
            continue
        detections = torch.cat((image_pred[:, :4], class_conf.float(), class_pred.float()), 1)
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            detections_class = detections[detections[:, -1] == c]
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4],
                nms_thres
            )
            max_detections = detections_class[keep]
            output[i] = max_detections if output[i] is None else torch.cat((output[i], max_detections))

        if output[i] is not None:
            output[i]           = output[i].cpu().numpy()
            box_xy, box_wh      = (output[i][:, 0:2] + output[i][:, 2:4])/2, output[i][:, 2:4] - output[i][:, 0:2]
            output[i][:, :4]    = centernet_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    return output


def post_process_Ultra_Fast_Lane_Detection(out : np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Post-process the predicted output from the Ultra-Fast-Lane-Detection model to obtain the final lane coordinates.

    Args:
        out (numpy.ndarray): The raw model outputs.

    Returns:
        Tuple[numpy.ndarray, float]: A tuple containing the final lane coordinates and the column sample width.
    """
    griding_num = 100
    col_sample = np.linspace(0, 800 - 1, griding_num)
    col_sample_w = col_sample[1] - col_sample[0]
    out_j = out[0]
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(griding_num) + 1
    idx = idx.reshape(-1, 1, 1)
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == griding_num] = 0
    out_j = loc
    return out_j, col_sample_w