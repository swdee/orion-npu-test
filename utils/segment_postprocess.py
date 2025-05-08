# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""
This is the script of cix segmentation postprocess.
"""
import numpy as np
import cv2
from PIL import Image
from .label import cityscapes_labels


def decode_voc_segmap(image, nc=21):
    label_colors = np.array(
        [
            (0, 0, 0),  # 0=background
            (128, 0, 0),  # 1=aeroplane
            (0, 128, 0),  # 2=bicycle
            (128, 128, 0),  # 3=bird
            (0, 0, 128),  # 4=boat
            (128, 0, 128),  # 5=bottle
            (0, 128, 128),  # 6=bus
            (128, 128, 128),  # 7=car
            (64, 0, 0),  # 8=cat
            (192, 0, 0),  # 9=chair
            (64, 128, 0),  # 10=cow
            (192, 128, 0),  # 11=diningtable
            (64, 0, 128),  # 12=dog
            (192, 0, 128),  # 13=horse
            (64, 128, 128),  # 14=motorbike
            (192, 128, 128),  # 15=person
            (0, 64, 0),  # 16=pottedplant
            (128, 64, 0),  # 17=sheep
            (0, 192, 0),  # 18=sofa
            (128, 192, 0),  # 19=train
            (0, 64, 128),  # 20=tv/monitor
        ]
    )

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)
    for l in range(0, nc):
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb


def voc_cmap(N : int = 256, normalized : bool = False) -> np.ndarray:
    """
    Generate a VOC format color map.

    Args:
        N (int, optional):
            The number of colors to generate, default is 256.
        normalized (bool, optional):
            Specifies whether to normalize the output color values to [0, 1], default is False.

    Returns:
        np.ndarray:
            The generated color map, shape (N, 3), where each row represents an RGB color.
    """
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = "float32" if normalized else "uint8"
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def voc_inverse_map():
    """
    Generate an inverse color mapping from the VOC color map.

    Returns:
        dict: A dictionary where keys are color values (encoded as integers)
          and values are the corresponding indices of the colors in the VOC color map.
    """
    cmap = voc_cmap()
    inv_cmap = dict()
    for i in range(cmap.shape[0]):
        if i <= 20:
            inv_cmap[cmap[i][0] * 255 * 255 + cmap[i][1] * 255 + cmap[i][2]] = i
        else:
            inv_cmap[cmap[i][0] * 255 * 255 + cmap[i][1] * 255 + cmap[i][2]] = 0
    return inv_cmap


def masks2segments(
    masks : np.ndarray,
    ) -> list:
    """
    Convert binary masks to segmentation contours.

    Parameters:
        masks (numpy.ndarray):
            The input binary masks with shape (num_masks, height, width), where
            each entry represents a binary mask of an object.

    Returns:
        list
            A list of numpy arrays where each array contains the contour points for a 
            corresponding mask. Each contour is represented as an array of shape (N, 2).
    """
    segments = []
    for x in masks.astype("uint8"):
        # Find contours in the binary mask
        c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        if c:
            c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
        else:# no segments found
            c = np.zeros((0, 2))
        segments.append(c.astype("float32"))
    return segments


def crop_mask(
    masks : np.ndarray,
    boxes : np.ndarray,
    ) -> np.ndarray:
    """
    Crop the input masks based on the specified bounding boxes.

    Args:
        masks (numpy.ndarray):
            The input masks with shape (num_masks, height, width).
        boxes (numpy.ndarray):
            The bounding boxes for cropping, with shape (num_boxes, 4) where each box is 
            represented as (x1, y1, x2, y2).

    Returns:
        numpy.ndarray
            The cropped masks with the same shape as the input masks, where areas outside 
            the bounding boxes are set to zero.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


# Process the masks to the original image size and crop the mask to the bounding box.
def process_mask(
    protos : np.ndarray,
    masks_in : np.ndarray,
    bboxes : np.ndarray,
    im0_shape : tuple,
    ) -> np.ndarray:
    """
    Process the input masks to generate binary masks based on protos and bounding boxes.

    Args:
        protos (numpy.ndarray):
            The prototype masks with shape (channels, height, width) used for mask generation.
        masks_in (numpy.ndarray):
            The input masks which need to be processed, has shape (num_masks, num_protos).
        bboxes (numpy.ndarray):
            The bounding boxes for the detected objects, used to crop the masks.
        im0_shape (tuple):
            The shape of the original image in the format (height, width).

    Returns:
        numpy.ndarray
            A binary mask array indicating the presence of the masks over the objects.
    """
    c, mh, mw = protos.shape
    masks = (
        np.matmul(masks_in, protos.reshape((c, -1)))
        .reshape((-1, mh, mw))
        .transpose(1, 2, 0)
    )
    masks = np.ascontiguousarray(masks)
    # re-scale mask shape to original input image shape
    masks = scale_mask(
        masks, im0_shape
    )
    # HWN -> NHW
    masks = np.einsum("HWN -> NHW", masks)
    masks = crop_mask(masks, bboxes)
    return np.greater(masks, 0.5)


#Process the masks to the original image size.
def scale_mask(
    masks : np.ndarray,
    im0_shape : tuple,
    ratio_pad : tuple = None
    ) -> np.ndarray:
    """
    Scale and adjust the mask dimensions to match the original image shape.

    Args:
        masks (numpy.ndarray)
            The input masks to be scaled. Can be 2D or 3D (height, width, channels).
        im0_shape (tuple)
            The shape of the original image in the format (height, width).
        ratio_pad :(tuple, optional)
            The padding ratio to be applied, in the format (padding_x, padding_y).
            If None, padding is calculated based on the scaling gain.

    Returns:
        numpy.ndarray
            The scaled and adjusted masks with dimensions matching the original image.
    """
    im1_shape = masks.shape[:2]
    if ratio_pad is None:
        # gain  = old / new
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])
        # wh padding
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2
    else:
        pad = ratio_pad[1]
    # Calculate top left bottom and right of mask
    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))
    bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
    if len(masks.shape) < 2:
        raise ValueError(
            f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}'
        )
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR)
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks


# Postprocess the inference result
def postprocess_yolov8seg(
    preds : list,
    im0 : np.ndarray,
    ratio : float,
    pad_w : float,
    pad_h : float,
    conf_threshold : float,
    iou_threshold : float,
    nm : int = 32):
    """
    Post-process the predictions from the YOLOv8 segmentation model.

    Args:
        preds (list):
            A list containing two items: the prediction outputs and protos (masks).
        im0 (numpy.ndarray)
            The original image from which predictions were made, used for scaling boxes and masks.
        ratio (float)
            The ratio used to rescale the bounding boxes back to the original image dimensions.
        pad_w (float)
            The width padding applied when resizing the image to model input size.
        pad_h (float)
            The height padding applied when resizing the image to model input size.
        conf_threshold (float)
            The confidence threshold for filtering weak predictions.
        iou_threshold (float)
            The intersection-over-union threshold for non-maximum suppression.
        nm (int, optional):
            The number of masks produced by the model (default is 32).

    Returns:
        Tuple[numpy.ndarray, list, numpy.ndarray]:
            - x : The bounding boxes and corresponding class scores.
            - segments : The segmentation contours extracted from processed masks.
            - masks : The processed masks corresponding to detected objects.
    """
    # Two outputs: predictions and protos
    x, protos = preds[0], preds[1]
    # Transpose the first output: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
    x = np.einsum("bcn->bnc", x)
    # Predictions filtering by conf-threshold
    x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]
    # Create a new matrix which merge these(box, score, cls, nm) into one
    x = np.c_[
        x[..., :4],
        np.amax(x[..., 4:-nm], axis=-1),
        np.argmax(x[..., 4:-nm], axis=-1),
        x[..., -nm:],
    ]
    # NMS filtering
    x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]
    # Decode and return
    if len(x) > 0:
        # Bounding boxes format change: cxcywh -> xyxy
        x[..., [0, 1]] -= x[..., [2, 3]] / 2
        x[..., [2, 3]] += x[..., [0, 1]]
        # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
        x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
        x[..., :4] /= min(ratio)
        # Bounding boxes boundary clamp
        x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
        x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])
        # Process masks
        masks = process_mask(protos[0], x[:, 6:], x[:, :4], im0.shape)
        # Masks -> Segments(contours)
        segments = masks2segments(masks)
        return x[..., :6], segments, masks
    else:
        return [], [], []


def postprocess_fcn(outputs: np.ndarray) -> np.ndarray:
    """
    Post-process the predictions from the FCN segmentation model.

    Args:
        outputs (numpy.ndarray):
            The output of the FCN segmentation model, with shape (1, height, width, num_classes).

    Returns:
        numpy.ndarray:
            The segmentation image with shape (height, width, 3) in BGR format.
    """
    output_predictions = np.argmax(outputs[0], axis=0)
    palette = np.array([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1], dtype=np.int32)
    colors = np.arange(21)[:, None] * palette
    colors = (colors % 255).astype(np.uint8)

    segmentation_image = colors[output_predictions.flatten()]
    segmentation_image = segmentation_image.reshape((*output_predictions.shape, 3))
    segmentation_image_bgr = cv2.cvtColor(segmentation_image, cv2.COLOR_RGB2BGR)
    return segmentation_image_bgr


def get_palette() -> list:
    """
    Get the color palette for the Cityscapes dataset.

    Returns:
        list:
            A list of RGB values representing the color palette for the Cityscapes dataset.
    """
    trainId2colors = {label.trainId: label.color for label in cityscapes_labels.labels}
    # prepare and return palette
    palette = [0] * 256 * 3
    for trainId in trainId2colors:
        colors = trainId2colors[trainId]
        if trainId == 255:
            colors = (0, 0, 0)
        for i in range(3):
            palette[trainId * 3 + i] = colors[i]
    return palette

def colorize(labels : np.ndarray) -> np.ndarray:
    """
    Colorize the output labels using the Cityscapes color palette.

    Args:
        labels (numpy.ndarray):
            The output labels with shape (height, width) or (height, width, 1).

    Returns:
        numpy.ndarray:
            The colorized image with shape (height, width, 3) in RGB format.
    """
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert('P')
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))

def postprocess_duc(output : np.ndarray,im : np.ndarray,result_shape : tuple) -> tuple:
    """
    Post-process the predictions from the DUC segmentation model.

    Args:
        output (numpy.ndarray):
            The output of the DUC segmentation model.
        im (numpy.ndarray):
            The original image from which predictions were made, used for scaling boxes and masks.
        result_shape (tuple):
            The shape of the output image.

    Returns:
        tuple:
            - result_img : The segmentation image with shape (height, width, 3) in BGR format.
    """
    # get input and output dimensions
    result_height, result_width = result_shape
    img_height, img_width = 800, 800
    ds_rate = 8
    cell_width = 2
    label_num = 19
    labels = output.squeeze()
    # re-arrange output
    test_width = int((int(img_width) / ds_rate) * ds_rate)
    test_height = int((int(img_height) / ds_rate) * ds_rate)
    feat_width = int(test_width / ds_rate)
    feat_height = int(test_height / ds_rate)
    labels = labels.reshape((label_num, 4, 4, feat_height, feat_width))
    labels = np.transpose(labels, (0, 3, 1, 4, 2))
    labels = labels.reshape((label_num, int(test_height / cell_width), int(test_width / cell_width)))

    labels = labels[:, :int(img_height / cell_width),:int(img_width / cell_width)]
    labels = np.transpose(labels, [1, 2, 0])
    labels = cv2.resize(labels, (result_width, result_height), interpolation=cv2.INTER_LINEAR)
    labels = np.transpose(labels, [2, 0, 1])
    softmax = labels
    # get classification labels
    results = np.argmax(labels, axis=0).astype(np.uint8)
    raw_labels = results
    # comput confidence scores
    result_img = colorize(raw_labels)
    result_img = cv2.resize(result_img, (result_width, result_height), interpolation=cv2.INTER_LINEAR)
    blended_img = cv2.addWeighted(im[:, :, ::-1], 0.5, result_img, 0.5, 0)
    return result_img, blended_img