# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
import cv2
import copy
import torch
import numpy as np
from PIL import Image
import math
from torchvision import transforms
from typing import Union, Tuple
from imageio import imread


def imagenet_transforms(size: int):
    """
    Creates a preprocessing pipeline for images, commonly used for ImageNet models.

    Preprocessing Steps:
    1. Resize: Adjusts the smaller side of the image to `size + 32` pixels, maintaining the aspect ratio.
    2. CenterCrop: Crops the center of the image to the target size `[size, size]`.
    3. ToTensor: Converts the image to a PyTorch tensor and scales pixel values to the range [0, 1].
    4. Normalize: Standardizes the image by subtracting the channel-wise mean and dividing by the standard deviation.
       - Mean: [0.485, 0.456, 0.406]
       - Std: [0.229, 0.224, 0.225]
    Args:
        size (int): The target size of the image after cropping (both height and width).

    Returns:
        torchvision.transforms.Compose: A composed transformation function for preprocessing.
    """
    transforms_ = transforms.Compose(
        [
            transforms.Resize(size + 32),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms_


def open_image_with_PIL(
    src_image: Union[Image.Image, str],
) -> Image.Image:
    """
    Opens or processes an image using PIL (Pillow).

    This function takes either a file path or an existing PIL.Image object as input
    and returns a PIL.Image object. If the input is a string (assumed to be a file path),
    it opens the image file. If the input is already a PIL.Image.Image object, it
    returns it unchanged.

    Args:
        src_image (Union[Image.Image, str]):
            - A string representing the file path to the image.
            - An instance of PIL.Image.Image.

    Returns:
        Image.Image: A PIL.Image.Image object representing the loaded image.
    """
    # Open the image if `src_image` is a file path
    if isinstance(src_image, str):
        image = Image.open(src_image)
    # if `src_image` is already an Image
    elif isinstance(src_image, Image.Image):
        image = src_image
    else:
        assert TypeError()
    return image


def open_image_with_imageio_YCbCr(src_image: Union[str, bytes]) -> np.ndarray:
    """
    Open an image using `imageio.imread` and load it in YCbCr color space.

    Args:
        src_image (Union[str, bytes]): Path to the image file or a file-like object.

    Returns:
        np.ndarray: Image in YCbCr color space, shape (H, W, 3), dtype uint8.
    """
    # Read the image in YCbCr color space
    image = imread(src_image, pilmode="YCbCr")
    return image


def colorize(y: np.ndarray, ycbcr: np.ndarray) -> np.ndarray:
    """
    Combine the luminance (Y) channel with chrominance (CbCr) channels
    to create a colorized image in YCbCr format.

    Args:
        y (np.ndarray): Luminance channel, shape (H, W), dtype uint8.
        ycbcr (np.ndarray): Full YCbCr image, shape (H, W, 3), dtype uint8.

    Returns:
        np.ndarray: Colorized image in YCbCr format, shape (H, W, 3), dtype uint8.
    """
    # Validate input dimensions
    if y.ndim != 2:
        raise ValueError("The Y channel must be a 2D array.")
    if ycbcr.shape[:2] != y.shape or ycbcr.shape[2] != 3:
        raise ValueError(
            "The YCbCr array must have shape (H, W, 3) and match the dimensions of Y."
        )

    # Combine Y and CbCr into a single YCbCr image
    img = np.zeros_like(ycbcr, dtype=np.uint8)
    img[:, :, 0] = y
    img[:, :, 1:] = ycbcr[:, :, 1:]
    return img


def imagenet_preprocess_method1(
    src_image: Union[Image.Image, str],
    input_size: int = 224,
    return_type: str = "np",
) -> Union[np.ndarray, torch.Tensor]:
    """
    Preprocess an input image for ImageNet-style models.

    Args:
        src_image (Union[Image, str]): The input image to preprocess.
            - If it's a string, it's treated as the file path to the image.
            - If it's an Image object, it's processed directly.
        input_size (int, optional): The target size for the image after preprocessing.
            Defaults to 224 (standard for many ImageNet models).
        return_type (str, optional): Determines the return format of the processed image.
            - "np": Returns the processed image as a NumPy array.
            - "pt": Returns the processed image as a PyTorch tensor.
            Defaults to "np".

    Returns:
        Union[torch.Tensor, np.ndarray]: The preprocessed image.
            - If `return_type` is "pt", returns a PyTorch tensor.
            - If `return_type` is "np", returns a NumPy array.

    """
    image = open_image_with_PIL(src_image)
    image = image.convert("RGB")
    transforms_ = imagenet_transforms(input_size)
    image = transforms_(image)
    image = torch.unsqueeze(image, dim=0)
    if return_type == "pt":
        return image  # Return as a PyTorch tensor
    elif return_type == "np":
        return image.cpu().numpy()  # Convert to NumPy array and return
    else:
        raise TypeError(f"Type {return_type} not support.")


def normalize_image(
    data: np.ndarray,
) -> np.ndarray:
    """
    Normalizes an image represented as a NumPy array.

    This function scales the pixel values of an image from the range [0, 255]
    to the range [0, 1].

    Args:
        data (np.ndarray):
            A NumPy array representing the image, typically with pixel values
            in the range [0, 255]. The array shape can be:
            - (H, W): For grayscale images.
            - (H, W, C): For RGB or multi-channel images.

    Returns:
        np.ndarray:
            A NumPy array of the same shape as the input, with pixel values
            normalized to the range [0, 1].

    Example:
        >>> import numpy as np
        >>> image = np.array([[0, 128, 255], [64, 192, 32]], dtype=np.uint8)
        >>> normalize_image(image)
        array([[0.    , 0.502 , 1.    ],
               [0.251 , 0.753 , 0.125 ]], dtype=float32)
    """
    data = data / 255
    return data


def normalize_image_with_mean_stand(
    data: np.ndarray,
    mean: Union[float, int, np.ndarray],
    std: Union[float, int, np.ndarray],
) -> np.ndarray:
    """
    Normalizes an image with mean and standard deviation.

    This function normalizes image pixel values in the following steps:
    1. Scales the pixel values from [0, 255] to [0, 1].
    2. Scales the values back to [0, 256].
    3. Applies normalization using the provided mean and standard deviation:
       \[
       \text{normalized\_data} = \frac{\text{data} * 256 - \text{mean}}{\text{std}}
       \]

    Args:
        data (np.ndarray):
            The input image as a NumPy array with pixel values in the range [0, 255].
            The array can be:
            - (H, W): For grayscale images.
            - (H, W, C): For RGB or multi-channel images.
        mean (Union[float, int, np.ndarray]):
            The mean value(s) used for normalization.
            - Can be a single scalar for all channels.
            - Or an array of shape matching the image's channel dimension.
        std (Union[float, int, np.ndarray]):
            The standard deviation value(s) used for normalization.
            - Can be a single scalar for all channels.
            - Or an array of shape matching the image's channel dimension.

    Returns:
        np.ndarray:
            A normalized image with pixel values standardized using the provided mean
            and standard deviation. The output has the same shape as the input, with
            data type `float32`.
    """
    data = data / 255
    data = (data * 256 - mean) / std
    data = data.astype(np.float32)
    return data


def preprocess_object_detect_method1(
    src_image: Union[Image.Image, str],
    target_size=Tuple[int, int],
    mode: str = "RGB",
) -> Tuple[Tuple[int, int], Tuple[int, int], np.ndarray, np.ndarray]:
    """
    Preprocesses an input image for object detection models.

    This function performs the following steps:
    1. Loads the input image, either from a file path or as a PIL.Image.
    2. Converts the image to RGB format.
    3. Optionally converts the image to BGR format if specified.
    4. Resizes the image to the target dimensions.
    5. Normalizes the pixel values to the range [0, 1].
    6. Transposes the image dimensions to (C, H, W) and adds a batch axis (1, C, H, W).

    Args:
        src_image (Union[Image.Image, str]):
            The input image, which can be:
            - A file path (str) pointing to the image.
            - A PIL.Image.Image object.
        target_size (Tuple[int, int]):
            The target dimensions of the image after resizing, specified as (width, height).
        mode (str):
            The desired color mode of the output image ("RGB" or "BGR"). Default is "RGB".

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int], np.ndarray, np.ndarray]:
            - `src_shape`: The original dimensions of the input image (height, width).
            - `new_shape`: The dimensions of the resized image (height, width).
            - `show_image`: A copy of the original image for visualization or debugging.
            - `image`: The preprocessed image as a NumPy array with shape (1, C, H, W).
    """
    image = open_image_with_PIL(src_image)
    image = np.array(image.convert("RGB"))

    if mode == "BGR":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    show_image = copy.deepcopy(image)
    src_shape = show_image.shape[:2]
    new_shape = target_size
    image = cv2.resize(image, dsize=new_shape)

    # Normalize the image pixel values to the range [0, 1]
    image = normalize_image(image).astype(np.float32)
    image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
    return src_shape, new_shape, show_image, image


# this method for yolox
def preprocess_object_detect_method2(
    src_image: Union[Image.Image, str, np.ndarray],
    target_size=Tuple[int, int],
    mode: str = "RGB",
) -> Tuple[Tuple[int, int], Tuple[int, int], np.ndarray, np.ndarray]:
    """
    Preprocesses an image for YOLOX object detection models.

    This function prepares an input image for YOLOX models by performing the following steps:
    - Loads the image if it's a file path or PIL.Image.
    - Converts the image to RGB or BGR mode as specified.
    - Resizes the image to the specified target size.
    - Converts the image to a NumPy array with a shape suitable for deep learning models.

    Args:
        src_image (Union[Image.Image, str, np.ndarray]):
            The input image to preprocess. Can be:
            - A file path (str).
            - A PIL.Image object.
            - A NumPy array.
        target_size (Tuple[int, int]):
            The desired size of the image after resizing, specified as (width, height).
        mode (str):
            The desired color mode of the output image ("RGB" or "BGR"). Default is "RGB".

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int], np.ndarray, np.ndarray]:
            - `src_shape`: Original image dimensions as (height, width).
            - `new_shape`: Resized image dimensions as (height, width).
            - `show_image`: A copy of the original image for visualization or debugging.
            - `image`: The preprocessed image as a NumPy array with shape (1, C, H, W).

    Example:
        >>> src_image = "example.jpg"
        >>> target_size = (640, 640)
        >>> src_shape, new_shape, show_image, image = preprocess_object_method2(src_image, target_size, mode="BGR")
    """

    if isinstance(src_image, str) or isinstance(src_image, Image.Image):
        image = open_image_with_PIL(src_image)
        image = np.array(image.convert("RGB"))

        if mode == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif isinstance(src_image, np.ndarray):
        image = src_image
    else:
        raise TypeError(
            "src_image must be a file path (str), a NumPy array, or a PIL Image."
        )

    show_image = copy.deepcopy(image)
    src_shape = show_image.shape[:2]
    new_shape = target_size
    image = cv2.resize(image, dsize=new_shape)
    image = np.expand_dims(image.transpose(2, 0, 1), axis=0).astype(np.float32)
    return src_shape, new_shape, show_image, image


def preprocess_object_detect_method3(
    src_image: str,
    target_size=Tuple[int, int],
    mode: str = "RGB",
):
    """
    Preprocess an image for object detection using Method 3 (normalize_image_with_mean_stand).
    mean = 128, std = 128,
    data = data / 255,
    data = (data * 256 - mean) / std.

    Args:
        src_image (Union[str, np.ndarray, Image.Image]):
            The input image, which can be a file path (str), a NumPy array, or a PIL Image.
        target_size (Tuple[int, int]):
            The target size (width, height) to resize the image.
        mode (str, optional):
            The desired color mode, either "RGB" or "BGR". Default is "RGB".

    Returns:
        Tuple[Tuple[int, int], Tuple[int, int], np.ndarray, np.ndarray]:
            - src_shape: Original image shape as (height, width).
            - new_shape: Resized image shape as (height, width).
            - show_image: A copy of the processed image for visualization.
            - image: Preprocessed image ready for model input, normalized and transposed.
    """
    if isinstance(src_image, str) or isinstance(src_image, Image.Image):
        image = open_image_with_PIL(src_image)
        image = np.array(image.convert("RGB"))

        if mode == "BGR":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif isinstance(src_image, np.ndarray):
        image = src_image
    else:
        raise TypeError(
            "src_image must be a file path (str), a NumPy array, or a PIL Image."
        )

    show_image = copy.deepcopy(image)
    src_shape = show_image.shape[:2]
    mean = 128
    std = 128
    new_shape = target_size
    image = cv2.resize(image, target_size)
    image = normalize_image_with_mean_stand(image, mean, std)
    image = np.expand_dims(image, axis=0).transpose(0, 3, 1, 2)
    return src_shape, new_shape, show_image, image


# Preprocess the image to meet the input size of the model
def preprocess_yolov8seg(
    image_path : str,
    ndtype : np.dtype,
    model_height : int = 640,
    model_width : int = 640):
    """
    Preprocess the input image for model inference.

    Args:
        image_path (str):
            The path to the input image that needs to be processed.
        ndtype (numpy.dtype):
            The desired data type for the output image array (e.g., np.float32).
        model_height (int, optional)
            The target height for the model input (default is 640).
        model_width (int, optional):
            The target width for the model input (default is 640).

    Returns:
        Tuple[numpy.ndarray, Tuple, Tuple[int, int]]:
            - img_process : The processed image ready for model inference, with shape adjusted and normalized.
            - ratio : The scaling ratio applied to the original image dimensions.
            - padding : The padding applied to the width and height of the image.
    """
    img = cv2.imread(image_path)
    # original image shape
    shape = img.shape[:2]
    new_shape = (model_height, model_width)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    # wh padding
    pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (
        new_shape[0] - new_unpad[1]
    ) / 2
    # resize
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
    img = (
        np.ascontiguousarray(np.einsum("HWC->CHW", img)[::-1], dtype=ndtype)
        / 255.0
    )
    img_process = img[None] if len(img.shape) == 3 else img
    return img_process, ratio, (pad_w, pad_h)


def preprocess_image_deeplabv3(image_path : str,  mean : list = [0.485, 0.456, 0.406], std : list = [0.229, 0.224, 0.225], target_size : tuple = (520, 520), flag : bool = True) -> np.ndarray:
    """
    Preprocess an input image for the DeepLabV3 model.

    This function reads an image from the specified path, resizes it to the target size, normalizes,
    and standardizes the pixel values before converting it into a tensor suitable for input into the model.

    Args:
        image_path (str): The file path to the input image.
        mean (list, optional): The mean values to use for normalization. Default is [0.485, 0.456, 0.406].
        std (list, optional): The standard deviation values to use for normalization. Default is [0.229, 0.224, 0.225].
        target_size (tuple): The desired size to resize the image to, specified as (height, width).
                            Default is (520, 520).
        flag (bool, optional): A flag to indicate whether to normalize the pixel values or not. Default is True.

    Returns:
        numpy.ndarray: A tensor representation of the processed image, with shape (1, C, H, W),
                    where C is the number of channels, H is height, and W is width.
    """
    mean = np.array(mean).astype(np.float32)
    std = np.array(std).astype(np.float32)
    image = cv2.imread(image_path)
    image = image[:, :, ::-1]  # BGR2RGB
    image_resized = cv2.resize(image, target_size)
    if flag:
        image_normalized = image_resized.astype(np.float32) / 255.0
    else:
        image_normalized = image_resized.astype(np.float32)
    image_standardized = (image_normalized - mean) / std
    image_transposed = image_standardized.transpose(2, 0, 1)
    input_tensor = np.expand_dims(image_transposed, axis=0)
    return input_tensor


def preprocess_image_yolo_v3(image : np.ndarray, target_size : tuple = (640, 640)):
    """
    Preprocess an input image for the YOLOv3 model.

    This function resizes the input image to the target size, normalizes the pixel values, 
    and prepares the image tensor for input into the YOLOv3 model.

    Parameters:
        image (numpy.ndarray): The input image to preprocess. It is expected to be in the format 
                            (height, width, channels).
        target_size (tuple): The desired size to resize the image to, specified as (height, width).
                            Default is (640, 640).

    Returns:
    tuple: A tuple containing:
        - old_shape (tuple): The original shape of the image as (height, width, channels).
        - new_shape (tuple): The new shape of the resized image as (height, width, channels).
        - show_data (numpy.ndarray): A deep copy of the original image for visualization purposes.
        - data (numpy.ndarray): The processed image tensor formatted as (1, channels, height, width) 
                                 and normalized to [0, 1] range.
    """
    data = image
    old_shape = data.shape
    show_data = copy.deepcopy(data)
    data = cv2.resize(data, target_size)
    new_shape = data.shape
    data = data.astype(np.float32).transpose(2, 0, 1)
    data = np.expand_dims(data, axis=0)

    data = data / 255.0
    return old_shape, new_shape, show_data, data


def pad_width(
    img : np.ndarray,
    stride : int,
    pad_value : tuple,
    min_dims : list,
    ) -> Tuple[np.ndarray, list]:
    """
    Pad the input image to the specified dimensions, ensuring that the dimensions are
    multiples of the given stride.

    Args:
        img (numpy.ndarray):
            The input image to be padded, with shape (height, width, channels).
        stride (int):
            The stride value used to ensure the dimensions are multiples of this value.
        pad_value (tuple):
            The value to use for padding the borders of the image.
        min_dims (list):
            A list containing the minimum target dimensions for padding in the format [min_height, min_width].

    Returns:
        Tuple[np.ndarray, list]:
            - padded_img : The padded image with shape adjusted to meet the minimum dimensions.
            - pad : A list containing the number of pixels added to each side of the image in the order [top, left, bottom, right].
    """
    h, w, _ = img.shape
    h = min(min_dims[0], h)
    min_dims[0] = math.ceil(min_dims[0] / float(stride)) * stride
    min_dims[1] = max(min_dims[1], w)
    min_dims[1] = math.ceil(min_dims[1] / float(stride)) * stride
    pad = []
    pad.append(int(math.floor((min_dims[0] - h) / 2.0)))
    pad.append(int(math.floor((min_dims[1] - w) / 2.0)))
    pad.append(int(min_dims[0] - h - pad[0]))
    pad.append(int(min_dims[1] - w - pad[1]))
    padded_img = cv2.copyMakeBorder(
        img, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=pad_value
    )
    return padded_img, pad


def preprocess_openpose(
    img : np.ndarray,
    net_input_height_size : int,
    stride : int = 8,
    img_mean : np.ndarray = np.array([128, 128, 128],np.float32),
    pad_value : tuple = (0, 0, 0),
    img_scale : float =np.float32(1 / 256),
    ) -> Tuple[np.ndarray, float, list]:
    """
    Preprocess the input image for model inference, including resizing, normalization,
    and padding.

    Args:
        img (numpy.ndarray):
            The input image to be preprocessed, with shape (height, width, channels).
        net_input_height_size (int):
            The target height for the input image to the neural network.
        stride (int, optional):
            The stride used in the model for padding calculations (default is 8).
        img_mean (numpy.ndarray, optional):
            The mean values to be subtracted from the image during normalization (default is [128, 128, 128]).
        pad_value (tuple, optional):
            The padding values to use for edges of the image (default is (0, 0, 0)).
        img_scale (float, optional):
            The scaling factor to apply to the normalized image (default is 1/256).

    Returns:
        Tuple[np.ndarray, float, list]:
            - tensor_img : The preprocessed image as a tensor with shape (1, channels, height, width).
            - scale : The scaling factor applied to the image.
            - pad : The padding applied to the image.
    """
    height, width, _ = img.shape
    scale = net_input_height_size / height
    scaled_img = cv2.resize(
        img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR
    )
    scaled_img = np.array(scaled_img, dtype=np.float32)
    scaled_img = (scaled_img - img_mean) * img_scale
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)
    tensor_img = padded_img.transpose(2, 0, 1)
    tensor_img = np.expand_dims(tensor_img, axis=0).astype(np.float32)
    return tensor_img, scale, pad


def preprocess_image_centerface(img : np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Preprocess an input image for the CenterFace model.

    This function resizes the input image while maintaining the aspect ratio, creates a 
    zero-padded image, and prepares a blob for input into the CenterFace model.

    Args:
        img (numpy.ndarray): The input image to preprocess, expected in the format 
                            (height, width, channels).

    Returns:
        Tuple[np.ndarray, float]:
            - blob: The preprocessed image blob suitable for input into the model.
            - det_scale : The scaling factor used to resize the original image to the input size.
    """
    height = 640
    width = 640
    mean = 127.5
    std = 128.0
    im_ratio = float(img.shape[0]) / img.shape[1]
    model_ratio = height / width
    if im_ratio > model_ratio:
        new_height = height
        new_width = int(new_height / im_ratio)
    else:
        new_width = width
        new_height = int(new_width * im_ratio)

    resized_image = cv2.resize(img, (new_width, new_height))
    det_scale = float(new_width) / img.shape[1]
    det_image = np.zeros((height, width, 3), dtype=np.uint8)
    det_image[:new_height, :new_width, :] = resized_image

    blob = cv2.dnn.blobFromImage(
        det_image,
        scalefactor=1.0,
        size=(height, width),
        mean=(mean, mean, mean),
        swapRB=True,
        crop=False,
    )
    return blob, det_scale


def preprocess_image(image_path, input_size=224, input_ch_type="nchw"):
    """
    Preprocess an image for model inference.
    Args:
        image_path: Path to the image file
        input_size: Target size for image resizing (default: 224x224)
        input_ch_type: Channel type, either 'nchw' or 'nhwc' (default: 'nchw')
    Returns:
        Preprocessed image data ready for inference
    """
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    image = Image.open(image_path).resize((input_size, input_size))
    image = np.array(image)
    img_data = image.astype("float32")

    # Normalize the image by subtracting the mean pixel values
    img_data[:, :, 0] = img_data[:, :, 0] - _R_MEAN
    img_data[:, :, 1] = img_data[:, :, 1] - _G_MEAN
    img_data[:, :, 2] = img_data[:, :, 2] - _B_MEAN

    # Add an extra dimension to represent the batch size (N, C, H, W)
    img_data = np.expand_dims(img_data, axis=0)  # N, H, W, C
    img_data = img_data.transpose(0, 3, 1, 2)  # N, C, H, W
    return img_data


def preprocess_duc(im : np.ndarray) -> np.ndarray:
    """
    Preprocessing function for DUC
    Args:
        im (numpy.ndarray): input image
    Returns:
        numpy.ndarray: preprocessed image
    """
    rgb_mean = cv2.mean(im)
    # Convert to float32
    test_img = im.astype(np.float32)
    # Extrapolate image with a small border in order obtain an accurate reshaped image after DUC layer
    test_shape = [im.shape[0],im.shape[1]]
    test_img = cv2.resize(test_img, (800,800))
    cell_shapes = [math.ceil(l / 8)*8 for l in test_shape]
    test_img = cv2.copyMakeBorder(test_img, 0, max(0, int(cell_shapes[0]) - im.shape[0]), 0, max(0, int(cell_shapes[1]) - im.shape[1]), cv2.BORDER_CONSTANT, value=rgb_mean)
    test_img = np.transpose(test_img, (2, 0, 1))
    # subtract rbg mean
    for i in range(3):
        test_img[i] -= rgb_mean[i]
    test_img = np.expand_dims(test_img, axis=0)
    return test_img


def process_crnn(img_path : str) -> np.ndarray:
    """
    Preprocess an input image for the CRNN model.

    This function reads an image from the specified path, resizes it to the target size, normalizes,
    and standardizes the pixel values before converting it into a tensor suitable for input into the model.

    Args:
        img_path (str): The file path to the input image.

    Returns:
        numpy.ndarray: A tensor representation of the processed image, with shape (1, C, H, W),
                    where C is the number of channels, H is height, and W is width.
    """
    img = Image.open(img_path).convert('L')
    img = img.resize((100, 32), Image.BILINEAR)
    img = np.array(img).astype(np.float32)
    img = img / 255.0
    img = (img - 0.5)/0.5
    img = img.transpose((0,1))
    img = np.expand_dims(img, axis=0)
    return img
# -------------------------(1)--------------------------


def preprocess_for_pytorch_standard(image_path, input_size=224):
    # Load image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_size + 32, input_size + 32))

    # Center crop to 224x224
    width, height = image.size
    new_width, new_height = input_size, input_size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    image = image.crop((left, top, right, bottom))
    image = image.resize((input_size, input_size))
    # Convert to numpy array and scale pixel values to [0, 1]
    image_np = np.array(image) / 255.0

    # Normalize with mean and standard deviation
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_np = ((image_np - mean) / std).astype(np.float32)

    # Rearrange dimensions to match PyTorch's (C, H, W) format
    image_np = image_np.transpose((2, 0, 1))
    image_np = np.expand_dims(image_np, axis=0)

    return image_np


def preprocess_image_caffe2(image_path):
    image = Image.open(image_path).convert("RGB")
    # Load image in BGR format
    image = np.array(image)
    image = image[:, :, ::-1]  # BGR

    # Resize so that the shorter side is 256 pixels, keeping the aspect ratio
    height, width = image.shape[:2]
    if width < height:
        new_width = 256
        new_height = int(height * (256 / width))
    else:
        new_height = 256
        new_width = int(width * (256 / height))
    image = cv2.resize(image, (new_width, new_height))

    # Center crop to 224x224
    start_x = (new_width - 224) // 2
    start_y = (new_height - 224) // 2
    image = image[start_y : start_y + 224, start_x : start_x + 224]

    # Convert to float32 and subtract mean values (BGR)
    image = image.astype(np.float32)
    mean = np.array([104, 117, 123])  # BGR mean values
    image -= mean

    # Transpose to CHW format, as required by Caffe2 (channels first)
    image = image.transpose((2, 0, 1))  # Convert HWC to CHW
    image = np.expand_dims(image, axis=0)
    return image

# -------------(2)--------------
#preprocess for depth_MiDaS_v2
def constrain_to_multiple_of(x, multiple_of=1, min_val=0, max_val=None):
    """Ensure that x is a multiple of `multiple_of` with optional constraints."""
    y = (np.round(x / multiple_of) * multiple_of).astype(int)

    if max_val is not None and y > max_val:
        y = (np.floor(x / multiple_of) * multiple_of).astype(int)

    if y < min_val:
        y = (np.ceil(x / multiple_of) * multiple_of).astype(int)

    return y

def get_depth_size(width, height, target_width, target_height, keep_aspect_ratio, resize_method, ensure_multiple_of=1):
    """Determine the new size considering the aspect ratio and constraints."""
    scale_height = target_height / height
    scale_width = target_width / width

    if keep_aspect_ratio:
        if resize_method == "lower_bound":
            if scale_width > scale_height:
                scale_height = scale_width
            else:
                scale_width = scale_height
        elif resize_method == "upper_bound":
            if scale_width < scale_height:
                scale_height = scale_width
            else:
                scale_width = scale_height
        elif resize_method == "minimal":
            if abs(1 - scale_width) < abs(1 - scale_height):
                scale_height = scale_width
            else:
                scale_width = scale_height
        else:
            raise ValueError(f"resize_method {resize_method} not implemented")

    if resize_method == "lower_bound":
        new_height = constrain_to_multiple_of(scale_height * height, multiple_of=ensure_multiple_of, min_val=target_height)
        new_width = constrain_to_multiple_of(scale_width * width, multiple_of=ensure_multiple_of, min_val=target_width)
    elif resize_method == "upper_bound":
        new_height = constrain_to_multiple_of(scale_height * height, multiple_of=ensure_multiple_of, max_val=target_height)
        new_width = constrain_to_multiple_of(scale_width * width, multiple_of=ensure_multiple_of, max_val=target_width)
    elif resize_method == "minimal":
        new_height = constrain_to_multiple_of(scale_height * height, multiple_of=ensure_multiple_of)
        new_width = constrain_to_multiple_of(scale_width * width, multiple_of=ensure_multiple_of)
    else:
        raise ValueError(f"resize_method {resize_method} not implemented")

    return new_width, new_height

def depth_resize_sample(sample, target_width, target_height, resize_target=True, keep_aspect_ratio=False, resize_method="lower_bound", image_interpolation_method=cv2.INTER_AREA, ensure_multiple_of=1):
    """Resize the sample (image, mask, etc.) to the target width and height."""
    width, height = get_depth_size(sample["image"].shape[1], sample["image"].shape[0], target_width, target_height, keep_aspect_ratio, resize_method, ensure_multiple_of)

    # Resize image
    sample["image"] = cv2.resize(sample["image"], (width, height), interpolation=image_interpolation_method)

    if resize_target:
        if "disparity" in sample:
            sample["disparity"] = cv2.resize(sample["disparity"], (width, height), interpolation=cv2.INTER_NEAREST)
        if "depth" in sample:
            sample["depth"] = cv2.resize(sample["depth"], (width, height), interpolation=cv2.INTER_NEAREST)

        sample["mask"] = cv2.resize(sample["mask"].astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
        sample["mask"] = sample["mask"].astype(bool)

    return sample

def prepare_depth_net(sample):
    """Prepare sample for network input (transpose and make contiguous)."""
    sample["image"] = np.transpose(sample["image"], (2, 0, 1))
    sample["image"] = np.ascontiguousarray(sample["image"]).astype(np.float32)

    if "mask" in sample:
        sample["mask"] = sample["mask"].astype(np.float32)
        sample["mask"] = np.ascontiguousarray(sample["mask"])

    if "disparity" in sample:
        sample["disparity"] = sample["disparity"].astype(np.float32)
        sample["disparity"] = np.ascontiguousarray(sample["disparity"])

    if "depth" in sample:
        sample["depth"] = sample["depth"].astype(np.float32)
        sample["depth"] = np.ascontiguousarray(sample["depth"])

    return sample
#preprocess for depth_MiDaS_v2 end