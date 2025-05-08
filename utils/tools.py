# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
"""
This script mainly includes some file processing and some functions.
"""
import os
import numpy as np
from typing import List, Union, Tuple
import torch


def get_file_list(file_path: str):
    """
    Given a path, determine if it's a file or a directory.
    Return a list of file paths.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"}

    # Helper function to check if a file is an image
    def is_image_file(file_name):
        _, ext = os.path.splitext(file_name)
        return ext.lower() in image_extensions

    if os.path.isfile(file_path):
        if is_image_file(file_path):
            return [file_path]
    # If the path is a directory, gather all image file paths
    elif os.path.isdir(file_path):
        file_list = []
        for root, dirs, files in os.walk(file_path):
            for file in files:
                if is_image_file(file):
                    file_list.append(os.path.join(root, file))
        return file_list
    else:
        return []


def get_all_image_files(directory: str):
    """
    Retrieve a list of full paths to all image files in the specified directory and its subdirectories.

    Parameters:
    directory (str): The root directory to search within.

    Returns:
    list: A list containing the full paths of all image files found.
    """
    # Supported image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}
    # List to store image file paths
    image_files = []

    # Walk through all folders and subfolders in the directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has a supported image extension
            if os.path.splitext(file)[1].lower() in image_extensions:
                # Add the full path to the list
                image_files.append(os.path.join(root, file))

    return image_files


def get_all_files(directory: str, extensions: List[str]) -> List[str]:
    """
    Retrieve a list of full paths to files in the specified directory and its subdirectories.
    If extensions are specified, only files with those extensions are included.

    Parameters:
    directory (str): The root directory to search within.
    extensions (list, optional): A list of file extensions to filter by (e.g., [".jpg", ".png"]).

    Returns:
    list: A list containing the full paths of all files found, filtered by the given extensions if provided.
    """

    # Convert extensions to lowercase if specified, for case-insensitive matching
    if extensions is not None:
        extensions = set(ext.lower() for ext in extensions)

    # Initialize list to store file paths
    found_files = []

    # Walk through all folders and subfolders in the directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Get the file's extension in lowercase
            file_extension = os.path.splitext(file)[1].lower()

            # If no extensions specified, add all files
            if extensions is None or file_extension in extensions:
                # Add the full path of the file to the list
                found_files.append(os.path.join(root, file))

    return found_files  # Return the list of found file paths


def cosine_similarity(a, b):
    a_float = a.astype(np.float32)
    b_float = b.astype(np.float32)
    dot_product = np.dot(a_float, b_float)
    norm_a = np.linalg.norm(a_float)
    norm_b = np.linalg.norm(b_float)
    cos_sim = dot_product / (norm_a * norm_b)
    return cos_sim


def load_image_labels(image_dir):
    image_paths = []
    labels = []
    class_names = sorted(os.listdir(image_dir))
    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(image_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith(".JPEG"):
                image_paths.append(os.path.join(class_dir, file_name))
                labels.append(class_to_index[class_name])

    return image_paths, labels, class_to_index


def match_image_pairs(
    file_list: List[str], scale: Union[str, int]
) -> List[Tuple[str, str]]:
    """
    Select image pairs according to scale
    """
    pairs = []
    for file in file_list:
        if f"_scale_{scale}" in file:
            original_file = file.replace(f"_scale_{scale}", "")
            if original_file in file_list:
                pairs.append((original_file, file))
    return pairs


def cal_cos_sim(text_features: np.ndarray, image_features: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between text features and image features
    Args:
        text_features: (np.ndarray) text features
        image_features: (np.ndarray) image features
    Returns:
        cos_sim: (np.ndarray) cosine similarity between text features and image features
    """
    text_features = torch.Tensor(text_features)
    image_features = torch.Tensor(image_features)
    text_features = text_features.reshape(1, -1)
    cos_sim = torch.cosine_similarity(image_features, text_features)
    return cos_sim
