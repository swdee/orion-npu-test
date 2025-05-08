# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------

import numpy as np
import math
from operator import itemgetter
import cv2

BODY_PARTS_KPT_IDS = [
    [1, 2],
    [1, 5],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [1, 8],
    [8, 9],
    [9, 10],
    [1, 11],
    [11, 12],
    [12, 13],
    [1, 0],
    [0, 14],
    [14, 16],
    [0, 15],
    [15, 17],
    [2, 16],
    [5, 17],
]
BODY_PARTS_PAF_IDS = (
    [12, 13],
    [20, 21],
    [14, 15],
    [16, 17],
    [22, 23],
    [24, 25],
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [28, 29],
    [30, 31],
    [34, 35],
    [32, 33],
    [36, 37],
    [18, 19],
    [26, 27],
)

class Pose:
    num_kpts = 18
    kpt_names = [
        "nose",
        "neck",
        "r_sho",
        "r_elb",
        "r_wri",
        "l_sho",
        "l_elb",
        "l_wri",
        "r_hip",
        "r_knee",
        "r_ank",
        "l_hip",
        "l_knee",
        "l_ank",
        "r_eye",
        "l_eye",
        "r_ear",
        "l_ear",
    ]
    sigmas = (
        np.array(
            [
                0.26,
                0.79,
                0.79,
                0.72,
                0.62,
                0.79,
                0.72,
                0.62,
                1.07,
                0.87,
                0.89,
                1.07,
                0.87,
                0.89,
                0.25,
                0.25,
                0.35,
                0.35,
            ],
            dtype=np.float32,
        )
        / 10.0
    )
    vars = (sigmas * 2) ** 2
    last_id = -1
    color = [255, 0, 255]


    def __init__(self, keypoints, confidence):
        super().__init__()
        self.keypoints = keypoints
        self.confidence = confidence
        self.bbox = Pose.get_bbox(self.keypoints)
        self.id = None


    @staticmethod
    def get_bbox(keypoints):
        found_keypoints = np.zeros(
            (np.count_nonzero(keypoints[:, 0] != -1), 2), dtype=np.int32
        )
        found_kpt_id = 0
        for kpt_id in range(Pose.num_kpts):
            if keypoints[kpt_id, 0] == -1:
                continue
            found_keypoints[found_kpt_id] = keypoints[kpt_id]
            found_kpt_id += 1
        bbox = cv2.boundingRect(found_keypoints)
        return bbox


    def draw(self, img):
        assert self.keypoints.shape == (Pose.num_kpts, 2)

        for part_id in range(len(BODY_PARTS_PAF_IDS) - 2):
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            global_kpt_a_id = self.keypoints[kpt_a_id, 0]
            if global_kpt_a_id != -1:
                x_a, y_a = self.keypoints[kpt_a_id]
                cv2.circle(img, (int(x_a), int(y_a)), 3, Pose.color, -1)
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            global_kpt_b_id = self.keypoints[kpt_b_id, 0]
            if global_kpt_b_id != -1:
                x_b, y_b = self.keypoints[kpt_b_id]
                cv2.circle(img, (int(x_b), int(y_b)), 3, Pose.color, -1)
            if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), Pose.color, 2)


def get_similarity(
    a,
    b,
    threshold=0.5
    ) -> int:
    """
    Calculate the similarity between two Pose objects based on their keypoints.

    Args:
        a (Pose):
            The first Pose object to compare, containing keypoints and bounding box.
        b (Pose):
            The second Pose object to compare, containing keypoints and bounding box.
        threshold (float, optional):
            The similarity threshold to determine if keypoints are considered similar (default is 0.5).

    Returns:
        int
            The number of keypoints that are similar between the two Pose objects.
    """
    num_similar_kpt = 0
    for kpt_id in range(Pose.num_kpts):
        if a.keypoints[kpt_id, 0] != -1 and b.keypoints[kpt_id, 0] != -1:
            distance = np.sum((a.keypoints[kpt_id] - b.keypoints[kpt_id]) ** 2)
            area = max(a.bbox[2] * a.bbox[3], b.bbox[2] * b.bbox[3])
            similarity = np.exp(
                -distance / (2 * (area + np.spacing(1)) * Pose.vars[kpt_id])
            )
            if similarity > threshold:
                num_similar_kpt += 1
    return num_similar_kpt

def extract_keypoints(
    heatmap : np.ndarray,
    all_keypoints : list,
    total_keypoint_num : int
    ) -> int:
    """
    Extract keypoints from the heatmap and append them to the provided list along with their scores.

    Args:
        heatmap (numpy.ndarray):
            The heatmap from which keypoints will be extracted, with higher values indicating potential keypoints.
        all_keypoints (list):
            A list to store extracted keypoints along with their scores and IDs.
        total_keypoint_num (int):
            The total number of keypoints extracted so far, to assign unique IDs to the new keypoints.

    Returns:
        int
            The number of keypoints extracted from the heatmap.
    """
    heatmap[heatmap < 0.1] = 0
    heatmap_with_borders = np.pad(heatmap, [(2, 2), (2, 2)], mode="constant")
    heatmap_center = heatmap_with_borders[
        1 : heatmap_with_borders.shape[0] - 1, 1 : heatmap_with_borders.shape[1] - 1
    ]
    heatmap_left = heatmap_with_borders[
        1 : heatmap_with_borders.shape[0] - 1, 2 : heatmap_with_borders.shape[1]
    ]
    heatmap_right = heatmap_with_borders[
        1 : heatmap_with_borders.shape[0] - 1, 0 : heatmap_with_borders.shape[1] - 2
    ]
    heatmap_up = heatmap_with_borders[
        2 : heatmap_with_borders.shape[0], 1 : heatmap_with_borders.shape[1] - 1
    ]
    heatmap_down = heatmap_with_borders[
        0 : heatmap_with_borders.shape[0] - 2, 1 : heatmap_with_borders.shape[1] - 1
    ]

    heatmap_peaks = (
        (heatmap_center > heatmap_left)
        & (heatmap_center > heatmap_right)
        & (heatmap_center > heatmap_up)
        & (heatmap_center > heatmap_down)
    )
    heatmap_peaks = heatmap_peaks[
        1 : heatmap_center.shape[0] - 1, 1 : heatmap_center.shape[1] - 1
    ]
    keypoints = list(
        zip(np.nonzero(heatmap_peaks)[1], np.nonzero(heatmap_peaks)[0])
    )  # (w, h)
    keypoints = sorted(keypoints, key=itemgetter(0))

    suppressed = np.zeros(len(keypoints), np.uint8)
    keypoints_with_score_and_id = []
    keypoint_num = 0
    for i in range(len(keypoints)):
        if suppressed[i]:
            continue
        for j in range(i + 1, len(keypoints)):
            if (
                math.sqrt(
                    (keypoints[i][0] - keypoints[j][0]) ** 2
                    + (keypoints[i][1] - keypoints[j][1]) ** 2
                )
                < 6
            ):
                suppressed[j] = 1
        keypoint_with_score_and_id = (
            keypoints[i][0],
            keypoints[i][1],
            heatmap[keypoints[i][1], keypoints[i][0]],
            total_keypoint_num + keypoint_num,
        )
        keypoints_with_score_and_id.append(keypoint_with_score_and_id)
        keypoint_num += 1
    all_keypoints.append(keypoints_with_score_and_id)
    return keypoint_num


def connections_nms(
    a_idx,
    b_idx,
    affinity_scores
    ):
    """
    Apply Non-Maximum Suppression (NMS) to filter out low-scoring connections that share the same keypoints.

    Args:
        a_idx (numpy.ndarray):
            The indices of keypoints for part A of the connections.
        b_idx (numpy.ndarray):
            The indices of keypoints for part B of the connections.
        affinity_scores (numpy.ndarray):
            The affinity scores for each connection, indicating how well the keypoints are connected.

    Returns:
        Tuple[numpy.ndarray, np.ndarray, np.ndarray]:
            - a_idx : The filtered indices of keypoints for part A.
            - b_idx : The filtered indices of keypoints for part B.
            - affinity_scores : The filtered affinity scores corresponding to the connections.
    """
    # From all retrieved connections that share the same starting/ending keypoints leave only the top-scoring ones.
    order = affinity_scores.argsort()[::-1]
    affinity_scores = affinity_scores[order]
    a_idx = a_idx[order]
    b_idx = b_idx[order]
    idx = []
    has_kpt_a = set()
    has_kpt_b = set()
    for t, (i, j) in enumerate(zip(a_idx, b_idx)):
        if i not in has_kpt_a and j not in has_kpt_b:
            idx.append(t)
            has_kpt_a.add(i)
            has_kpt_b.add(j)
    idx = np.asarray(idx, dtype=np.int32)
    return a_idx[idx], b_idx[idx], affinity_scores[idx]


def group_keypoints(
    all_keypoints_by_type : list,
    pafs : np.ndarray,
    pose_entry_size : int = 20,
    min_paf_score : float = 0.05
    ):
    """
    Group keypoints into pose entries based on Part Affinity Fields (PAFs).

    Parameters:
        all_keypoints_by_type (list):
            A list of arrays containing keypoints categorized by body part types.
        pafs (numpy.ndarray):
            The Part Affinity Fields, indicating connections between keypoints.
        pose_entry_size (int, optional)
            The size of each pose entry array (default is 20).
        min_paf_score (float, optional):
            The minimum score of PAFs required to consider a connection valid (default is 0.05).

    Returns:
        Tuple[numpy.ndarray, numpy.ndarray]:
            - pose_entries : An array of pose entries, where each entry contains information about connected keypoints.
            - all_keypoints : A flattened array containing all extracted keypoints.
    """
    pose_entries = []
    all_keypoints = np.array([item for sublist in all_keypoints_by_type for item in sublist])
    points_per_limb = 10
    grid = np.arange(points_per_limb, dtype=np.float32).reshape(1, -1, 1)
    all_keypoints_by_type = [np.array(keypoints, np.float32) for keypoints in all_keypoints_by_type]
    for part_id in range(len(BODY_PARTS_PAF_IDS)):
        part_pafs = pafs[:, :, BODY_PARTS_PAF_IDS[part_id]]
        kpts_a = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][0]]
        kpts_b = all_keypoints_by_type[BODY_PARTS_KPT_IDS[part_id][1]]
        n = len(kpts_a)
        m = len(kpts_b)
        if n == 0 or m == 0:
            continue

        # Get vectors between all pairs of keypoints, i.e. candidate limb vectors.
        a = kpts_a[:, :2]
        a = np.broadcast_to(a[None], (m, n, 2))
        b = kpts_b[:, :2]
        vec_raw = (b[:, None, :] - a).reshape(-1, 1, 2)

        # Sample points along every candidate limb vector.
        steps = 1 / (points_per_limb - 1) * vec_raw
        points = steps * grid + a.reshape(-1, 1, 2)
        points = points.round().astype(dtype=np.int32)
        x = points[..., 0].ravel()
        y = points[..., 1].ravel()

        # Compute affinity score between candidate limb vectors and part affinity field.
        field = part_pafs[y, x].reshape(-1, points_per_limb, 2)
        vec_norm = np.linalg.norm(vec_raw, ord=2, axis=-1, keepdims=True)
        vec = vec_raw / (vec_norm + 1e-6)
        affinity_scores = (field * vec).sum(-1).reshape(-1, points_per_limb)
        valid_affinity_scores = affinity_scores > min_paf_score
        valid_num = valid_affinity_scores.sum(1)
        affinity_scores = (affinity_scores * valid_affinity_scores).sum(1) / (
            valid_num + 1e-6
        )
        success_ratio = valid_num / points_per_limb

        # Get a list of limbs according to the obtained affinity score.
        valid_limbs = np.where(
            np.logical_and(affinity_scores > 0, success_ratio > 0.8)
        )[0]
        if len(valid_limbs) == 0:
            continue
        b_idx, a_idx = np.divmod(valid_limbs, n)
        affinity_scores = affinity_scores[valid_limbs]

        # Suppress incompatible connections.
        a_idx, b_idx, affinity_scores = connections_nms(a_idx, b_idx, affinity_scores)
        connections = list(
            zip(
                kpts_a[a_idx, 3].astype(np.int32),
                kpts_b[b_idx, 3].astype(np.int32),
                affinity_scores,
            )
        )
        if len(connections) == 0:
            continue

        if part_id == 0:
            pose_entries = [
                np.ones(pose_entry_size) * -1 for _ in range(len(connections))
            ]
            for i in range(len(connections)):
                pose_entries[i][BODY_PARTS_KPT_IDS[0][0]] = connections[i][0]
                pose_entries[i][BODY_PARTS_KPT_IDS[0][1]] = connections[i][1]
                pose_entries[i][-1] = 2
                pose_entries[i][-2] = (
                    np.sum(all_keypoints[connections[i][0:2], 2]) + connections[i][2]
                )
        elif part_id == 17 or part_id == 18:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                for j in range(len(pose_entries)):
                    if (
                        pose_entries[j][kpt_a_id] == connections[i][0]
                        and pose_entries[j][kpt_b_id] == -1
                    ):
                        pose_entries[j][kpt_b_id] = connections[i][1]
                    elif (
                        pose_entries[j][kpt_b_id] == connections[i][1]
                        and pose_entries[j][kpt_a_id] == -1
                    ):
                        pose_entries[j][kpt_a_id] = connections[i][0]
            continue
        else:
            kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]
            kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
            for i in range(len(connections)):
                num = 0
                for j in range(len(pose_entries)):
                    if pose_entries[j][kpt_a_id] == connections[i][0]:
                        pose_entries[j][kpt_b_id] = connections[i][1]
                        num += 1
                        pose_entries[j][-1] += 1
                        pose_entries[j][-2] += (
                            all_keypoints[connections[i][1], 2] + connections[i][2]
                        )
                if num == 0:
                    pose_entry = np.ones(pose_entry_size) * -1
                    pose_entry[kpt_a_id] = connections[i][0]
                    pose_entry[kpt_b_id] = connections[i][1]
                    pose_entry[-1] = 2
                    pose_entry[-2] = (
                        np.sum(all_keypoints[connections[i][0:2], 2])
                        + connections[i][2]
                    )
                    pose_entries.append(pose_entry)

    filtered_entries = []
    for i in range(len(pose_entries)):
        if pose_entries[i][-1] < 3 or (pose_entries[i][-2] / pose_entries[i][-1] < 0.2):
            continue
        filtered_entries.append(pose_entries[i])
    pose_entries = np.asarray(filtered_entries)
    return pose_entries, all_keypoints


def postprocess_output(
    stage2_heatmaps : np.ndarray,
    stage2_pafs : np.ndarray,
    scale : float,
    pad : tuple,
    stride : int = 8,
    upsample_ratio : int = 4
    ):
    """
    Post-process the heatmaps and Part Affinity Fields (PAFs) from the model output
    to extract and normalize keypoints for detected poses.

    Args:
        stage2_heatmaps (numpy.ndarray):
            The heatmaps generated from the second stage of the model, indicating keypoint locations.
        stage2_pafs (numpy.ndarray):
            The Part Affinity Fields generated from the second stage of the model, indicating parts connectivity.
        scale (float):
            The scaling factor used to normalize the output keypoints.
        pad (tuple):
            The padding applied to the image before it was processed, in the format (pad_height, pad_width).
        stride (int, optional)
            The stride used in the model to downscale the input image (default is 8).
        upsample_ratio (int, optional)
            The ratio for upsampling the heatmaps and PAFs to the original image size (default is 4).

    Returns:
        list
            A list of Pose objects, each containing keypoints of detected poses.
    """
    heatmaps = np.transpose(np.squeeze(stage2_heatmaps), (1, 2, 0))
    heatmaps = cv2.resize(heatmaps,(0, 0),fx=upsample_ratio,fy=upsample_ratio,interpolation=cv2.INTER_CUBIC,)

    pafs = np.transpose(np.squeeze(stage2_pafs), (1, 2, 0))
    pafs = cv2.resize(pafs,(0, 0),fx=upsample_ratio,fy=upsample_ratio,interpolation=cv2.INTER_CUBIC,)
    total_keypoints_num = 0
    all_keypoints_by_type = []
    num_keypoints = Pose.num_kpts
    for kpt_idx in range(num_keypoints):
        total_keypoints_num += extract_keypoints(
            heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num
        )

    pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
    for kpt_id in range(all_keypoints.shape[0]):
        all_keypoints[kpt_id, 0] = (
            all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]
        ) / scale
        all_keypoints[kpt_id, 1] = (
            all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]
        ) / scale
    current_poses = []
    for n in range(len(pose_entries)):
        if len(pose_entries[n]) == 0:
            continue
        pose_keypoints = np.ones((num_keypoints, 2), dtype=np.int32) * -1
        for kpt_id in range(num_keypoints):
            if pose_entries[n][kpt_id] != -1.0:
                pose_keypoints[kpt_id, 0] = int(
                    all_keypoints[int(pose_entries[n][kpt_id]), 0]
                )
                pose_keypoints[kpt_id, 1] = int(
                    all_keypoints[int(pose_entries[n][kpt_id]), 1]
                )
        pose = Pose(pose_keypoints, pose_entries[n][18])
        current_poses.append(pose)
    return current_poses
