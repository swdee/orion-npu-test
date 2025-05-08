# ---------------------------------------------------------------------
# Copyright 2024 Cix Technology Group Co., Ltd.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------------------------
import os
from tqdm import tqdm
import numpy as np
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../..")
from utils.image_process import imagenet_preprocess_method1
from typing import List, Dict, Tuple


class ImageNet_Metric:
    def __init__(
        self,
        model: any,
        model_type: str,
        valid_path: str = None,
        sel_imgs: int = 50000,
        rand_seed: int = 42,
    ):
        if valid_path is not None:
            self.valid_path = valid_path
        else:
            self.valid_path = os.path.join(
                os.path.dirname(__file__),
                "../../datasets/ILSVRC2012/val/",
            )
        self.rand_seed = rand_seed
        self.sel_imgs = sel_imgs
        self.model_type = model_type
        self.model = model

    @staticmethod
    def load_image_labels(
        image_dir: str,
    ) -> Tuple[List[str], List[int], Dict[str, int]]:
        image_paths = []
        labels = []
        class_names = sorted(os.listdir(image_dir))
        class_to_index = {
            class_name: index for index, class_name in enumerate(class_names)
        }

        for class_name in class_names:
            class_dir = os.path.join(image_dir, class_name)
            for file_name in os.listdir(class_dir):
                if file_name.endswith(".JPEG"):
                    image_paths.append(os.path.join(class_dir, file_name))
                    labels.append(class_to_index[class_name])

        return image_paths, labels, class_to_index

    def load_data(self):
        return self.load_image_labels(self.valid_path)

    def run(self, input_size: int = 224, data_type: str = "np"):
        np.random.seed(self.rand_seed)
        image_paths, labels, class2indexs = self.load_data()
        total_img_num = len(image_paths)
        if self.sel_imgs >= total_img_num:
            sel_imgs = total_img_num
        else:
            sel_imgs = self.sel_imgs
        rand_ids = np.random.choice(total_img_num, sel_imgs, replace=False)

        print(f"total images : {total_img_num}")
        print(f"select images : {self.sel_imgs}")

        if self.model_type == "onnx":
            input_name = self.model.get_inputs()[0].name
            output_name = self.model.get_outputs()[0].name

        # elif self.model_type == "cix":

        top1_count = 0
        top5_count = 0
        total_count = 0

        for i in tqdm(rand_ids):
            image_path = image_paths[i]
            true_label = labels[i]
            image = imagenet_preprocess_method1(
                image_path, input_size=input_size, return_type=data_type
            )

            if self.model_type == "onnx":
                # Run the model inference and get the output
                outputs = self.model.run([output_name], {input_name: image})[0]
            elif self.model_type == "cix":
                outputs = self.model.forward(image)[0]
                outputs = np.expand_dims(outputs, axis=0)
            else:
                NotImplementedError()

            top1_pred = np.argmax(outputs)
            top5_pred = np.argsort(outputs)[:, -5:][::-1]
            if outputs.shape[-1] == 1000:
                if top1_pred == true_label:
                    top1_count += 1
                if true_label in top5_pred:
                    top5_count += 1
            elif outputs.shape[-1] == 1001:
                if (top1_pred - 1) == true_label:
                    top1_count += 1
                if true_label in (top5_pred - 1):
                    top5_count += 1
            total_count += 1

        top1_accuracy = top1_count / total_count
        top5_accuracy = top5_count / total_count

        print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
        print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
        if self.model_type == "cix":
            ave_fps = self.model.get_ave_fps()
            max_fps = self.model.get_max_fps()
            print(f"Ave forward : {ave_fps:.5f} fps")
            print(f"Max forward : {max_fps:.5f} fps")
            self.model.clean()


class StreamSegMetrics:
    """
    Stream Metrics for Semantic Segmentation Task
    """
    def __init__(self, n_classes,data_dir):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.data_dir = data_dir

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist( lt.flatten(), lp.flatten() )

    def get_data_list(self):
        with open(os.path.join(self.data_dir, "ImageSets/Segmentation/val.txt"), "r") as f:
            val_datasets = f.readlines()
        images_list_gt = list()
        images_list = list()
        for tmp_file_name in val_datasets:
            images_name = os.path.join(
                self.data_dir, "JPEGImages", tmp_file_name.strip("\n") + ".jpg"
            )
            images_gt_name = os.path.join(
                self.data_dir,
                "SegmentationClass",
                tmp_file_name.strip("\n") + ".png",
            )
            if os.path.exists(images_name) and os.path.exists(images_gt_name):
                images_list_gt.append(images_gt_name)
                images_list.append(images_name)
        return images_list, images_list_gt

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            if k!="Class IoU":
                string += "%s: %f\n"%(k, v)
        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return {
                "Overall Acc": acc,
                "Mean Acc": acc_cls,
                "FreqW Acc": fwavacc,
                "Mean IoU": mean_iu,
                "Class IoU": cls_iu,
            }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))