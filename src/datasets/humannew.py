from torch.utils.data import Dataset
import pandas as pd
from abc import ABC
import os
import random
import json
import torch
import cv2
import numpy as np
from utils import DatasetSplit
from src.datasets.utils import file_to_list
from random import shuffle
from einops import reduce, rearrange, repeat
from utils import DatasetSplit, SPLIT_TO_DATASETSPLIT
from src.datasets.utils import resize, path_to_tensor
from src.datasets.dataset import FOBADataset
from log import logger


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


class HumanDataset(FOBADataset):
    def __init__(self, opt, H, W, mask_dir=None):
        super().__init__(opt, H, W, mask_dir)

        self.load_bboxes = opt.dataset_args.get("load_bboxes", False)
        #self.data_dict =
        self.data_dict = json.load(open(os.path.join(self.base_dir, "data_dict.json")))
        self.test_data_dict = json.load(open(os.path.join(self.base_dir, "test_data_dict.json")))

        self._build_dataset()

    def _load_images(self, index):
        assert len(index) == 1
        entry = self.data[index[0]].copy()
        img = path_to_tensor(entry["img_path"])
        entry["img"] = img

        assert img.size()[2] == 256 and img.size()[3] == 256
        x = repeat(img, "1 c h w -> 1 c (h h2) (w w2)", h2=2, w2=2)

        entry["x"] = x
        entry["slice"] = (slice(None), slice(None), slice(None), slice(None))

        if self.load_bboxes:
            file = entry["rel_path"].split("/")
            if file[1] == "training":
                data_dict = self.data_dict
            else:
                data_dict = self.test_data_dict
            frame = data_dict[file[2]][file[3]][int(file[4].lstrip("frame").rstrip(".png")) - 1]
            bbx = np.array(frame["bounding_box"]["__ndarray__"])
            bbox_img = torch.zeros((1000, 1000))
            bbox_img[bbx[0, 1]: bbx[1, 1], bbx[0, 0]: bbx[1, 0]] = 1
            center = np.array((bbx[0, 0] + (bbx[1, 0] - bbx[0, 0]) // 2, bbx[0, 1] + (bbx[1, 1] - bbx[0, 1]) // 2))
            scale = ((bbx[1][1] - bbx[0][1]) / 170)
            affine_transform = get_affine_transform(center,
                                 scale,
                                 0,
                                (128, 128),
                                 shift=np.array([0, 0], dtype=np.float32),
                                 inv=0)
            bbx_img = cv2.warpAffine(
                bbox_img.numpy(),
                affine_transform,
                (128, 128),
                flags=cv2.INTER_NEAREST)
            entry["bbx_img"] = torch.tensor(bbx_img)

        if self.load_segmentations:
            seg = (path_to_tensor(entry["seg_path"]) + 1) / 2
            entry["segmentation"] = seg
            y = torch.full((1, 3, self.H, self.W), 0.)
            y[0, :, :seg.size()[2], :seg.size()[3]] = seg
            entry["segmentation_x"] = y

        if self._preliminary_masks_path is not None:
            entry["preliminary_mask"] = torch.load(os.path.join(self._preliminary_masks_path, entry["rel_path"] + ".pt"))
            if not self.latent_attention_mask:
                entry["preliminary_mask"] = repeat(entry["preliminary_mask"], "1 1 c h w -> 1 1 c (h h2) (w w2)", h2=self.opt.f, w2=self.opt.f)

        if self._inpainted_images_path is not None:
            entry["inpainted_image"] = torch.load(os.path.join(self._inpainted_images_path, entry["rel_path"] + ".pt"))
        return entry