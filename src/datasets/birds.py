import os
import torch
import cv2
from src.datasets.dataset import FOBADataset
import numpy as np
from utils import DatasetSplit, SPLIT_TO_DATASETSPLIT
from src.datasets.utils import path_to_tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from log import logger
from einops import rearrange, repeat


def file_to_list(path):
    lines = []
    with open(path) as fp:
        lines = fp.readlines()
    return lines


class BirdDataset(FOBADataset):
    def __init__(self, opt, H, W,mask_dir=None):
        super().__init__(opt, H, W, mask_dir)
        self.load_segmentations = opt.dataset_args.get("load_segmentations", False)
        self.load_bboxes = opt.dataset_args.get("load_bboxes", False)
        self._build_dataset()

        if self.load_segmentations:
            for i in range(len(self.data)):
                self.data[i]["seg_path"] = os.path.join(self.base_dir,
                                                        "segmentations",
                                                        self.data[i]["rel_path"][7:]
                                                        )[:-3] + "png"

    def _load_images(self, index):
        assert len(index) == 1
        entry = self.data[index[0]].copy()
        img = path_to_tensor(entry["img_path"])
        entry["img"] = img

        x = torch.full((1, 3, self.H, self.W), -1.)
        x[0, :, :img.size()[2], :img.size()[3]] = img

        entry["x"] = x
        entry["slice"] = (slice(None), slice(None), slice(0, img.size()[2]), slice(0, img.size()[3]))
        if self.load_segmentations:
            seg = (path_to_tensor(entry["seg_path"]) + 1) / 2
            seg = seg.round() #middle

            entry["segmentation"] = seg

            y = torch.full((1, 3, self.H, self.W), 0.)
            y[0, :, :seg.size()[2], :seg.size()[3]] = seg
            entry["segmentation_x"] = y

        if self._preliminary_masks_path is not None:
            entry["preliminary_mask"] = torch.load(os.path.join(self._preliminary_masks_path, entry["rel_path"] + ".pt"))
            if not self.latent_attention_mask:
                entry["preliminary_mask"] = repeat(entry["preliminary_mask"], "1 1 c h w -> 1 1 c (h h2) (w w2)",
                                                   h2=self.opt.f, w2=self.opt.f)

        if self._inpainted_images_path is not None:
            entry["inpainted_image"] = torch.load(os.path.join(self._inpainted_images_path, entry["rel_path"] + ".pt"))

        if self.load_bboxes:
            for i, entry in enumerate(file_to_list(os.path.join(self.base_dir, "bounding_boxes.txt"))):
                entry = entry.strip().split(" ")
                bounding_box = [int(float(x)) for x in entry[1:]]
                bbox_img = torch.zeros_like(img)
                bbox_img[:, :, int(bounding_box[1]):int(bounding_box[3] + 1), int(bounding_box[0]):int(bounding_box[2] + 1)] = 1  # overlaps with img
                raise NotImplementedError("Unetested Feature!")


        tmp_mask_path = os.path.join(self.opt.base_dir, "refined_mask_tmp", entry["rel_path"] + ".pt")
        if os.path.isfile(tmp_mask_path):
            entry["refined_mask"] = torch.load(tmp_mask_path)
        return entry


