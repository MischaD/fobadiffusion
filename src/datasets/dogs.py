import os
import random
import torch
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from utils import DatasetSplit
from random import shuffle
from src.datasets.utils import file_to_list, resize, path_to_tensor
from src.datasets.dataset import FOBADataset
from einops import rearrange, repeat
import scipy.ndimage as ndimage


class DogDataset(FOBADataset):
    def __init__(self, opt, H, W, mask_dir=None):
        super().__init__(opt, H, W, mask_dir)
        self.load_bboxes = opt.dataset_args.get("load_bboxes", False)
        self._build_dataset()

    def _load_images(self, index):
        assert len(index)
        entry = self.data[index[0]].copy()
        img = path_to_tensor(entry["img_path"])

        # images too large are resized to self.W^2
        if max(img.size()) > self.W:
            #assert False, "reimplement this"
            img = resize(img, tosize=self.W)
        entry["img"] = img

        x = torch.full((1, 3, self.H, self.W), -1.)
        x[0, :, :img.size()[2], :img.size()[3]] = img

        entry["x"] = x
        entry["slice"] = (slice(None), slice(None), slice(0, img.size()[2]), slice(0, img.size()[3]))

        if self._preliminary_masks_path is not None:
            entry["preliminary_mask"] = torch.load(os.path.join(self._preliminary_masks_path, entry["rel_path"] + ".pt"))
            if not self.latent_attention_mask:
                entry["preliminary_mask"] = repeat(entry["preliminary_mask"], "1 1 c h w -> 1 1 c (h h2) (w w2)", h2=self.opt.f, w2=self.opt.f)

        if self._inpainted_images_path is not None:
            entry["inpainted_image"] = torch.load(os.path.join(self._inpainted_images_path, entry["rel_path"] + ".pt"))

        if self.load_bboxes:
            path_preproc = "n" + entry["rel_path"].rstrip(".jpg").lstrip("stanford_dogs/")
            bbox_path = os.path.join(self.base_dir, "Annotation", path_preproc)
            tree = ET.parse(bbox_path)
            root = tree.getroot()
            bbox = [int(root[5][4][i].text) for i in range(4)]
            entry["bbox"] = bbox
            bbox_img = torch.zeros_like(img)
            bbox_img[:, :, bbox[1]:(bbox[3]+1), bbox[0]:(bbox[2]+1)] = 1# overlaps with img
            entry["bbx_img"] = bbox_img

        tmp_mask_path = os.path.join(self.opt.base_dir, "refined_mask_tmp", entry["rel_path"] + ".pt")
        if os.path.isfile(tmp_mask_path):
            entry["refined_mask"] = torch.load(tmp_mask_path)
            if max(entry["refined_mask"].size()) > self.W:
                assert False, "reimplement this"

            if entry["refined_mask"].sum() < 100:
                logger.warn("Redo sampling for missing labels")
                return self._load_images(np.random.randint(len(self)))
            return entry

        return entry

