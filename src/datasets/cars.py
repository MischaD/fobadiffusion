import os
import torch
from src.datasets.utils import resize, path_to_tensor
from src.datasets.dataset import FOBADataset
import scipy.io
import numpy as np


class CarDataset(FOBADataset):
    def __init__(self, opt, H, W, mask_dir=None):
        super().__init__(opt, H, W, mask_dir)

        self.load_bboxes = opt.dataset_args.get("load_bboxes", False)
        if self.load_bboxes:
            self.annotations = scipy.io.loadmat(os.path.join(opt.base_dir, "devkit/cars_test_annos.mat"))["annotations"][0]
            # 0000n.jpg  at position n - 1

        self._build_dataset()

    def _load_images(self, index):
        assert len(index) == 1
        entry = self.data[index[0]].copy()
        img = path_to_tensor(entry["img_path"])
        # images too large are resized to self.W^2
        if max(img.size()) > self.W:
            img = resize(img, tosize=self.W)
        entry["img"] = img

        x = torch.full((1, 3, self.H, self.W), -1.)
        x[0, :, :img.size()[2], :img.size()[3]] = img

        entry["x"] = x
        entry["slice"] = (slice(None), slice(None), slice(0, img.size()[2]), slice(0, img.size()[3]))

        if self._preliminary_masks_path is not None:
            entry["preliminary_mask"] = torch.load(os.path.join(self._preliminary_masks_path, entry["rel_path"] + ".pt"))

        if self._inpainted_images_path is not None:
            entry["inpainted_image"] = torch.load(os.path.join(self._inpainted_images_path, entry["rel_path"] + ".pt"))

        tmp_mask_path = os.path.join(self.opt.base_dir, "refined_mask_tmp", entry["rel_path"] + ".pt")
        if os.path.isfile(tmp_mask_path):
            entry["refined_mask"] = torch.load(tmp_mask_path)

        if self.load_bboxes:
            nimg = int(entry["rel_path"].split("/")[-1].split(".")[0]) - 1
            bbox = self.annotations[nimg]
            bbox = np.array([bbox[0][0, 0], bbox[1][0, 0], bbox[2][0, 0], bbox[3][0, 0]])

            entry["bbox"] = bbox
            bbox_img = torch.zeros_like(img)
            bbox_img[:, :, bbox[1]:(bbox[3]+1), bbox[0]:(bbox[2]+1)] = 1# overlaps with img
            entry["bbx_img"] = bbox_img

        return entry


