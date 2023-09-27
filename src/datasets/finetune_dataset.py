import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from einops import repeat, reduce
from log import logger
from PIL import Image
from torchvision import transforms
from einops import rearrange, repeat
from ldm.util import instantiate_from_config
import pandas as pd
from src.datasets import get_dataset
from src.foreground_masks import GMMMaskSuggestor
from utils import make_exp_config
from utils import DatasetSplit
import importlib
import torch
from torch import optim
import numpy as np
from inspect import isfunction
from PIL import Image, ImageDraw, ImageFont



def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

def foba_human_dataset(
    opt_path="PATH",
    image_transforms=[],
    entire_image_synthesis_probability=1,
):

    opt = make_exp_config(opt_path)
    ds = get_dataset(opt)
    ds.add_preliminary_masks()
    mask_suggestor = GMMMaskSuggestor(opt)

    print(f"Finetuning on inpainting with a probability of p={entire_image_synthesis_probability}")
    if not isinstance(image_transforms, transforms.Compose):
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        tform = transforms.Compose(image_transforms)

    no_crop = transforms.Compose(
        [
        transforms.Resize(512),
        ]
    )


    def pre_process(examples):
        processed = {}
        p = torch.rand(1)
        if p <= entire_image_synthesis_probability:
            # reconstruct image
            processed["txt"] = opt.foreground_prompt
            full_mask = torch.ones_like(examples["preliminary_mask"]).to(bool).squeeze()
        elif p > entire_image_synthesis_probability:
            processed["txt"] = opt.background_prompt
            full_mask = mask_suggestor(examples).squeeze()

        # move mask to image space
        mask = repeat(full_mask, "h w -> 1 1 (h h4) (w w4)", h4=opt.f, w4=opt.f)

        tf_in = examples["x"]
        tf_in = torch.cat([tf_in, mask], dim=1)
        if torch.rand(1) < 0.1:
            tf_out = no_crop(tf_in)
        else:
            tf_out = tform(tf_in)
        mask = tf_out[0, 3]
        tf_out =  tf_out[0:1, :3]
        mask = reduce(mask, "(h h4) (w w4) -> h w", "mean", h4=opt.f, w4=opt.f).round().to(torch.bool)
        #mask = repeat(mask, "h w -> c h w", c=4)
        processed["mask"] = repeat(mask_suggestor.get_rectangular_inpainting_mask(mask), "h w -> c h w", c=4)
        processed["image"] = rearrange(tf_out, "1 c h w -> h w c")
        return processed

    ds.set_transform(pre_process)
    return ds

def foba_chest_dataset(
    opt_path="PATH",
    image_transforms=[],
    entire_image_synthesis_probability=1,
):

    opt = make_exp_config(opt_path)
    ds = get_dataset(opt)

    print(f"Finetuning on inpainting with a probability of p={entire_image_synthesis_probability}")
    if not isinstance(image_transforms, transforms.Compose):
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        tform = transforms.Compose(image_transforms)

    def pre_process(examples):
        processed = {}
        p = torch.rand(1)

        if p <= entire_image_synthesis_probability:
            # reconstruct image
            full_mask = torch.ones((64, 64)).to(bool).squeeze()
            processed["txt"] = examples["prompt"]
            processed["mask"] = repeat(full_mask, "h w -> c h w", c=4)
            tf_in = examples["x"]
            tf_in = tf_in[examples["slice"]]
            tf_out = torch.clamp(tform(tf_in), -1, 1)
            processed["image"] = rearrange(tf_out, "1 c h w -> h w c")
        elif p > entire_image_synthesis_probability:
            raise NotImplementedError()
        return processed
    ds.set_transform(pre_process)
    return ds

class ChestFobaValDataset(Dataset):
    def __init__(self, opt_path, image_transforms, captions, caption_masks, output_size, image_key="image", caption_key="txt", n_gpus=1, latent_mask=False, limit_dataset=None):
        """Returns only captions with dummy images"""

        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        self.caption_masks = caption_masks
        self.captions = captions
        self.latent_mask = latent_mask
        if not isinstance(image_transforms, transforms.Compose):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
            image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms

        self.opt = make_exp_config(opt_path)
        #opt.dataset_args["split"] = DatasetSplit("val")
        self.opt.dataset_args["limit_dataset"] = limit_dataset
        ds = get_dataset(self.opt)

    def __getitem__(self, item):
        #item = item // 2 # fix for double gpu
        n_captions = len(self.caption_masks)
        validation_sample = self.ds[(item // n_captions) % len(self.ds)]
        img = validation_sample["x"]

        caption_mask = self.caption_masks[i]
        caption = self.captions[i]

        mask = torch.ones((1, 64, 64))
        return {"image":torch.zeros((3, 512, 512)),
                "mask": mask,
                "item":item,
                self.caption_key: caption,
                "caption_mask": caption_mask,
                "rel_path": validation_sample["rel_path"],
        }

    def __len__(self):
        return len(self.captions)


def foba_dataset(
    opt_path="PATH",
    image_transforms=[],
    entire_image_synthesis_probability=1,
):

    opt = make_exp_config(opt_path)
    ds = get_dataset(opt)
    ds.add_preliminary_masks()
    mask_suggestor = GMMMaskSuggestor(opt)

    print(f"Finetuning on inpainting with a probability of p={entire_image_synthesis_probability}")
    if not isinstance(image_transforms, transforms.Compose):
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        tform = transforms.Compose(image_transforms)

    def pre_process(examples):
        processed = {}
        p = torch.rand(1)

        if p <= entire_image_synthesis_probability:
            # reconstruct image
            processed["txt"] = opt.foreground_prompt
            full_mask = torch.ones_like(examples["preliminary_mask"]).to(bool).squeeze()
            processed["mask"] = repeat(full_mask, "h w -> c h w", c=4)
            tf_in = examples["x"]
            tf_in = tf_in[examples["slice"]]
            tf_out = torch.clamp(tform(tf_in), -1, 1)
            processed["image"] = rearrange(tf_out, "1 c h w -> h w c")
        elif p > entire_image_synthesis_probability:
            processed["txt"] = opt.background_prompt
            full_mask = mask_suggestor(examples).squeeze()

            mask = repeat(full_mask, "h w -> 1 1 (h h4) (w w4)", h4=opt.f, w4=opt.f)

            tf_in = examples["x"]
            tf_in = torch.cat([tf_in, mask], dim=1)
            tf_in = tf_in[examples["slice"]]
            tf_out = tform(tf_in)
            mask = tf_out[0, 3]
            tf_out = tf_out[0:1, :3]
            mask = reduce(mask, "(h h4) (w w4) -> h w", "mean", h4=opt.f, w4=opt.f).round().to(torch.bool)

            processed["mask"] = repeat(mask_suggestor.get_rectangular_inpainting_mask(mask), "h w -> c h w", c=4)
            processed["image"] = rearrange(tf_out, "1 c h w -> h w c")
        return processed

    ds.set_transform(pre_process)
    return ds


class FobaValDataset(Dataset):
    def __init__(self, opt_path, image_transforms, captions, caption_masks, output_size, image_key="image", caption_key="txt", n_gpus=1, latent_mask=False, limit_dataset=None):
        """Returns only captions with dummy images"""

        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        self.caption_masks = caption_masks
        self.captions = captions
        self.latent_mask = latent_mask
        if not isinstance(image_transforms, transforms.Compose):
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
            image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms

        self.opt = make_exp_config(opt_path)
        #opt.dataset_args["split"] = DatasetSplit("val")
        self.opt.dataset_args["limit_dataset"] = limit_dataset
        ds = get_dataset(self.opt)
        ds.add_preliminary_masks()
        self.ds = ds
        self.mask_suggestor = GMMMaskSuggestor(self.opt)

        #assert self.test_inpaint_img.size()[-1] == self.inpaint_rect_mask.size()[-1] == self.inpaint_obj_mask.size()[-1] == self.output_size, f"wrong size of mask or image!img: {self.test_inpaint_img.size()} {self.inpaint_obj_mask.size()}, output size: {self.output_size}"

    def __getitem__(self, item):
        #item = item // 2 # fix for double gpu
        n_captions = len(self.caption_masks)
        validation_sample = self.ds[(item // n_captions) % len(self.ds)]
        img = validation_sample["x"]
        fullmask = repeat(self.mask_suggestor(validation_sample), "h w -> 1 1 (h h4) (w w4)", h4=self.opt.f, w4=self.opt.f)
        tf_in = torch.cat([img, fullmask], dim=1)
        tf_in = tf_in[validation_sample["slice"]]
        tf_out = self.tform(tf_in)
        mask = tf_out[0, 3]
        tf_out = tf_out[0:1, :3]
        mask = reduce(mask, "(h h4) (w w4) -> h w", "mean", h4=self.opt.f, w4=self.opt.f).round().to(torch.bool)

        test_inpaint_img = rearrange(tf_out, "1 c h w -> h w c") # test_inpaint_img # inpaint image to test inpainting, and removal
        inpaint_obj_mask = rearrange(mask, "h w -> 1 1 h w")
        inpaint_rect_mask = rearrange(self.mask_suggestor.get_rectangular_inpainting_mask(mask), "h w -> 1 1 h w")

        i = item % n_captions
        caption_mask = self.caption_masks[i]
        caption = self.captions[i]

        if caption_mask == "full":
            mask = torch.ones_like(inpaint_obj_mask)
        elif caption_mask == "fg":
            mask = torch.clone(inpaint_obj_mask)
        elif caption_mask == "bg":
            mask = torch.clone(inpaint_rect_mask)
        else:
            raise ValueError(f"Unkown caption mask: {caption_mask}")
        mask = repeat(mask, "1 1 h w -> c h w", c=4)

        return {"image":test_inpaint_img,
                "mask": mask,
                "item":item,
                self.caption_key: caption,
                "caption_mask": caption_mask,
                "rel_path": validation_sample["rel_path"],
        }

    def __len__(self):
        return len(self.captions)


class FobaSynthesisDataset(Dataset):
    def __init__(self, opt_path, caption, caption_mask, output_size, n_gpus=1, limit_dataset=None):
        """Returns only captions with dummy images"""

        self.output_size = output_size
        self.caption_mask = caption_mask
        self.caption = caption

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

        self.opt = make_exp_config(opt_path)
        self.opt.dataset_args["limit_dataset"] = limit_dataset
        ds = get_dataset(self.opt)
        self.ds = ds
        self.mask_suggestor = GMMMaskSuggestor(self.opt)

    def __getitem__(self, item):
        validation_sample = self.ds[item]
        img = validation_sample["img"]
        ret_dict =  {"image":img,
                "item":item,
                "slice":validation_sample["slice"],
                "caption_mask": self.caption_mask,
                "rel_path": validation_sample["rel_path"],

        }

        if validation_sample.get("preliminary_mask") is not None:
            ret_dict["preliminary_mask"] = validation_sample["preliminary_mask"]
        return ret_dict

    def __len__(self):
        return len(self.ds)

    def tform(self, tf_in):
        tf_out = tf_in
        return tf_out