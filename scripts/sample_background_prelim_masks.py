import argparse, os, sys, datetime, glob, importlib, csv
import numpy as np
import time
import time
import torch
from src.models.segmentation_unet import UNet
from collections import OrderedDict
import pytorch_lightning as pl
from packaging import version
from omegaconf import OmegaConf
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from functools import partial
from PIL import Image
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.distributed import rank_zero_only
from pytorch_lightning.utilities import rank_zero_info
import argparse
from torchvision.transforms import ToPILImage
import shutil
import time
import os
import pickle
import numpy as np
import datetime
from tqdm import tqdm
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
import torch
from torch import autocast
from contextlib import contextmanager, nullcontext
from src.datasets import get_dataset
from src.visualization.utils import model_to_viz
from log import logger
from utils import get_compute_background_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz, resize_long_edge
from einops import reduce, rearrange, repeat
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from src.preliminary_masks import reorder_attention_maps, normalize_attention_map_size
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from src.foreground_masks import GMMMaskSuggestor
from src.datasets.finetune_dataset import FobaSynthesisDataset
import torch
from tqdm import tqdm
from log import logger
import torchvision


def list_to_batch(list_):
    for i in range(len(list_)):
        assert len(list_[i]) == 1, "multiple captions found in single sample - just run the script twice - not worth it"
        list_[i] = list_[i][0]
    return torch.stack(list_,dim=0)


def log_images(
        save_path,
        should_i_log,
        input,
        masks,
        output,
        reconstruction,
        iteration=-1,

):
    resize_fn = torchvision.transforms.Resize(512)
    input = torch.clamp((input + 1.0) / 2.0, min=0.0, max=1.0)
    masks = masks.cpu()
    output = output.cpu()
    reconstruction = reconstruction.cpu()
    resized_masks = resize_fn(masks).to(torch.float32)[:, :3] # latent masks only first three channels (all are the same)
    imgs = []
    if should_i_log["input"]:
        imgs.extend(input)
    if should_i_log["reconstruction"]:
        imgs.extend(reconstruction)
    if should_i_log["mask"]:
        imgs.extend(resized_masks)
    if should_i_log["masked_input"]:
        imgs.extend(reconstruction * (1 - resized_masks))
    if should_i_log["output"]:
        imgs.extend(output)
    if should_i_log["foreground"]:
        foreground = torch.zeros_like(output)
        foreground += input * resized_masks # some weighting of input
        imgs.extend(foreground)
    imgs = torch.stack(imgs)
    grid = make_grid(imgs, nrow=len(input))#len(imgs)//len(input))
    img = torchvision.transforms.ToPILImage()(grid)
    logger.info(f"Saving results to {save_path}")
    img.save(os.path.join(save_path, f"batch_{iteration:05d}.png"))


def compute_inpainting_mask(mask_suggestor, samples, caption_mask):
    # compute segmentation mask
    x0_unaltered = samples["image"]
    masks = samples["preliminary_mask"]
    masks_gmm = []
    ldm_in = torch.full((len(x0_unaltered), 3, 512, 512), -1, dtype=torch.float32)
    img_slices = []
    for i in range(len(x0_unaltered)):
        img = x0_unaltered[i]

        img_ldm = resize_long_edge(img, size_long_edge=512)
        ldm_slice = [slice(None), slice(0,img_ldm.size()[-2]), slice(0, img_ldm.size()[-1])]
        img_slices.append(ldm_slice)
        ldm_in[i][ldm_slice] = img_ldm

        mask = mask_suggestor(masks[i], key=None)
        masks_gmm.append(mask.unsqueeze(dim=0).unsqueeze(dim=0))

    samples["resized_input"] = ldm_in
    samples["image_slices"] = img_slices
    mask = torch.cat(masks_gmm)

    if caption_mask == "fg":
        mask = repeat(mask, "b 1 h w -> b c h w", c=4)
        return mask
    if caption_mask == "full":
        return torch.ones((len(x0_unaltered), 4, 64, 64), dtype=torch.float32)
    else:
        raise NotImplementedError("Only 'full' and 'fg' inpainting implemeted for inference!")


def main(opt, opt_path):
    logger.info(f"Starting experiment {__file__} with name: {args.NAME}")
    if args.save_output_dir is not None:
        logger.info(f"Saving all output to {args.save_output_dir}")
    limit_dataset = None if args.start is None else [args.start, args.stop]
    cfg_scale = args.scale

    if args.synthesis_caption_mask is not None:
        synthesis_caption_mask = args.synthesis_caption_mask
    else:
        synthesis_caption_mask = opt.synthesis_caption_mask

    if args.caption is not None:
        caption = opt.background_prompt if args.caption == "bg" else opt.foreground_prompt
    else:
        caption = opt.background_prompt if synthesis_caption_mask == "fg" else opt.foreground_prompt


    dataset = FobaSynthesisDataset(opt_path,
                             caption=caption,
                             caption_mask=synthesis_caption_mask,
                             output_size=512,
                             limit_dataset=limit_dataset,
                    )
    logger.info(f"Length of dataset: {len(dataset)}, limit_dataset: {limit_dataset}")
    logger.info(f"Running with caption: {caption}, mask:{synthesis_caption_mask}, num_steps: {opt.synthesis_steps}, cfguidance: {cfg_scale}")

    # segmentation masks
    dataset.ds.add_preliminary_masks()
    mask_suggestor = GMMMaskSuggestor(opt)

    # Diffusion Model
    config = OmegaConf.load(f"{opt.config}")
    config["model"]["params"]["use_ema"] = True
    model = load_model_from_config(config, f"{opt.ckpt_ft}")
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    if args.use_plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.out_dir, exist_ok=True)

    batch_size = 3#opt.batch_size
    precision_scope = autocast

    #unseed everything
    seed_everything(time.time())

    # visualization args

    if args.log_all:
        should_i_log = {
            "input": True,
            "reconstruction": True,
            "mask":True,
            "masked_input": True,
            "output": True,
            "foreground":True,
        }
    else:
        should_i_log = {
            "input": False,
            "reconstruction": False,
            "mask":False,
            "masked_input": False,
            "output": False,
            "foreground":False,
        }

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_path = os.path.join(opt.log_dir, args.NAME + now)
    os.makedirs(save_path)
    logger.info(f"Logging to {save_path}")
    topil = ToPILImage()

    cnt = 0
    for rr in tqdm(np.arange(1000)):
        for i in np.arange(0, len(dataset), batch_size):
            with torch.no_grad():
                with precision_scope("cuda"):
                    with model.ema_scope():
                        samples = {}#{"image":[], "item":[], "slice":[], "caption_mask":[], "rel_path":[]}
                        for s in range(i, min(i + batch_size, len(dataset))):
                            sample = dataset[s]
                            if samples == {}:
                                samples = {k: [] for k in sample.keys()}
                            for k, v in sample.items():
                                samples[k].append(v)
                        b = len(samples["image"])

                        # computes resized_input as well
                        mask = compute_inpainting_mask(mask_suggestor, samples, synthesis_caption_mask).to(torch.float32).round()

                        x0_posterior = model.encode_first_stage(samples["resized_input"].to(torch.float32).to("cuda"))
                        x0 = model.get_first_stage_encoding(x0_posterior)

                        c = model.get_learned_conditioning([dataset.caption,] * b)
                        cc = None
                        if cfg_scale != 1.:
                            counter_caption = opt.background_prompt if args.caption == "fg" else opt.foreground_prompt
                            cc = model.get_learned_conditioning([counter_caption,] * b)

                        start_code = torch.randn([b, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)
                        samples_ddim, _ = sampler.sample(
                                                    S=opt.synthesis_steps,
                                                                     conditioning=c[:b],
                                                                     batch_size=b,
                                                                     shape=x0.size()[1:],
                                                                     verbose=False,
                                                                     unconditional_guidance_scale=cfg_scale,
                                                                     unconditional_conditioning=cc,
                                                                     eta=opt.ddim_eta,
                                                                     x_T=start_code[:b],
                                                                     mask=1 - mask.to(device),
                                                                     x0=x0,
                            )

                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        output = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)


                        if i < 10:
                            reconstruction = torch.clamp((model.decode_first_stage(x0) + 1.0) / 2.0, min=0.0, max=1.0)
                            log_images(save_path, should_i_log={k:"True" for k in should_i_log}, input=samples["resized_input"].cpu(), masks=mask.cpu(), output=output.cpu(),reconstruction=reconstruction.cpu(), iteration=cnt)

                        for j, path in enumerate(samples["rel_path"]):
                            if not args.save_output_dir:
                                result_path = os.path.join(opt.base_dir, f"{args.NAME}", path)
                                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                                img_save_path = result_path + '_synth.png'
                                logger.info(f"Saving sample inpainted sample to {img_save_path}")
                                topil(output[j]).save(result_path + "_synth.png")
                            else:
                                os.makedirs(args.save_output_dir, exist_ok=True)
                                img_cnt = cnt * 3 + j
                                img_save_path = os.path.join(args.save_output_dir, f"{img_cnt:06d}.png")
                                logger.info(f"Saving sample inpainted sample to {img_save_path}")
                                topil(output[j]).save(img_save_path)

                        # fastest way to save(probably)
                        #for j, path in enumerate(samples["rel_path"]):
                        #    # reconstruction of original image is lossy --> maybe save this as well and train on this
                        #    reconstruction = torch.clamp((model.decode_first_stage(x0) + 1.0) / 2.0, min=0.0, max=1.0)

                        #    os.makedirs(os.path.dirname(path), exist_ok=True)
                        #    result_path = os.path.join(opt.base_dir, f"{args.NAME}", path)
                        #    reconstruction_path = os.path.join(opt.base_dir, "background_reconstruction", path)
                        #    os.makedirs(os.path.dirname(result_path), exist_ok=True)
                        #    os.makedirs(os.path.dirname(reconstruction_path), exist_ok=True)
                        #    logger.info(f"Saving output to {result_path + '.pt'}")
                        #    torch.save(reconstruction[j][samples["image_slices"][j]].cpu(), reconstruction_path + ".pt")
                        #    torch.save(output[j][samples["image_slices"][j]].cpu(), result_path + ".pt") #os.path.join(save_path, path.replace("/", "___")))
                        cnt += 1


if __name__ == '__main__':
    args = get_compute_background_args()
    exp_config = make_exp_config(args.EXP_PATH)

    # make log dir
    if not os.path.exists(exp_config.log_dir): os.mkdir(exp_config.log_dir)
    log_dir = os.path.join(exp_config.log_dir, exp_config.name, datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    os.makedirs(log_dir)
    shutil.copy(exp_config.__file__, log_dir)
    exp_config.log_dir = log_dir

    logger.info(f"Executing: {__file__} - with args: {args}")
    exp_config.args = args
    main(exp_config, args.EXP_PATH)