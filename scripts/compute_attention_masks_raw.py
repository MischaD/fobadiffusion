import argparse
import shutil
import time
import os
import pickle
import numpy as np
import pytorch_lightning as pl
import datetime
import torchvision
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
from utils import get_compute_mask_args, make_exp_config, load_model_from_config, collate_batch, img_to_viz
from einops import reduce, rearrange, repeat
from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.util import AttentionSaveMode
from ldm.models.diffusion.plms import PLMSSampler
from src.preliminary_masks import reorder_attention_maps, normalize_attention_map_size
from ldm.models.diffusion.ddim import DDIMSampler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


def get_latent_slice(batch, opt):
    ds_slice = []
    for slice_ in batch["slice"]:
        if slice_.start is None:
            ds_slice.append(slice(None, None, None))
        else:
            ds_slice.append(slice(slice_.start // opt.f, slice_.stop // opt.f, None))
    return tuple(ds_slice)

from torch.utils.data import DataLoader

def add_viz_of_data_and_pred(images, batch, x_samples_ddim, opt):
    # append input
    x0_norm = torch.clamp((batch["x"] + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    x0_norm = reduce(x0_norm, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=opt.f, w2=opt.f)
    images.append(x0_norm)

    # append model output

    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0).cpu()
    images.append(
        reduce(x_samples_ddim, 'b c (h h2) (w w2) -> b c h w', 'mean', h2=opt.f, w2=opt.f))

    # append gt mask
    images.append(
        reduce(batch["segmentation_x"], 'b c (h h2) (w w2) -> b c h w', 'max', h2=opt.f, w2=opt.f))


def main(opt):
    dataset = get_dataset(opt)
    logger.info(f"Length of dataset: {len(dataset)}")

    logger.info(f"=" * 50 + f"Running with prompt: {opt.prompt}" + "="*50)
    attention_save_mode = AttentionSaveMode("cross")
    logger.info(f"enable attention save mode: {attention_save_mode}")
    logger.info(f"Config file: {opt.config}")


    config = OmegaConf.load(f"{opt.config}")
    config["model"]["params"]["use_ema"] = False
    config["model"]["params"]["unet_config"]["params"]["attention_save_mode"] = attention_save_mode
    ckpt = opt.ckpt
    model = load_model_from_config(config, f"{ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    #sampler = DDIMSampler(model)
    sampler = PLMSSampler(model)

    os.makedirs(opt.out_dir, exist_ok=True)

    batch_size = 3#opt.batch_size

    start_code = None
    seed_everything(opt.seed)
    if opt.fixed_code:
        start_code = torch.randn([batch_size, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast

    # visualization args
    rev_diff_steps = opt.rev_diff_steps
    num_repeat_each_diffusion_step = opt.num_repeat_each_diffusion_step

    start_reverse_diffusion_from_t = int((rev_diff_steps - 1) * (sampler.ddpm_num_timesteps // opt.ddim_steps) + 1)

    logger.info(f"Relative path to first sample: {dataset[0]['rel_path']}")
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,#opt.num_workers,
                            collate_fn=collate_batch,
                            )


    # for visualization of intermediate results
    cnt = 0
    topil = torchvision.transforms.ToPILImage()
    resize_to_imag_size = torchvision.transforms.Resize(512)
    pl.seed_everything(52)
    for samples in tqdm(dataloader, "generating masks"):
        with torch.no_grad():
            with precision_scope("cuda"):
                with model.ema_scope():
                    x0_posterior = model.encode_first_stage(samples["x"].to(torch.float32).to(device))
                    x0 = model.get_first_stage_encoding(x0_posterior)
                    mask = torch.ones_like(x0)

                    c = model.get_learned_conditioning(batch_size * [opt.prompt, ])
                    logger.info(f"Start reverse diffusion from {start_reverse_diffusion_from_t}")
                    b = len(x0)
                    samples_ddim, intermediates = sampler.sample_with_attention(
                                                                model=model,
                                                                 t=start_reverse_diffusion_from_t,
                                                                 repeat_steps=num_repeat_each_diffusion_step,
                                                                 S=opt.ddim_steps,
                                                                 conditioning=c[:b],
                                                                 batch_size=b,
                                                                 shape=x0.size(),
                                                                 verbose=False,
                                                                 unconditional_guidance_scale=opt.scale,
                                                                 eta=opt.ddim_eta,
                                                                 x_T=start_code[:len(x0)],
                                                                 mask=mask,
                                                                 x0=x0,
                                                                 )

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1)
                    #x_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                    #if not opt.skip_save:
                    #    for x_sample in x_image_torch:
                    #        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    #        img = Image.fromarray(x_sample.astype(np.uint8))
                            #img.save(os.path.join(s))


                    attention_masks = intermediates["attention"]
                    attention_masks = reorder_attention_maps(attention_masks)
                    attention_masks = normalize_attention_map_size(attention_masks)
                    for i in range(len(attention_masks)):
                        attention_mask = opt.attention_extractor(attention_masks[i])
                        attention_mask = attention_mask.cpu()

                        path = os.path.join(opt.out_dir, samples["rel_path"][i])
                        os.makedirs(os.path.dirname(path), exist_ok=True)

                        # save intermediate attention maps
                        path = os.path.join(opt.out_dir, samples["rel_path"][i]) + ".pt"
                        logger.info(f"Saving attention mask to {path}")
                        torch.save(attention_mask, path)

                        if cnt < 10:
                            log_path = os.path.join(opt.log_dir, os.path.dirname(samples["rel_path"][i]))
                            os.makedirs(log_path, exist_ok=True)
                            log_path = os.path.join(opt.log_dir, samples["rel_path"][i])
                            out_img = (samples["x"][i] + 1.) / 2.
                            attention_mask = (attention_mask - attention_mask.min()) / (attention_mask.max() - attention_mask.min())
                            attention_mask = repeat(resize_to_imag_size(rearrange(attention_mask, "1 1 1 h w -> 1 h w")), "1 h w -> c h w", c=3)
                            logger.info(f"Saving preliminary results to {log_path}")
                            topil(torch.cat([out_img, attention_mask], dim=2)).save(log_path)
                            cnt += 1


if __name__ == '__main__':
    args = get_compute_mask_args()
    exp_config = make_exp_config(args.EXP_PATH)

    # make log dir
    log_dir = os.path.join(exp_config.log_dir, os.path.basename(__file__.rstrip(".py")), datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    os.makedirs(log_dir)
    shutil.copy(exp_config.__file__, log_dir)
    exp_config.log_dir = log_dir

    if args.prompt != None:
        logger.info(f"Overwriting prompt from exp to {args.prompt}")
        exp_config.prompt = args.prompt

    exp_config.args = args
    main(exp_config)