import os
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import torch
from log import logger
from einops import reduce, rearrange, repeat
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt


def model_out_to_grid(model, x, slice=None):
    x_decoded = []
    for i in range(len(x)):
        decoded = model.decode_first_stage(x[i:i+1].to("cuda"))
        if slice is not None:
            decoded = decoded[slice]

        decoded = ((decoded - decoded.min()) / ( decoded.max() - decoded.min())   * 255.)
        #decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0) * 255.
        #decoded = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0) * 255.
        decoded = decoded.cpu()
        x_decoded.append(decoded)
    x_decoded = torch.cat(x_decoded)
    return x_decoded


def model_to_viz(model, x, out_dir, file_name, nrow=1, slice=None):
    grid = torch.stack(x, 0).squeeze(dim=1)
    grid = model_out_to_grid(model, grid, slice)
    grid = make_grid(grid, nrow=nrow)
    grid = rearrange(grid, "c h w -> h w c")
    img = Image.fromarray(grid.numpy().astype(np.uint8))
    if not file_name.endswith(".png"):
        file_name = file_name + ".png"
    out_file = os.path.join(out_dir, file_name)
    img.save(out_file)
    logger.info(f"Saving output to {out_file}")

def draw_grid(t, title, nrows, ncols, img_size_mult=4):
    plt.clf()
    fig = plt.figure(figsize=(img_size_mult * ncols, img_size_mult * nrows))
    fig.suptitle(title, fontsize=img_size_mult * 10)

    grid = ImageGrid(fig, 111, nrows_ncols=(nrows, ncols), axes_pad=0.05)
    for ax, im in zip(grid, t):
        if im.ndim == 3:
            if im.size()[0] == 3:
                im = rearrange(im, "c h w -> h w c")

        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(im.cpu(), cmap="Greys_r")
    plt.savefig(f"{title.replace(' ', '_')}.pdf")


def get_latent_slice(batch, opt):
    ds_slice = []
    for slice_ in batch["slice"]:
        if slice_.start is None:
            ds_slice.append(slice(None, None, None))
        else:
            ds_slice.append(slice(slice_.start // opt.f, slice_.stop // opt.f, None))
    return tuple(ds_slice)


def load_attention_maps(dataset, n):
    attention_maps = {}
    for i in range(n):
        img = dataset[i]
        path = os.path.join(dataset.mask_dir, img["rel_path"] + ".pt")

        attention = torch.load(path)
        attention_maps[i] = attention
    return attention_maps


def downsample_img_to_latent(img, opt):
    return reduce(img, 'b c (h h2) (w w2) -> b c h w', reduction='mean', h2=opt.f, w2=opt.f)