from enum import Enum
import argparse
from importlib.machinery import SourceFileLoader
import importlib
import torch
import numpy as np
from einops import rearrange
from scipy import ndimage
import torchvision


def get_compute_mask_args():
    parser = argparse.ArgumentParser(description="Compute Masks")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--prompt", type=str, default=None)
    return parser.parse_args()

def get_train_segmentation_refined():
    parser = argparse.ArgumentParser(description="Compute Unet Refined")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("--exp_name", type=str, default=None, help="Path to experiment files")
    parser.add_argument("--postprocess", action="store_true", default=False)
    parser.add_argument("--test_only", action="store_true", default=False)
    parser.add_argument("--bbox_mode", action="store_true", default=False)
    parser.add_argument("--ckpt_path", default=None)
    return parser.parse_args()

def get_compute_background_args():
    parser = argparse.ArgumentParser(description="Compute Background")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("NAME", type=str, help="Name of experiment - will be used to save masks")
    parser.add_argument("--log_all", action="store_true", default=False, help="logs all information s.a. mask, input, reconstruction ")
    parser.add_argument("--save_output_dir", default=None, help="Save all output to single outdir")
    parser.add_argument("--use_plms", action="store_true", default=False, help="Use plms sampling")
    parser.add_argument("--start", type=int, default=None, help="first sample to generate, inclusive")
    parser.add_argument("--stop", type=int, default=None, help="last sample to generate, exclusive")
    parser.add_argument("--synthesis_caption_mask", type=str, default=None, choices=["fg", "bg", "full"], help="last sample to generate, exclusive")
    parser.add_argument("--caption", type=str, default=None, choices=["fg", "bg"], help="Diffusion Model Prompt - either foreground prompt or background prompt")
    parser.add_argument("--scale", type=float, default=1., help="Classifier free guidance scale")
    return parser.parse_args()

def get_inpaint_baseline_args():
    parser = argparse.ArgumentParser(description="Compute Background")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment file")
    parser.add_argument("NAME", type=str, help="Name of experiment - will be used to save masks")
    parser.add_argument("--log_all", action="store_true", default=False, help="logs all information s.a. mask, input, reconstruction ")
    parser.add_argument("--use_plms", action="store_true", default=False, help="Use plms sampling")
    parser.add_argument("--start", type=int, default=None, help="first sample to generate, inclusive")
    parser.add_argument("--stop", type=int, default=None, help="last sample to generate, exclusive")
    return parser.parse_args()

def make_exp_config(exp_file):
    # get path to experiment
    exp_name = exp_file.split('/')[-1].rstrip('.py')

    # import experiment configuration
    exp_config = SourceFileLoader(exp_name, exp_file).load_module()
    exp_config.name = exp_name
    return exp_config


def resize_to(img, tosize):
    assert img.ndim == 4
    b, c, h, w = img.size()
    max_size = max(h, w)

    zoom_factor = tosize / max_size

    return torch.tensor(ndimage.zoom(img, (1, 1, zoom_factor,zoom_factor)))

class DatasetSplit(Enum):
    train="train"
    test="test"
    val="val"
    all="all"

def resize_long_edge(img, size_long_edge):
    # torchvision resizes so shorter edge has length - I want longer edge to have spec. length
    assert img.size()[-3] == 3, "Channel dimension expected at third position"
    img_longer_edge = max(img.size()[-2:])
    img_shorter_edge = min(img.size()[-2:])
    resize_factor = size_long_edge / img_longer_edge

    # resized_img = torchvision.transforms.functional.resize(img_longer_edge/img_shorter_edge)
    resize_to = img_shorter_edge * resize_factor
    resizer = torchvision.transforms.Resize(size=round(resize_to))
    return resizer(img)[..., :size_long_edge, :size_long_edge]


SPLIT_TO_DATASETSPLIT = {0:DatasetSplit("test"), 1:DatasetSplit("train"), 2:DatasetSplit("val")}


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


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

def get_tok_idx(prompt, obj):
    object_categories = prompt.split(" ")
    tok_idx = [i for i in range(len(object_categories)) if object_categories[i] == obj][0]
    return tok_idx + 1

def img_to_viz(img):
    img = rearrange(img, "1 c h w -> h w c")
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = np.array(((img + 1) * 127.5), np.uint8)
    return img


def collate_batch(batch):
    # make list of dirs to dirs of lists with batchlen
    batched_data = {}
    for data in batch:
        # label could be img, label, path, etc
        for key, value in data.items():
            if batched_data.get(key) is None:
                batched_data[key] = []
            batched_data[key].append(value)

    # cast to torch.tensor
    for key, value in batched_data.items():
        if isinstance(value[0],torch.Tensor):
            if value[0].size()[0] != 1:
                for i in range(len(value)):
                    value[i] = value[i][None,...]
            # check if concatenatable
            if all([value[0].size() == value[i].size() for i in range(len(value))]):
                batched_data[key] = torch.concat(batched_data[key])
    return batched_data


def update_matplotlib_font(fontsize=11, fontsize_ticks=8, tex=True):
    import matplotlib.pyplot as plt
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": tex,
        "font.family": "serif",
        # Use 11pt font in plots, to match 11pt font in document
        "axes.labelsize": fontsize,
        "font.size": fontsize,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": fontsize_ticks,
        "xtick.labelsize": fontsize_ticks,
        "ytick.labelsize": fontsize_ticks
    }
    plt.rcParams.update(tex_fonts)


def set_size(width, fraction=1, subplots=(1, 1), ratio= (5**.5 - 1) / 2):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == "MICCAI":
        width_pt = 347.12354
    elif width == "AAAI":
        width_pt = 505.89
    elif width == "AAAISingleCol":
        width_pt = 239.39438
    elif width == "NEURIPS":
        width_pt = 397.48499
    elif width == "ICCV":
        width_pt = 496.85625
    elif width == "ICCVSingleCol":
        width_pt = 237.13594
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    ratio

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)
