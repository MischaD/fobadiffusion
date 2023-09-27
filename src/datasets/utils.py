import torch
import scipy.ndimage as ndimage
import cv2
from einops import rearrange, repeat, reduce


def path_to_tensor(path):
    img = torch.tensor(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), dtype=torch.float32)
    img = ((img / 127.5) - 1)
    img = rearrange(img, "h w c -> 1 c h w")
    return img


def file_to_list(path):
    lines = []
    with open(path) as fp:
        lines = fp.readlines()
    return lines


def resize(img, tosize):
    """resize height and width of dataset that is too large"""
    assert img.ndim == 4
    b, c, h, w = img.size()
    max_size = max(h, w)

    zoom_factor = tosize / max_size

    return torch.tensor(ndimage.zoom(img, (1, 1, zoom_factor,zoom_factor)))
