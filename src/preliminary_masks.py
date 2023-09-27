import numpy as np
from einops import reduce, rearrange, repeat
import torch
import pickle
from functools import partial


class AttentionExtractor():
    def __init__(self, function=None, *args, **kwargs):
        if isinstance(function, str):
            self.reduction_function = getattr(self, function)
        else:
            self.reduction_function = function

        if args or kwargs:
            self.reduction_function = partial(self.reduction_function, *args, **kwargs)

    def __call__(self, inp, *args, **kwargs):
        """ Called with: Iterations x Layers x Channels X Height x Width

        :param inp: tensor
        :return: attention map
        """
        assert inp.ndim == 5
        out = self.reduction_function(inp, *args, **kwargs)
        assert out.ndim == 5
        return out

    def all_mean(self, inp):
        return inp.mean(dim=(0, 1, 2), keepdim=True)

    def diffusion_steps_mean(self, x, steps):
        assert x.size()[2] == 1
        return x[-steps:, :, 0].mean(dim=(0, 1), keepdim=True)

    def relevant_token_step_mean(self, x, tok_idx, steps):
        return x[-steps:, :, tok_idx:(tok_idx+1)].mean(dim=(0, 1), keepdim=True)

    def all_token_mean(self, x, steps):
        return x[-steps:].mean(dim=(0, 1), keepdim=True)

    def multi_relevant_token_step_mean(self, x, tok_idx, steps):
        res = None
        for tok_id in tok_idx:
            if res is None:
                res = x[-steps:, :, tok_id:(tok_id+1)].mean(dim=(0, 1), keepdim=True)
            else:
                res += x[-steps:, :, tok_id:(tok_id+1)].mean(dim=(0, 1), keepdim=True)

        res = res.mean(dim=(0,1), keepdim=True)
        return res


def get_attention_masks(data_obj, attention_dir):
    path = data_obj["seg_path"].replace("segmentation", attention_dir)
    path = path[:-4] + ".pkl"
    with open(path, "rb") as fp:
        obj = pickle.load(fp)
    return obj


def print_attention_info(attention):
    print(f"Num Forward passes: {len(attention)}, Depth:{len(attention[0])}")
    for i in range(len(attention[0])):
        print(f"Layer: {i} - {attention[0][i].size()}")


def reorder_attention_maps(attention):
    for i in range(len(attention)):
        for j in range(len(attention[i])):
            layer = attention[i][j]
            map_size = int(np.sqrt(layer.size()[-2]))
            layer = rearrange(layer.squeeze(dim=1), 'b (h w) tok -> b tok h w', h=map_size, w=map_size)
            attention[i][j] = layer
    return attention


def normalize_attention_map_size(attention_maps):
    for iteration in range(len(attention_maps)): # trough layers / diffusion steps
        for layer in range(len(attention_maps[iteration])):
            attention_map = attention_maps[iteration][layer]# B x num_resblocks x numrevdiff x H x W
            if attention_map.size()[-1] != 64:
                upsampling_factor = 64 // attention_map.size()[-1]
                attention_map = repeat(attention_map, 'b tok h w -> b tok (h h2) (w w2)',h2=upsampling_factor, w2=upsampling_factor)
            attention_maps[iteration][layer] = rearrange(attention_map, "b tok h w -> b 1 tok h w")

        attention_maps[iteration] = rearrange(torch.cat(attention_maps[iteration], dim=1), "b depth tok h w -> b 1 depth tok h w")
    attention_maps = torch.cat(attention_maps, dim=1)
    return attention_maps


def get_latent_slice(batch, opt):
    ds_slice = []
    for slice_ in batch["slice"]:
        if slice_.start is None:
            ds_slice.append(slice(None, None, None))
        else:
            ds_slice.append(slice(slice_.start // opt.f, slice_.stop // opt.f, None))
    return tuple(ds_slice)