import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
from torchvision import transforms
from einops import rearrange, repeat
from ldm.util import instantiate_from_config
from datasets import load_dataset
import pandas as pd
from src.datasets import get_dataset
from src.foreground_masks import GMMMaskSuggestor
from utils import make_exp_config
from utils import DatasetSplit


class JSONDataset(Dataset):
    """Dataset from JSON files
    JSON must contain a list of dicts, each dict must contain a path to an image ('imgpath') and a caption('caption').
    """
    def __init__(self, json_path, image_key='imgpath', caption_key='caption', image_transforms=None, mode='train', output_size=512):
        self.json_path = Path(json_path)
        self.image_key = image_key
        self.caption_key = caption_key
        self.mode = mode
        self.output_size = output_size

        if image_transforms is not None:
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms

        with open(self.json_path, 'r') as fp:
            self.data = json.load(fp)
        
        self.keys = list(self.data.keys())
        self.test_keys = self.keys[:1000] # remove the 1000 first keys for testing
        self.train_keys = self.keys[1000:]
        if self.mode == 'test':
            self.keys = self.test_keys
        elif self.mode == 'train':
            self.keys = self.train_keys

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        caption = self.data[key][self.caption_key]
        imgpath = self.data[key][self.image_key]

        if self.mode == 'test':
            im = torch.zeros(3, self.output_size, self.output_size)
            im = rearrange(im * 2. - 1., 'c h w -> h w c')
        else:
            im = Image.open(imgpath)
            # Squarify by padding with black to keep aspect ratio + all the data
            if im.size[0] < im.size[1]:
                new_im = Image.new("RGB", (im.size[1], im.size[1]))
                new_im.paste(im, (0, (im.size[1] - im.size[0]) // 2))
            elif im.size[0] > im.size[1]:
                new_im = Image.new("RGB", (im.size[0], im.size[0]))
                new_im.paste(im, ((im.size[0] - im.size[1]) // 2, 0))
            im = self.process_im(im)

        return {"image": im, "txt": caption}
    
    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


class BirdsDataset(Dataset):
    def __init__(self, csv_path, image_transforms=None, output_size=512, mode='train'):
        self.csv_path = Path(csv_path)
        self.data = pd.read_csv(self.csv_path)
        self.output_size = output_size
        self.mode = mode

        self.captions = self.data['description'].values
        self.imgpaths = self.data['path'].values

        if image_transforms is not None:
            image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        else:
            image_transforms = []
        image_transforms.extend([transforms.ToTensor(),
                            transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms


    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        imgpath = self.imgpaths[idx]

        if self.mode == 'test':
            im = torch.zeros(3, self.output_size, self.output_size)
            im = rearrange(im * 2. - 1., 'c h w -> h w c')
        else:
            im = Image.open(imgpath)
            im = self.process_im(im)
            
        return {"image": im, "txt": caption}

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)


class FolderData(Dataset):
    def __init__(self, root_dir, caption_file, image_transforms, ext="jpg") -> None:
        self.root_dir = Path(root_dir)
        with open(caption_file, "rt") as f:
            captions = json.load(f)
        self.captions = captions

        self.paths = list(self.root_dir.rglob(f"*.{ext}"))
        image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
        image_transforms.extend([transforms.ToTensor(),
                                 transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        image_transforms = transforms.Compose(image_transforms)
        self.tform = image_transforms

        # assert all(['full/' + str(x.name) in self.captions for x in self.paths])

    def __len__(self):
        return len(self.captions.keys())

    def __getitem__(self, index):
        chosen = list(self.captions.keys())[index]
        im = Image.open(self.root_dir/chosen)
        im = self.process_im(im)
        caption = self.captions[chosen]
        if caption is None:
            caption = "old book illustration"
        return {"jpg": im, "txt": caption}

    def process_im(self, im):
        im = im.convert("RGB")
        return self.tform(im)

def hf_dataset(
    name,
    image_transforms=[],
    image_column="image",
    text_column="text",
    split='train',
    image_key='image',
    caption_key='txt',
    ):
    """Make huggingface dataset with appropriate list of transforms applied
    """
    ds = load_dataset(name, split=split)
    image_transforms = [instantiate_from_config(tt) for tt in image_transforms]
    image_transforms.extend([transforms.ToTensor(),
                                transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
    tform = transforms.Compose(image_transforms)

    assert image_column in ds.column_names, f"Didn't find column {image_column} in {ds.column_names}"
    assert text_column in ds.column_names, f"Didn't find column {text_column} in {ds.column_names}"

    def pre_process(examples):
        processed = {}
        processed[image_key] = [tform(im) for im in examples[image_column]]
        processed[caption_key] = examples[text_column]
        return processed

    ds.set_transform(pre_process)
    return ds



class TextOnly(Dataset):
    def __init__(self, captions, output_size, image_key="image", caption_key="txt", n_gpus=1):
        """Returns only captions with dummy images"""
        self.output_size = output_size
        self.image_key = image_key
        self.caption_key = caption_key
        if isinstance(captions, Path):
            self.captions = self._load_caption_file(captions)
        else:
            self.captions = captions

        if n_gpus > 1:
            # hack to make sure that all the captions appear on each gpu
            repeated = [n_gpus*[x] for x in self.captions]
            self.captions = []
            [self.captions.extend(x) for x in repeated]

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        dummy_im = torch.zeros(3, self.output_size, self.output_size)
        dummy_im = rearrange(dummy_im * 2. - 1., 'c h w -> h w c')
        return {self.image_key: dummy_im, self.caption_key: self.captions[index]}

    def _load_caption_file(self, filename):
        with open(filename, 'rt') as f:
            captions = f.readlines()
        return [x.strip('\n') for x in captions]