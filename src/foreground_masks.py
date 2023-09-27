import os.path

import torch
from einops import rearrange
import numpy as np
from sklearn.mixture import GaussianMixture
from utils import resize_long_edge
import torchvision
from scipy.ndimage import binary_fill_holes, binary_closing


class GMMMaskSuggestor:
    def __init__(self, opt):
        self.opt = opt
        self.gmm = GaussianMixture(n_components=2)

    def filter_orphan_pixel(self, img):
        assert len(img.size()) == 2
        img = rearrange(img, "h w -> 1 1 h w").to(torch.float32)
        weights = torch.full((1, 1, 3, 3), 1/9)
        img[...,1:-1,1:-1] = torch.nn.functional.conv2d(img, weights.to(img.device), bias=torch.zeros(1, device=img.device))
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        return img.squeeze()

    def get_gaussian_mixture_prediction(self, mask):
        device = mask.device
        self.gmm.fit(rearrange(mask.to("cpu"), "h w -> (h w) 1"))
        threshold = np.mean(self.gmm.means_)
        binary_img = mask > threshold
        return binary_img.to(device)

    def get_rectangular_inpainting_mask(self, segmentation_mask):
        """ Get rectangular map of where to inpaint. Can overlap with object. In this case the object will not be inpainted and multiple inpainting spots are generate around the object.
        Input mask should contain False at all background and True at all Foreground pixels.
        Output will be mask that is True where inpainting is possible (part of background) and False where inpainting is not possible)

        :param segmentation_mask: binary mask where True is the foreground and False is the background we want to sample from
        :return:
        """
        binary_mask = segmentation_mask
        inpainting_mask = torch.zeros_like(binary_mask)
        if binary_mask.sum() >= ((binary_mask.size()[0] * binary_mask.size()[1]) - 1):
            # all foreground
            return inpainting_mask

        x, y = np.where(binary_mask == False)
        number_of_retries = 100  # after some attempts falls back to inpainting whole background
        while number_of_retries > 0:
            random_corner = np.random.randint(0, len(x))
            random_other_corner = np.random.randint(0, len(x))
            if random_corner == random_other_corner:
                continue

            # tl corner
            tl = (min(x[random_corner], x[random_other_corner]),
                  min(y[random_corner], y[random_other_corner]))

            # br corner
            br = (max(x[random_corner], x[random_other_corner]),
                  max(y[random_corner], y[random_other_corner]))

            width = br[0] - tl[0]
            height = br[1] - tl[1]
            area = width * height
            is_not_large_enough = (width <= 10 or height <= 10 or area < 16 ** 2)
            is_too_large = (width > 32 and height > 32) or area > 32 ** 2
            if (is_not_large_enough or is_too_large) and number_of_retries >= 0:
                number_of_retries -= 1
                continue

            box_location = torch.zeros_like(binary_mask)
            slice_ = (slice(tl[0], (br[0] + 1)), slice(tl[1], (br[1] + 1)))
            box_location[slice_] = True

            background_pixels = np.logical_and(np.logical_not(binary_mask), box_location)

            ratio = background_pixels.sum() / area
            if ratio < 2/3 and number_of_retries >= 0:
                # too many foreground pixels
                number_of_retries -= 1
                continue
            else:
                inpainting_mask = background_pixels
                break
        if number_of_retries == 0:
            inpainting_mask = np.logical_not(segmentation_mask)
        return inpainting_mask.to(bool)

    def __call__(self, sample, key="preliminary_mask"):
        if key is None:
            prelim_mask = sample
        else:
            prelim_mask = sample[key]
        prelim_mask = self.get_gaussian_mixture_prediction(prelim_mask.squeeze())
        orphan_filtered = self.filter_orphan_pixel(prelim_mask)
        return orphan_filtered.to(bool)

    def refined_mask_suggestion(self, sample):
        if sample.get("preliminary_mask") is None:
            raise ValueError("Preliminary mask not part of sample dict - please call FOBADataset.add_preliminary_masks")
        if sample.get("inpainted_image") is None:
            raise ValueError("Inpainted Image not part of sample dict - please call FOBADataset.add_inpaintings")

        tmp_mask_path = os.path.join(self.opt.base_dir, "refined_mask_tmp", sample["rel_path"] + ".pt")
        if os.path.isfile(tmp_mask_path):
            return torch.load(tmp_mask_path)

        # get original image
        original_image = sample["img"]
        original_image = (original_image + 1) / 2
        y_resized = resize_long_edge(original_image, 512) # resized s.t. long edge has lenght 512
        #y = torch.zeros((1, 3, 512, 512))
        #y[:, :, :y_resized.size()[-2], :y_resized.size()[-1]]

        # get preliminary mask
        resize_to_img_space = torchvision.transforms.Resize(512)
        prelim_mask = resize_to_img_space(self(sample, "preliminary_mask").unsqueeze(dim=0))

        # get inpainted image
        inpainted = sample["inpainted_image"]

        # get gmm diff mask
        diff = (abs(y_resized - inpainted))
        diff = rearrange(diff, "1 c h w -> 1 h w c")
        diff_mask = self(diff.mean(dim=3), key=None).unsqueeze(dim=0)

        prelim_mask = prelim_mask[:, :diff_mask.size()[-2], :diff_mask.size()[-1]]
        refined_mask = prelim_mask * diff_mask
        refined_mask = refined_mask.unsqueeze(dim=0)
        os.makedirs(os.path.dirname(tmp_mask_path), exist_ok=True)
        torch.save(refined_mask, tmp_mask_path)
        return refined_mask

    def _compute_post_processed(self, refined_mask):
        refined_mask = refined_mask.squeeze()
        refined_mask = torch.tensor(binary_fill_holes(binary_closing(refined_mask.cpu()))).to("cuda")
        refined_mask = rearrange(refined_mask, "h w -> 1 1 h w ")
        return refined_mask

    def postprocessd_refined_mask_suggestion(self, sample):
        refined_mask = self.refined_mask_suggestion(sample)
        return self._compute_post_processed(refined_mask)


"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from src.datasets import get_dataset
from utils import make_exp_config
from src.preliminary_masks import get_latent_slice
from einops import reduce, rearrange, repeat

EXP_PATH = "../experiments/human36/compute_preliminary_human_masks_batch_1_arms_and_legs.py"
opt= make_exp_config(EXP_PATH)
opt.dataset_args["load_segmentations"] = True
dataset = get_dataset(opt)
dataset.add_preliminary_masks()

mask_suggestor = GMMMaskSuggestor(opt)
sample = dataset[0]
binary_mask = mask_suggestor(sample)
plt.imshow(sample["preliminary_mask"].squeeze())
plt.show()
plt.imshow(binary_mask)
plt.show()
fig, axs = plt.subplots(3, 3)
for i in range(3):
    for j in range(3):
        axs[i,j].imshow(mask_suggestor.get_rectangular_inpainting_mask(binary_mask))
plt.show()
"""