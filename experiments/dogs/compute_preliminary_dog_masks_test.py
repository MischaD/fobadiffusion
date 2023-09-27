from utils import DatasetSplit, get_tok_idx
import os
from src.preliminary_masks import AttentionExtractor

base_dir = "/vol/ideadata/ed52egek/data/fobadiffusion/dogs"
out_dir = os.path.join(base_dir, "preliminary_masks/", "compute_preliminary_dog_masks")

rev_diff_steps = 40
num_repeat_each_diffusion_step = 1

prompt = "a photo of a dog"
goal_prompt = "dog" # object we want to detect
foreground_prompt = "a photo of a dog"
background_prompt = "a photo of a background"
attention_extractor = AttentionExtractor("relevant_token_step_mean", tok_idx=5, steps=40)
latent_attention_masks = True

dataset = "dog"
dataset_args = dict(
    base_dir=base_dir,
    split=DatasetSplit("test"),
    #limit_dataset=[0, 10]#[0,16185]
)

# dataset
C=4 # latent channels
H=512
W=512
f=8

# stable diffusion args
seed=4200
ddim_steps=50
ddim_eta = 0.0 # 0 corresponds to deterministic sampling
fixed_code = True
scale = 1
ckpt = "/vol/ideadata/ed52egek/pycharm/foba/stable-diffusion/sd-v1-4.ckpt"
config = "/home/saturn/iwai/iwai003h/pycharm/foba/experiments/configs/dog_inpainting_hpc.yaml"
ckpt_ft = "/home/saturn/iwai/iwai003h/pycharm/foba/finetune-stable-diffusion/logs/2022-11-06T21-48-25_dog_inpainting_hpc/checkpoints/last.ckpt"
synthesis_steps=50

# dataloading
batch_size=1
num_workers=1

# logging
log_dir = "./log/"