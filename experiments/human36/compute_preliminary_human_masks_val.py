from utils import DatasetSplit, get_tok_idx
import os
from src.preliminary_masks import AttentionExtractor

base_dir="/vol/ideadata/ed52egek/data/fobadiffusion/h36preprocessed"
out_dir = os.path.join(base_dir, "preliminary_masks/","compute_preliminary_human_masks") #os.path.basename(__file__).rstrip('.py'))

rev_diff_steps = 40
num_repeat_each_diffusion_step = 1
prompt = "a photo of a human with arms and legs"
foreground_prompt = "a photo of a human"
background_prompt = "a photo of a background"
goal_prompt = "human" # object we want to detect
attention_extractor = AttentionExtractor("multi_relevant_token_step_mean", tok_idx=[5,7,9], steps=40)
latent_attention_masks = True

dataset = "human36"
dataset_args = dict(
    base_dir=base_dir,
    limit_dataset=[0, 100],  # limit dataset, interpeted as array indices (i.e. excludes upper limit)
    split=DatasetSplit("val"),#######
    shuffle=True,
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
config = "/vol/ideadata/ed52egek/pycharm/foba/experiments/configs/v1-attentionviz.yaml"
ckpt_ft = "/vol/ideadata/ed52egek/pycharm/foba/models_finetuned/humans/human_bg_200epochs.ckpt"

synthesis_caption_mask = "fg" # full, fg
synthesis_steps=50
exp_name="human_train"# inpainted masks

# dataloading
batch_size=7
num_workers=5

# logging
log_dir = "./log/"