from .cars import CarDataset
from .birds import BirdDataset
from .human import HumanDataset
from .dogs import DogDataset
from .chest import ChestXrayDataset


def get_dataset(opt):
    datasets = {"bird":BirdDataset, "human36": HumanDataset, "dog":DogDataset, "car": CarDataset, "chestxray": ChestXrayDataset}
    assert opt.dataset in datasets.keys(), f"Dataset has to be one of: {datasets.keys()}"
    dataset = datasets[opt.dataset](opt, opt.H, opt.W, mask_dir=opt.out_dir)
    return dataset