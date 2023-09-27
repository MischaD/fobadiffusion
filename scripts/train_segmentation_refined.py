import wandb
from src.models.segmentation_unet import UNet
import time
import shutil
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import make_grid
import pytorch_lightning as pl
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
from log import logger
from utils import get_compute_mask_args, make_exp_config, collate_batch, get_train_segmentation_refined
from utils import resize_long_edge
from utils import DatasetSplit
from sklearn.metrics import accuracy_score, jaccard_score
from datetime import datetime
import torchvision
from einops import rearrange
from scipy.ndimage import binary_fill_holes, binary_closing
from torch import optim
from pytorch_lightning.callbacks import ModelCheckpoint
from src.foreground_masks import GMMMaskSuggestor
from src.datasets.dataset import FOBADataset
from src.datasets.utils import path_to_tensor
from src.datasets import get_dataset


class FOBAUnetDataset(FOBADataset):
    def __init__(self, opt, H, W, mask_dir=None):
        super().__init__(opt, H, W, mask_dir)
        self.refined_mask_base_dir = os.path.join(self.opt.base_dir, "refined_mask_tmp")
        logger.info(f"Expecting refined masks in: {self.refined_mask_base_dir}")

        self.load_segmentations = opt.dataset_args.get("load_segmentations", False)
        self._build_dataset()

    def _load_images(self, index):
        assert len(index) == 1
        entry = self.data[index[0]].copy()
        img = path_to_tensor(entry["img_path"])

        entry["img"] = img
        entry["refined_mask"] = torch.load(os.path.join(self.refined_mask_base_dir, entry["rel_path"] + ".pt"))

        if self._preliminary_masks_path is not None:
            entry["preliminary_mask"] = torch.load(os.path.join(self._preliminary_masks_path, entry["rel_path"] + ".pt"))

        if self.load_segmentations:
            seg = (path_to_tensor(entry["seg_path"]) + 1) / 2
            entry["segmentation"] = seg
            y = torch.full((1, 3, self.H, self.W), 0.)
            y[0, :, :seg.size()[2], :seg.size()[3]] = seg
            entry["segmentation_x"] = y
        return entry

    def check_all_masks_computed(self):
        for i in range(len(self)):
            entry = self.data[i]
            if not os.path.isfile(os.path.join(self.refined_mask_base_dir, entry["rel_path"] + ".pt")):
                logger.warn(f"File number {i} not found: {os.path.join(self.refined_mask_base_dir, entry['rel_path'] + '.pt')}")
                return False
        logger.info("All refined masks successfully computed")
        return True

    def calculate_refined_masks(self, exp_name):
        self.add_inpaintings(exp_name)
        self.add_preliminary_masks()

        mask_suggestor = GMMMaskSuggestor(self.opt)

        assert self._preliminary_masks_path is not None, "Define preliminary mask path"
        assert self._inpainted_images_path is not None, "Define inpainting image path"
        for i in tqdm(range(len(self)), "computing refined inpainting"):
            sample = self.data[i].copy()
            img = path_to_tensor(sample["img_path"])
            sample["img"] = img
            sample["preliminary_mask"] = torch.load(os.path.join(self._preliminary_masks_path, sample["rel_path"] + ".pt"))
            sample["inpainted_image"] = torch.load(os.path.join(self._inpainted_images_path, sample["rel_path"] + ".pt"))

            tmp_mask_path = os.path.join(self.opt.base_dir, "refined_mask_tmp", sample["rel_path"] + ".pt")
            if os.path.isfile(tmp_mask_path):
                continue

            # get original image
            original_image = sample["img"]
            original_image = (original_image + 1) / 2
            y_resized = resize_long_edge(original_image, 512)  # resized s.t. long edge has lenght 512
            y = torch.zeros((1, 3, 512, 512))
            y[:, :, :y_resized.size()[-2], :y_resized.size()[-1]] = y_resized

            # get preliminary mask
            resize_to_img_space = torchvision.transforms.Resize(512)
            prelim_mask = resize_to_img_space(mask_suggestor(sample, "preliminary_mask").unsqueeze(dim=0))

            # get inpainted image
            inpainted = sample["inpainted_image"]

            # get gmm diff mask
            diff = (abs(y[0, :, :inpainted.size()[-2], :inpainted.size()[-1]] - inpainted))
            diff = rearrange(diff, "c h w -> 1 h w c")
            diff_mask = mask_suggestor(diff.mean(dim=3), key=None).unsqueeze(dim=0)

            prelim_mask = prelim_mask[:, :diff_mask.size()[-2], :diff_mask.size()[-1]]
            refined_mask = prelim_mask * diff_mask
            refined_mask = refined_mask.unsqueeze(dim=0)
            os.makedirs(os.path.dirname(tmp_mask_path), exist_ok=True)
            torch.save(refined_mask, tmp_mask_path)



def get_train_preprocessing_function(postprocess_function):
    transforms = []
    transforms.append(torchvision.transforms.Resize(128))
    transforms.append(torchvision.transforms.RandomCrop(128))
    transforms.append(torchvision.transforms.RandomHorizontalFlip())
    tform = torchvision.transforms.Compose(transforms)

    def preprocess(batch):
        samples = []
        for i in range(len(batch["img"])):
            sample = {k: batch[k][i] for k in batch}
            samples.append(sample)

        batch["x128"] = []
        batch["y128"] = []
        for i in range(len(samples)):
            x_resized = resize_long_edge(samples[i]["img"], 512)
            y = samples[i]["refined_mask"]
            y = postprocess_function(y)
            if x_resized.ndim == 3 and x_resized.size()[0] == 3:
                x_resized = x_resized.unsqueeze(dim=0)
            if y.ndim == 3 and y.size()[0] == 1:
                y = y.unsqueeze(dim=0)
            xcaty = torch.cat([x_resized, y[:, :, :x_resized.size()[-2], :x_resized.size()[-1]]], dim=1)
            xcaty = tform(xcaty)
            batch["x128"].append(xcaty[:, :3])
            batch["y128"].append(xcaty[:, 3:])

        batch["x128"] = torch.cat(batch["x128"]).to("cuda")
        batch["y128"] = torch.cat(batch["y128"]).to("cuda")
        return batch
    return preprocess


def get_val_preprocessing_function(postprocess_function):
    transforms = []
    transforms.append(torchvision.transforms.Resize(128))
    transforms.append(torchvision.transforms.CenterCrop(128))
    tform = torchvision.transforms.Compose(transforms)

    def preprocess(batch):
        samples = []
        for i in range(len(batch["img"])):
            sample = {k: batch[k][i] for k in batch}
            samples.append(sample)

        batch["x128"] = []
        batch["y128"] = []
        for i in range(len(samples)):
            x_resized = resize_long_edge(samples[i]["img"], 512)
            y = samples[i]["refined_mask"]
            y = postprocess_function(y)
            if x_resized.ndim == 3 and x_resized.size()[0] == 3:
                x_resized = x_resized.unsqueeze(dim=0)
            if y.ndim == 3 and y.size()[0] == 1:
                y = y.unsqueeze(dim=0)
            xcaty = torch.cat([x_resized, y[:, :, :x_resized.size()[-2], :x_resized.size()[-1]]], dim=1)
            xcaty = tform(xcaty)
            batch["x128"].append(xcaty[:, :3])
            batch["y128"].append(xcaty[:, 3:])

        batch["x128"] = torch.cat(batch["x128"])
        batch["y128"] = torch.cat(batch["y128"])
        return batch
    return preprocess


def get_test_preprocessing_function():
    transforms = []
    transforms.append(torchvision.transforms.Resize(128))
    transforms.append(torchvision.transforms.CenterCrop(128))
    tform = torchvision.transforms.Compose(transforms)

    def preprocess(sample):
        sample["x128"] = []
        sample["y128"] = []

        for i in range(len(sample["x"])):
            x = sample["img"][i]
            y = sample["segmentation"][i]
            xcaty = torch.cat([x, y], dim=1)
            xcaty = tform(xcaty)
            sample["x128"].append(xcaty[:, :3])
            sample["y128"].append(xcaty[:, 3:])

        sample["x128"] = torch.cat(sample["x128"])
        sample["y128"] = torch.cat(sample["y128"]).mean(dim=1, keepdims=True)
        return sample
    return preprocess

def get_test_bbox_preprocessing_function():
    transforms = []
    transforms.append(torchvision.transforms.Resize(128))
    transforms.append(torchvision.transforms.CenterCrop(128))
    tform = torchvision.transforms.Compose(transforms)

    def preprocess(sample):
        sample["x128"] = []
        sample["y128"] = []

        for i in range(len(sample["x"])):
            x = sample["img"][i]
            y = sample["segmentation"][i]
            xcaty = torch.cat([x, y], dim=1)
            xcaty = tform(xcaty)
            sample["x128"].append(xcaty[:, :3])
            sample["y128"].append(xcaty[:, 3:])

        sample["x128"] = torch.cat(sample["x128"])
        sample["y128"] = torch.cat(sample["y128"]).mean(dim=1, keepdims=True)
        return sample
    return preprocess


class SegmentationUnet(pl.LightningModule):
    def __init__(self, model, train_transform, val_transform, test_transform, bbox_mode):
        super().__init__()
        self.model = model
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform
        self.step = 0
        self.save_hyperparameters()
        self.bbox_mode = bbox_mode
        #self.compute_

    def training_step(self, train_batch, batch_idx):
        x = self.train_transform(train_batch)
        x = train_batch["x128"]
        y = train_batch["y128"]

        x_out = self.model(x)

        self.step += 1
        loss = torch.nn.functional.binary_cross_entropy(x_out, y)

        self.log('global_step', self.step)
        self.log('train/loss', loss)

        if batch_idx == 0:
            self.log_prediction(x, y, x_out)
        return loss

    def validation_step(self, val_batch, batch_idx):
        self.val_transform(val_batch)

        x = val_batch["x128"]
        y = val_batch["y128"]

        x_out = self.model(x)

        loss = torch.nn.functional.binary_cross_entropy(x_out, y)
        self.log("val/loss", loss)

        if batch_idx == 0:
            self.log_prediction(x, y, x_out)
        return {"loss": loss, "x_out": x_out.flatten().cpu(), "y": y.flatten().cpu().to(bool)}

    def log_prediction(self, x, y, x_out, test=False):
        inputs = make_grid(x, nrows=8)
        gt = make_grid(y, nrows=8)
        predictions = make_grid(x_out, nrows=8)
        appendix = "val" if not test else "test"
        wandb.log({"images" + appendix: wandb.Image(inputs.cpu()),
                   "gt_seg" + appendix: wandb.Image(gt.mean(dim=0).cpu().numpy()),
                   "predictions" + appendix: wandb.Image(self.threshold_prediction(predictions.cpu(), th=0.5).to(torch.float32).mean(dim=0).numpy()),
                   })

    def threshold_prediction(self, pred, th):
        pred[pred >= th] = 1
        pred[pred < th] = 0
        pred = pred.to(bool)
        return pred

    def validation_epoch_end(self, validation_step_outputs):
        losses = []
        y_pred = []
        y_true = []
        for step_outputs in validation_step_outputs:
            losses.append(step_outputs["loss"])
            y_pred.extend(step_outputs["x_out"])
            y_true.extend(step_outputs["y"])
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)

        y_pred = self.threshold_prediction(y_pred, 0.5) # sigmoid

        accuracy = accuracy_score(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred)
        miou = (iou + jaccard_score(torch.logical_not(y_true), torch.logical_not(y_pred))) /2

        self.log("val/batch_loss", torch.tensor(losses).mean())
        self.log("val/acc", accuracy)
        self.log("val/iou", iou)
        self.log("val/miou", miou)
        return {"accuracy":accuracy, "iou":iou, "miou":miou}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.002)
        return optimizer

    def test_step(self, test_batch, batch_idx):
        self.test_transform(test_batch)

        x = test_batch["x128"]
        y = test_batch["y128"]

        x_out = self.model(x)
        y_pred = self.threshold_prediction(x_out, 0.5) # sigmoid
        for i in range(len(x_out)):
            y_pred[i] = torch.tensor(binary_fill_holes(binary_closing(y_pred[i].cpu())))

        loss = torch.nn.functional.binary_cross_entropy(x_out, y)
        self.log("test/loss", loss)

        if batch_idx == 0:
            self.log_prediction(x, y, x_out, test=True)
        return {"loss": loss, "x_out": x_out.cpu(), "y": y.cpu().to(bool)}

    def test_epoch_end(self, validation_step_outputs):
        losses = []
        y_pred = []
        y_true = []
        for step_outputs in validation_step_outputs:
            losses.append(step_outputs["loss"])
            y_pred.extend(step_outputs["x_out"].flatten())
            y_true.extend(step_outputs["y"].flatten())
        y_true = torch.tensor(y_true)
        y_pred = torch.tensor(y_pred)

        y_pred = self.threshold_prediction(y_pred, 0.5) # sigmoid
        if self.bbox_mode:
            pass

        accuracy = accuracy_score(y_true, y_pred)
        iou = jaccard_score(y_true, y_pred)
        miou = (iou + jaccard_score(torch.logical_not(y_true), torch.logical_not(y_pred))) /2

        self.log("test/batch_loss", torch.tensor(losses).mean())
        self.log("test/acc", accuracy)
        self.log("test/iou", iou)
        self.log("test/miou", miou)
        return {"accuracy":accuracy, "iou":iou, "miou":miou}


def get_datasets_splits_human(opt):
    opt.dataset_args["load_segmentations"] = False

    opt.dataset_args["split"] = DatasetSplit("train")
    opt.dataset_args["limit_dataset"] = [0, 5000]
    train_dataset = FOBAUnetDataset(opt, opt.H, opt.W, mask_dir=opt.out_dir)
    if not train_dataset.check_all_masks_computed():
        train_dataset.calculate_refined_masks(opt.exp_name)

    opt.dataset_args["split"] = DatasetSplit("val")
    opt.dataset_args["limit_dataset"] = [0, 100]
    val_dataset = FOBAUnetDataset(opt, opt.H, opt.W, mask_dir=opt.out_dir)
    if not val_dataset.check_all_masks_computed():
        val_dataset.calculate_refined_masks(opt.exp_name)

    return train_dataset, val_dataset, val_dataset#2nd one unused

def get_datasets_splits_bbox(opt):
    opt.dataset_args["load_segmentations"] = False
    opt.dataset_args["load_bboxes"] = False
    #opt.dataset_args["limit_dataset"] = [0, 100]

    opt.dataset_args["split"] = DatasetSplit("train")
    train_dataset = FOBAUnetDataset(opt, opt.H, opt.W, mask_dir=opt.out_dir)
    if not False: # train_dataset.check_all_masks_computed():
        train_dataset.calculate_refined_masks(opt.exp_name)

    opt.dataset_args["split"] = DatasetSplit("train")
    opt.dataset_args["limit_dataset"] = [0, 1] # ununsed as I train for fixed number of steps
    val_dataset = FOBAUnetDataset(opt, opt.H, opt.W, mask_dir=opt.out_dir)
    if not val_dataset.check_all_masks_computed():
        val_dataset.calculate_refined_masks(opt.exp_name)

    opt.dataset_args["split"] = DatasetSplit("test")
    opt.dataset_args["load_bboxes"] = True
    test_dataset = get_dataset(opt)
    return train_dataset, val_dataset, test_dataset#2nd one unused


def get_datasets_splits(opt):
    opt.dataset_args["split"] = DatasetSplit("train")
    #opt.dataset_args["limit_dataset"] =[0, 10]
    train_dataset = FOBAUnetDataset(opt, opt.H, opt.W, mask_dir=opt.out_dir)
    if not train_dataset.check_all_masks_computed():
        train_dataset.calculate_refined_masks(opt.exp_name)

    opt.dataset_args["split"] = DatasetSplit("val")
    val_dataset = FOBAUnetDataset(opt, opt.H, opt.W, mask_dir=opt.out_dir)
    if not val_dataset.check_all_masks_computed():
        val_dataset.calculate_refined_masks(opt.exp_name)

    opt.dataset_args["split"] = DatasetSplit("test")
    opt.dataset_args["load_segmentations"] = True
    test_dataset = get_dataset(opt)
    return train_dataset, val_dataset, test_dataset


def train_segmentation_network(opt):
    opt.exp_name = args.exp_name if args.exp_name is not None else opt.exp_name
    logger.info(f"Experiment name:{opt.exp_name}")

    model = UNet(
        n_channels=3,
        n_classes=1,
    )

    if opt.dataset == "human36":
        train_dataset, val_dataset, test_dataset = get_datasets_splits_human(opt)
        # assert test_dataset is val_dataset
    elif opt.dataset == "bird":
        train_dataset, val_dataset, test_dataset = get_datasets_splits(opt)
    elif opt.dataset == "dog":
        train_dataset, val_dataset, test_dataset = get_datasets_splits_bbox(opt)
    elif opt.dataset == "car":
        train_dataset, val_dataset, test_dataset = get_datasets_splits_bbox(opt)
    else:
        raise ValueError("No dataset selected in exp file")



    logger.info(f"Splits: Train {len(train_dataset)}, Val {len(val_dataset)}, Test {len(test_dataset)}")

    device = torch.device("cuda")
    model = model.to(device)

    train_dataloader = DataLoader(train_dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=0,#opt.num_workers,
                            collate_fn=collate_batch,
                            )
    val_dataloader = DataLoader(val_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=0,#opt.num_workers,
                            collate_fn=collate_batch,
                            )
    test_dataloader = DataLoader(test_dataset,
                            batch_size=32,
                            shuffle=False,
                            num_workers=0,#opt.num_workers,
                            collate_fn=collate_batch,
                            )

    if not args.postprocess:
        logger.info(f"No postprocessing")
        postprocess_function = lambda x: x
    else:
        logger.info(f"Postprocessing masks with filling and closing")
        mask_suggestor = GMMMaskSuggestor(opt)
        postprocess_function = mask_suggestor._compute_post_processed

    bbox_mode = args.bbox_mode
    logger.info(f"Running test in with BBox Mode set: {bbox_mode}")
    test_transform = get_test_preprocessing_function() if not bbox_mode else get_test_bbox_preprocessing_function()

    module = SegmentationUnet(model,
                        train_transform=get_train_preprocessing_function(postprocess_function),
                        val_transform=get_val_preprocessing_function(postprocess_function),
                        test_transform=test_transform,
                        bbox_mode=bbox_mode,
                    )


    steps_per_epoch =(len(train_dataset) // 32) + 1 - (len(train_dataset) % 32 == 0)
    n_epochs = int(12000 / steps_per_epoch)
    #n_epochs = 3

    logger.info(f"Training for: {n_epochs} epochs")
    wandb_logger = WandbLogger(project="foba-unet")
    checkpoint_callback = ModelCheckpoint(dirpath=os.path.join(opt.log_dir, "checkpoints"),
                                          save_last=True,
                                          save_top_k=2,
                                          monitor="val/acc",
                                          mode="max",
                                          )
    logger.info(f"Logging checkpoints to {os.path.join(opt.log_dir, 'checkpoints')}")

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="ddp",
        devices=torch.cuda.device_count(),
        max_epochs=n_epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
    )

    if args.test_only:
        assert args.ckpt_path is not None
        logger.info(f"Start testing with latest model: {args.ckpt_path}")
        test_trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=wandb_logger,
        )
        test_trainer.test(
            module,
            dataloaders=[test_dataloader],
            ckpt_path=args.ckpt_path,
        )
        exit(0)



    trainer.fit(module,
                train_dataloaders=train_dataloader,
                val_dataloaders=[val_dataloader],
                )

    if trainer.global_rank == 0:
        save_path = os.path.join(opt.log_dir, 'checkpoints', "last.ckpt")
        logger.info(f"Saving latest model to: {save_path}")
        trainer.save_checkpoint(filepath=save_path)

        logger.info(f"Start testing with latest model: {save_path}")
        test_trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            logger=wandb_logger,
        )
        test_trainer.test(
            module,
            dataloaders=[test_dataloader],
            ckpt_path=save_path,
        )


if __name__ == "__main__":
    args = get_train_segmentation_refined()
    exp_config = make_exp_config(args.EXP_PATH)

    # make log dir
    if not os.path.exists(exp_config.log_dir): os.mkdir(exp_config.log_dir)
    log_dir = os.path.join(exp_config.log_dir, exp_config.name, datetime.now().strftime("%Y-%m-%dT%H-%M-%S"))
    os.makedirs(log_dir)
    os.makedirs(os.path.join(log_dir, "checkpoints"))
    shutil.copy(exp_config.__file__, log_dir)
    exp_config.log_dir = log_dir

    wandb.init(project="foba-unet", entity="hr-biomedia")
    exp_config.args = args
    train_segmentation_network(exp_config)