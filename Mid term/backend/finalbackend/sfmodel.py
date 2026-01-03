
import os
import argparse
from typing import List

import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, Resize, CenterCrop

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import torchmetrics

import numpy as np
import matplotlib.pyplot as plt

from pytorchvideo.models.hub import slowfast_r50


class UCFCrimeDataset(Dataset):
    
    def __init__(self, root_dir: str, class_names: List[str], transform=None, frames: int = 32):
        self.root_dir = root_dir
        self.class_names = class_names
        self.transform = transform
        self.frames = frames
        self.samples = []

        for label, cls in enumerate(class_names):
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for vid in os.listdir(cls_dir):
                if vid.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    self.samples.append((os.path.join(cls_dir, vid), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = self._load_video_cv2(path, self.frames)  # (C, T, H, W)

        if self.transform:
            clip = self.transform(clip)

        return clip, int(label)

    @staticmethod
    def _load_video_cv2(path: str, frames: int):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise RuntimeError(f"Video {path} has zero frames")

        # Frame indices to sample evenly
        if total_frames < frames:
            indices = list(range(total_frames)) + [total_frames - 1] * (frames - total_frames)
        else:
            indices = np.linspace(0, total_frames - 1, frames).astype(int).tolist()

        frames_list = []
        frame_pos = 0
        ret = True
        sampled_idx = 0

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i == indices[sampled_idx]:
                # Convert BGR to RGB, resize later in transform
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_list.append(frame_rgb)
                sampled_idx += 1
                if sampled_idx >= len(indices):
                    break
        cap.release()

        # If frames less than requested (rare), pad last frame
        while len(frames_list) < frames:
            frames_list.append(frames_list[-1])

        # Convert to numpy array (T, H, W, C)
        clip_np = np.stack(frames_list, axis=0)

        # Convert to tensor (C, T, H, W)
        clip_tensor = torch.from_numpy(clip_np).permute(3, 0, 1, 2).float() / 255.0

        return clip_tensor


class VideoTransform(nn.Module):

    def __init__(self, size=224):
        super().__init__()
        self.frame_transform = Compose([
            Resize((256, 256)),
            CenterCrop(size),
        ])

    def forward(self, x: torch.Tensor):
        # x shape: (C, T, H, W)
        C, T, H, W = x.shape
        x = x.permute(1, 0, 2, 3)  # (T, C, H, W)
        frames = []
        for i in range(T):
            frame = x[i]
            frame = self.frame_transform(frame)
            frames.append(frame)
        x = torch.stack(frames, dim=1)  # (C, T, H, W)
        return x


# Lightning Module
class SlowFastLitModel(pl.LightningModule):
    def __init__(self, num_classes: int, lr: float = 1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = slowfast_r50(pretrained=True)

        in_features = self.model.blocks[-1].proj.in_features
        self.model.blocks[-1].proj = nn.Linear(in_features, num_classes)

        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=num_classes)
        self.val_precision = torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro")
        self.val_recall = torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro")
        self.val_f1 = torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro")

    def forward(self, x):
        return self.model(x)

    def _prepare_pathways(self, x: torch.Tensor):
        slow = x[:, :, ::4, :, :]
        fast = x
        return [slow, fast]

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_in = self._prepare_pathways(x)
        logits = self.forward(x_in)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.train_acc(preds, y)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x_in = self._prepare_pathways(x)
        logits = self.forward(x_in)
        loss = self.loss_fn(logits, y)

        preds = torch.argmax(logits, dim=1)
        acc = self.val_acc(preds, y)
        precision = self.val_precision(preds, y)
        recall = self.val_recall(preds, y)
        f1 = self.val_f1(preds, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_precision", precision, on_step=False, on_epoch=True)
        self.log("val_recall", recall, on_step=False, on_epoch=True)
        self.log("val_f1", f1, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}


# Data Module
class UCFDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, class_names: List[str], batch_size: int = 4, num_workers: int = 4, frames: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.class_names = class_names
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frames = frames

    def setup(self, stage=None):
        transform = VideoTransform()
        dataset = UCFCrimeDataset(self.data_dir, self.class_names, transform=transform, frames=self.frames)
        n = len(dataset)
        train_n = int(0.8 * n)
        val_n = n - train_n
        self.train_ds, self.val_ds = random_split(
            dataset, [train_n, val_n], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=torch.cuda.is_available(), prefetch_factor=2)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=torch.cuda.is_available(), prefetch_factor=2)


# Callback for metrics plotting
class MetricsPlotter(pl.Callback):
    def __init__(self, out_dir="logs/plots"):
        super().__init__()
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
        }

    def on_validation_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        def _get(k):
            return float(metrics[k]) if k in metrics else None

        self.history["val_loss"].append(_get("val_loss"))
        self.history["val_acc"].append(_get("val_acc"))
        self.history["val_precision"].append(_get("val_precision"))
        self.history["val_recall"].append(_get("val_recall"))
        self.history["val_f1"].append(_get("val_f1"))
        self.history["train_loss"].append(_get("train_loss"))
        self.history["train_acc"].append(_get("train_acc"))

    def on_train_end(self, trainer, pl_module):
        epochs = range(1, len(self.history["val_loss"]) + 1)

        plt.figure()
        if any(x is not None for x in self.history["train_loss"]):
            plt.plot(epochs, self.history["train_loss"], label="train_loss")
        if any(x is not None for x in self.history["val_loss"]):
            plt.plot(epochs, self.history["val_loss"], label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.title("Loss per epoch")
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "loss.png"))
        plt.close()

        plt.figure()
        if any(x is not None for x in self.history["train_acc"]):
            plt.plot(epochs, self.history["train_acc"], label="train_acc")
        if any(x is not None for x in self.history["val_acc"]):
            plt.plot(epochs, self.history["val_acc"], label="val_acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.legend()
        plt.title("Accuracy per epoch")
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "accuracy.png"))
        plt.close()

        plt.figure()
        if any(x is not None for x in self.history["val_precision"]):
            plt.plot(epochs, self.history["val_precision"], label="precision")
        if any(x is not None for x in self.history["val_recall"]):
            plt.plot(epochs, self.history["val_recall"], label="recall")
        if any(x is not None for x in self.history["val_f1"]):
            plt.plot(epochs, self.history["val_f1"], label="f1")
        plt.xlabel("epoch")
        plt.ylabel("score")
        plt.legend()
        plt.title("Precision / Recall / F1")
        plt.grid(True)
        plt.savefig(os.path.join(self.out_dir, "prf1.png"))
        plt.close()

        print(f"Saved plots to {self.out_dir}")




import sys
import argparse

if __name__ == "__main__":
    sys.argv = [
        "notebook_script",
        "--data_dir", "/kaggle/input/ucf-crime-full",
        "--batch_size", "2",
        "--epochs", "20",
        "--num_workers", "4",
        "--lr", "0.0001",
        "--frames", "32",
        "--log_dir", "logs",
        "--experiment_name", "slowfast_experiment"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    # ... rest of the code


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--experiment_name", type=str, default="slowfast_experiment")
    args = parser.parse_args()

    CLASS_NAMES = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    NUM_CLASSES = len(CLASS_NAMES)

    print(f"Detected classes: {CLASS_NAMES}")
    print(f"Number of classes: {NUM_CLASSES}")

    dm = UCFDataModule(args.data_dir, CLASS_NAMES, batch_size=args.batch_size, num_workers=args.num_workers, frames=args.frames)
    model = SlowFastLitModel(num_classes=NUM_CLASSES, lr=args.lr)

    logger = CSVLogger(save_dir=args.log_dir, name=args.experiment_name)

    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1,
                                          filename="best-checkpoint-{epoch:02d}-{val_acc:.4f}")

    metrics_plotter = MetricsPlotter(out_dir=os.path.join(args.log_dir, args.experiment_name, "plots"))

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        precision="16-mixed" if torch.cuda.is_available() else 32,
        log_every_n_steps=10,
        logger=logger,
        callbacks=[checkpoint_callback, metrics_plotter],
    )

    trainer.fit(model, datamodule=dm)