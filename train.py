import os
import fire
import time
import torch
import contextlib
import numpy as np
import torch.nn as nn
import torch.utils.data as data

from dataclasses import asdict
from collections import OrderedDict
from torch.nn import functional as F
from torchvision.transforms import v2
from config import get_config
from factory import ModelFactory, DatasetFactory

SEED = 20240605
CKPT_DIR = "out"
LABEL_SMOOTH_RATIO = 0.5


class Trainer:
    def __init__(self):
        self.config = None
        self.device_type = "cuda" if torch.cuda.is_available() else "mps"
        self.dtype = torch.bfloat16
        if self.device_type == "cuda":
            self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            self.ctx = torch.amp.autocast(
                device_type=self.device_type, dtype=self.dtype
            )
        else:
            self.scaler = torch.amp.GradScaler(enabled=False)
            self.ctx = contextlib.suppress()
        self.train_loader = self.val_loader = None
        self.loss_fn = nn.BCEWithLogitsLoss()

    def criterion(self, outputs, targets):
        one_hot = torch.zeros(
            (targets.shape[0], self.config.num_classes),
            dtype=self.dtype,
            device=self.device_type,
        )
        one_hot.fill_((1 - LABEL_SMOOTH_RATIO) / self.config.num_classes)
        one_hot = one_hot + (targets * LABEL_SMOOTH_RATIO)
        return torch.sum(-one_hot * F.log_softmax(outputs, -1), -1).mean()

    def train_step(self, model, optimizer, batch, step_index):
        images, labels = batch
        if len(images.size()) <= 3:
            images = images.unsqueeze(1)
        images = images.to(self.device_type).to(self.dtype)
        labels = F.one_hot(labels.to(torch.long), self.config.num_classes)
        labels = labels.to(self.device_type)
        images, labels = self.cm(images, labels)

        with self.ctx:
            out = model(images)
            loss = self.criterion(out, labels)
            loss = loss / self.config.accumulation_steps

        self.scaler.scale(loss).backward()

        if (step_index + 1) % self.config.accumulation_steps == 0:
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            optimizer.zero_grad(set_to_none=True)

        return out, labels, loss * self.config.accumulation_steps

    def get_accuracy(self, out, target):
        _, predict = torch.max(out, dim=-1)
        correct = predict == torch.max(target, dim=-1)[1]
        accuracy = correct.sum().item() / correct.size(0)
        return accuracy

    def validate(self, cmodel):
        cmodel.eval()

        accumu_out = []
        accumu_labels = []
        accumu_loss = 0
        for data_entry in self.val_loader:
            images, labels = data_entry
            if len(images.size()) <= 3:
                images = images.unsqueeze(1)
            images = images.to(self.device_type).to(self.dtype)
            labels = labels.to(self.device_type).to(torch.long)
            labels = F.one_hot(labels, self.config.num_classes)
            # forward
            with self.ctx:
                out = cmodel(images)
                loss = self.criterion(out, labels)
                accumu_out.append(out.cpu().detach())
                accumu_labels.append(labels.cpu().detach())
                accumu_loss += loss.detach().item()

        # accuracy
        out = torch.cat(accumu_out)
        labels = torch.cat(accumu_labels)
        accuracy = self.get_accuracy(out, labels)
        res = OrderedDict(
            [
                ("loss", accumu_loss / len(self.val_loader)),
                ("accuracy", accuracy),
            ]
        )

        cmodel.train()
        return res

    def load_dataset(self, dataset_name, config):
        train_ds, val_ds = DatasetFactory.create_dataset(dataset_name, config)

        self.train_loader = data.DataLoader(
            train_ds,
            self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
        )

        self.val_loader = data.DataLoader(
            val_ds,
            self.config.batch_size // 4,
            num_workers=self.config.num_workers,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=2,
        )

    def init_model(self, model_name: str, config, resume: str):
        model = ModelFactory.create_model(model_name, config)
        model = model.to(self.device_type)
        print("model:", model)

        if resume:
            checkpoint = torch.load(resume, map_location=self.device_type)
            state_dict = checkpoint["model"]
            self.config.lr = 1e-3
            model.load_state_dict(state_dict)
            print("Resume training...")

        return model

    def train(
        self, model_name="resnet50", dataset_name="mnist", resume="", learning_rate=None
    ):
        self.config = get_config(dataset_name)
        self.cutmix = v2.CutMix(num_classes=self.config.num_classes)
        self.mixup = v2.MixUp(num_classes=self.config.num_classes)
        self.cm = v2.RandomChoice([self.cutmix, self.mixup])

        self.load_dataset(dataset_name, self.config)
        model = self.init_model(model_name, self.config, resume)
        if learning_rate:
            self.config.lr = learning_rate
        cmodel = torch.compile(model) if torch.cuda.is_available() else model
        optimizer = torch.optim.AdamW(cmodel.parameters(), self.config.lr)

        best_metric = 0.0
        begin = time.time()

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            "max",
            factor=0.5,
            patience=3,
            threshold=5e-3,
            threshold_mode="abs",
        )

        log_iters = len(self.train_loader) // 5
        iteration = 0

        for epoch in range(self.config.epochs):
            for step_index, batch in enumerate(self.train_loader):
                out, labels, loss = self.train_step(
                    cmodel, optimizer, batch, step_index
                )

                if iteration % log_iters == 0 and iteration > 0:
                    metrics = OrderedDict(
                        [
                            ("loss", loss.item()),
                        ],
                    )
                    metrics["accuracy"] = self.get_accuracy(out, labels)

                    now = time.time()
                    duration = now - begin
                    begin = now
                    messages = [f"[{epoch:03d}: {iteration:06d}]"]
                    for name, val in metrics.items():
                        messages.append(f"{name}: {val:.3f}")
                    lr = optimizer.param_groups[0]["lr"]
                    messages.append(f"lr: {lr:.3e}")
                    messages.append(f"time: {duration:.1f}")
                    print(" ".join(messages), flush=True)
                iteration += 1

            accumulator = self.validate(cmodel)
            val_metric = accumulator["accuracy"]
            best_metric = max(best_metric, val_metric)
            checkpoint = {
                "model": model.state_dict(),
                "epoch": epoch,
                "train_config": asdict(self.config),
                "eval_metric": val_metric,
            }
            torch.save(
                checkpoint,
                os.path.join(
                    CKPT_DIR,
                    f"{iteration}.pt",
                ),
            )
            messages = ["[Val]"]
            for name, val in accumulator.items():
                messages.append(f"{name}: {val:.3f}")
            print(" ".join(messages), flush=True)
            scheduler.step(val_metric)

            current_lr = optimizer.param_groups[0]["lr"]
            if current_lr < 1e-6:
                break

        print("Best validating metric:", best_metric)
        return best_metric


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.set_float32_matmul_precision("high")
    np.random.seed(SEED)

    trainer = Trainer()
    fire.Fire(trainer.train)
