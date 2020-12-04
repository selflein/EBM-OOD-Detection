import os
import sys


from argparse import ArgumentParser

import pytorch_lightning as pl
import seml
import torch
from sacred import Experiment
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets.mnist import MNIST

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.data.dataloaders import get_dataloaders

sys.path.insert(0, os.getcwd())
ex = Experiment()
seml.setup_logger(ex)


@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)


@ex.config
def config():
    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(
            seml.create_mongodb_observer(db_collection, overwrite=overwrite)
        )


class CEBaseline(pl.LightningModule):
    def __init__(self, backbone, learning_rate, momentum, weight_decay):
        super().__init__()
        self.backbone = backbone
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )


@ex.automain
def run(
    trainer_config,
    arch_name,
    arch_config,
    lr,
    momentum,
    weight_decay,
    dataset,
    seed,
    batch_size,
    monitor=None,
):
    pl.seed_everything(seed)

    arch = get_arch(arch_name, arch_config)
    model = CEBaseline(arch, float(lr), momentum, weight_decay)

    train_loader, val_loader, test_loader = get_dataloaders(
        "../data", dataset, batch_size=batch_size
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor=monitor)
    early_stopping_callback = pl.callbacks.EarlyStopping(monitor=monitor, patience=10)

    trainer = pl.Trainer(
        **trainer_config,
        logger=True,
        callbacks=[ckpt_callback, early_stopping_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    return result


if __name__ == "__main__":
    ex.run_commandline()
