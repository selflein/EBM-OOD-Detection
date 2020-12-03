import sys, os
sys.path.insert(0, os.getcwd())

from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from sacred import Experiment

from uncertainty_est.archs.arch_factory import get_arch
from uncertainty_est.data.dataloaders import get_dataloaders


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
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))


class CEBaseline(pl.LightningModule):
    def __init__(self, backbone, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        acc = (y == y_hat.argmax(1)).mean(0).item()
        self.log('val_acc', loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('test_loss', loss)
        acc = (y == y_hat.argmax(1)).mean(0).item()
        self.log('test_acc', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


@ex.automain
def run(trainer_config, model_config, lr, dataset, seed, batch_size, monitor=None, config_file=None):
    pl.seed_everything(seed)

    arch = get_arch(model_config)
    model = CEBaseline(arch, lr)

    train_loader, val_loader, test_loader = get_dataloaders("../data", dataset, batch_size=batch_size)

    ckpt_callback = pl.callbacks.ModelCheckpoint(monitor=monitor)
    early_stopping_callback = pl.callbacks.EarlyStopping()

    trainer = pl.Trainer(**trainer_config, logger=True)
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    return results

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()

    ex.add_config(args.config)

    ex.run_commandline()
