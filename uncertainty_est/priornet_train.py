import os
import sys

sys.path.insert(0, os.getcwd())

from pathlib import Path
from datetime import datetime

import torch
import seml
from sacred import Experiment
import pytorch_lightning as pl

from uncertainty_est.data.dataloaders import get_dataloader
from uncertainty_est.models.priornet.priornet import PriorNet


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
    monitor,
    ood_dataset,
    gamma,
    target_concentration,
    concentration,
    reverse_kl,
    alpha_fix,
):
    pl.seed_everything(seed)

    out_path = Path("logs") / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_path.mkdir(exist_ok=False, parents=True)

    train_loader = get_dataloader(
        dataset, "train", batch_size, img_size=32, ood_dataset=ood_dataset
    )
    val_loader = get_dataloader(dataset, "val", batch_size, img_size=32)
    test_loader = get_dataloader(dataset, "test", batch_size, img_size=32)

    model = PriorNet(
        arch_name,
        dict(arch_config),
        float(lr),
        momentum,
        weight_decay,
        reverse_kl=reverse_kl,
        concentration=concentration,
        target_concentration=target_concentration,
        gamma=gamma,
        alpha_fix=alpha_fix,
    )

    ckpt_callback = pl.callbacks.ModelCheckpoint(out_path)
    es_callback = pl.callbacks.EarlyStopping(monitor=monitor, patience=10, mode="max")
    logger = pl.loggers.TensorBoardLogger(out_path)

    trainer = pl.Trainer(
        **trainer_config, logger=logger, callbacks=[ckpt_callback, es_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    return result


if __name__ == "__main__":
    ex.run_commandline()