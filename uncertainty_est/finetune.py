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
from uncertainty_est.models.energy_finetuning import EnergyFinetune


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
    m_in,
    m_out,
    score,
    checkpoint,
):
    pl.seed_everything(seed)

    out_path = Path("logs") / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    out_path.mkdir(exist_ok=False, parents=True)

    train_loader = get_dataloader(
        dataset, "train", batch_size, img_size=32, ood_dataset=ood_dataset
    )
    val_loader = get_dataloader(dataset, "val", batch_size, img_size=32)
    test_loader = get_dataloader(dataset, "test", batch_size, img_size=32)

    model = EnergyFinetune(
        arch_name,
        dict(arch_config),
        float(lr),
        momentum,
        weight_decay,
        score,
        m_in,
        m_out,
        trainer_config["min_epochs"],
        len(train_loader),
    )
    ckpt = torch.load(checkpoint)
    model.load_state_dict(ckpt["state_dict"])

    ckpt_callback = pl.callbacks.ModelCheckpoint(out_path)
    logger = pl.loggers.TensorBoardLogger(out_path)
    trainer = pl.Trainer(**trainer_config, logger=logger, callbacks=[ckpt_callback])
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    return result


if __name__ == "__main__":
    ex.run_commandline()
