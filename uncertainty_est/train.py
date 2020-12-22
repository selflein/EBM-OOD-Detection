import os
import sys

sys.path.insert(0, os.getcwd())

from pathlib import Path
from datetime import datetime

import seml
from sacred import Experiment
import pytorch_lightning as pl

from uncertainty_est.models import MODELS
from uncertainty_est.data.dataloaders import get_dataloader


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
    model_name,
    model_config,
    dataset,
    seed,
    batch_size,
    ood_dataset,
    earlystop_config,
    checkpoint_config,
    sigma=0.0,
):
    pl.seed_everything(seed)

    model = MODELS[model_name](**model_config)

    train_loader = get_dataloader(
        dataset, "train", batch_size, img_size=32, ood_dataset=ood_dataset, sigma=sigma
    )
    val_loader = get_dataloader(dataset, "val", batch_size, img_size=32, sigma=sigma)
    test_loader = get_dataloader(dataset, "test", batch_size, img_size=32, sigma=sigma)

    out_path = (
        Path("logs")
        / model_name
        / dataset
        / datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    )
    out_path.mkdir(exist_ok=False, parents=True)

    ckpt_callback = pl.callbacks.ModelCheckpoint(out_path, **checkpoint_config)
    es_callback = pl.callbacks.EarlyStopping(**earlystop_config)
    logger = pl.loggers.TensorBoardLogger(out_path)

    trainer = pl.Trainer(
        **trainer_config, logger=logger, callbacks=[ckpt_callback, es_callback]
    )
    trainer.fit(model, train_loader, val_loader)

    result = trainer.test(test_dataloaders=test_loader)
    return result


if __name__ == "__main__":
    ex.run_commandline()
