# Fix https://github.com/pytorch/pytorch/issues/37377
import numpy as _

import os
import sys
from uuid import uuid4

sys.path.insert(0, os.getcwd())

from pathlib import Path
from datetime import datetime

import yaml
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


@ex.main
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
    data_shape,
    _run,
    num_classes=1,
    sigma=0.0,
    output_folder=None,
    log_dir=None,
    num_workers=4,
    test_ood_datasets=[],
    mutation_rate=0.0,
    num_cat=1,
    **kwargs,
):
    pl.seed_everything(seed)
    assert num_classes > 0

    model = MODELS[model_name](**model_config)

    train_loader = get_dataloader(
        dataset,
        "train",
        batch_size,
        data_shape=data_shape,
        ood_dataset=ood_dataset,
        sigma=sigma,
        num_workers=num_workers,
        mutation_rate=mutation_rate,
    )
    val_loader = get_dataloader(
        dataset,
        "val",
        batch_size,
        data_shape=data_shape,
        sigma=sigma,
        ood_dataset=ood_dataset,
        num_workers=num_workers,
    )
    test_loader = get_dataloader(
        dataset,
        "test",
        batch_size,
        data_shape=data_shape,
        sigma=sigma,
        ood_dataset=None,
        num_workers=num_workers,
    )

    if log_dir == None:
        out_path = Path("logs") / model_name / dataset
    else:
        out_path = Path(log_dir)

    output_folder = (
        f'{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}_{uuid4()}'
        if output_folder is None
        else output_folder
    )

    # Circumvent issue when starting multiple versions with the same name
    trys = 10
    for i in range(trys):
        try:
            logger = pl.loggers.TensorBoardLogger(
                out_path, name=output_folder, default_hp_metric=False
            )
            out_dir = Path(logger.log_dir)
            out_dir.mkdir(exist_ok=False, parents=True)
            break
        except FileExistsError as e:
            if i == (trys - 1):
                raise ValueError("Could not create log folder") from e
            print("Failed to create unique log folder. Trying again.")

    with (out_dir / "config.yaml").open("w") as f:
        f.write(yaml.dump(_run.config))

    callbacks = []
    callbacks.append(pl.callbacks.ModelCheckpoint(dirpath=out_dir, **checkpoint_config))

    if earlystop_config is not None:
        es_callback = pl.callbacks.EarlyStopping(**earlystop_config)
        callbacks.append(es_callback)

    trainer = pl.Trainer(
        **trainer_config,
        logger=logger,
        callbacks=callbacks,
        progress_bar_refresh_rate=100,
    )
    trainer.fit(model, train_loader, val_loader)

    try:
        _ = trainer.test(test_dataloaders=test_loader)
    except:
        _ = trainer.test(test_dataloaders=test_loader, ckpt_path=None)

    test_ood_dataloaders = []
    for test_ood_dataset in test_ood_datasets:
        loader = get_dataloader(
            test_ood_dataset,
            "test",
            batch_size,
            data_shape=data_shape,
            sigma=sigma,
            ood_dataset=None,
            num_workers=num_workers,
        )
        test_ood_dataloaders.append((test_ood_dataset, loader))
    ood_results = model.eval_ood(test_loader, test_ood_dataloaders)
    ood_results = {", ".join(k): v for k, v in ood_results.items()}

    clf_results = model.eval_classifier(test_loader)

    results = {**ood_results, **clf_results}

    logger.log_hyperparams(model.hparams, results)
    return results


if __name__ == "__main__":
    ex.run_commandline()
