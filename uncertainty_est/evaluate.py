import os
import sys

sys.path.insert(0, os.getcwd())

import logging
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

from uncertainty_est.data.dataloaders import get_dataloader
from uncertainty_est.models import load_checkpoint


parser = ArgumentParser()
parser.add_argument("--checkpoint", type=str, action="append")
parser.add_argument("--dataset", type=str)
parser.add_argument("--ood_dataset", type=str, action="append")
parser.add_argument("--eval-classification", action="store_true")
parser.add_argument("--output_csv", type=str)
parser.add_argument("--max-eval", type=int, default=-1)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def eval_model(
    model,
    dataset,
    ood_datasets,
    eval_classification=False,
    output_folder=None,
    model_name="",
    batch_size=128,
    max_items=-1,
):

    # Reset logger handlers
    logger.handlers = []
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)

    if output_folder is not None:
        logger.addHandler(logging.FileHandler(output_folder / "out.log", mode="w"))

    id_test_loader = get_dataloader(dataset, "test", batch_size=batch_size)

    if eval_classification:
        clf_results = model.eval_classifier(id_test_loader, max_items)
        for k, v in clf_results.items():
            logger.info(f"{k}: {v:.02f}")

    test_ood_dataloaders = []
    for test_ood_dataset in ood_datasets:
        loader = get_dataloader(
            test_ood_dataset,
            "test",
            data_shape=id_test_loader.dataset.data_shape,
            batch_size=batch_size,
        )
        test_ood_dataloaders.append((test_ood_dataset, loader))

    ood_results = model.eval_ood(id_test_loader, test_ood_dataloaders)

    accum = []
    for k, v in ood_results.items():
        accum.append((model_name, model.__class__.__name__, dataset, k, v))

    return accum


if __name__ == "__main__":
    args = parser.parse_args()

    rows = []
    for checkpoint in args.checkpoint:
        checkpoint_path = Path(checkpoint)
        output_folder = checkpoint_path.parent
        model_name = output_folder.stem

        model, config = load_checkpoint(checkpoint_path, strict=False)
        model.eval()
        model.cuda()

        rows.extend(
            eval_model(
                model,
                args.dataset,
                args.ood_dataset,
                args.eval_classification,
                output_folder=output_folder,
                model_name=model_name,
                batch_size=128,
                max_items=args.max_eval,
            )
        )

    if args.output_csv:
        ood_df = pd.DataFrame(
            rows,
            columns=(
                "model",
                "model_type",
                "id_dataset",
                "score",
                "metric",
            ),
        )
        ood_df.to_csv(args.output_csv, index=False)
