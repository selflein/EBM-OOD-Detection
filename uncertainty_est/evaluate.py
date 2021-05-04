import os
import sys

sys.path.insert(0, os.getcwd())

import logging
from pathlib import Path
from argparse import ArgumentParser

import pandas as pd

from uncertainty_est.data.dataloaders import get_dataloader
from uncertainty_est.models import load_checkpoint, load_model


parser = ArgumentParser()
parser.add_argument("--checkpoint", type=str, action="append", default=[])
parser.add_argument("--dataset", type=str)
parser.add_argument("--ood_dataset", type=str, action="append")
parser.add_argument("--eval-classification", action="store_true")
parser.add_argument("--output-folder", type=str)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--max-eval", type=int, default=-1)
parser.add_argument("--checkpoint-dir", type=str)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def eval_model(
    model,
    dataset,
    ood_datasets,
    eval_classification=False,
    model_name="",
    batch_size=128,
    max_items=-1,
):
    id_test_loader = get_dataloader(dataset, "test", batch_size=batch_size)

    clf_accum = []
    if eval_classification:
        clf_results = model.eval_classifier(id_test_loader, max_items)
        for k, v in clf_results.items():
            logger.info(f"{k}: {v:.02f}")
            clf_accum.append((model_name, model.__class__.__name__, dataset, k, v))

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
        logger.info(f"{k}: {v:.02f}")
        accum.append((model_name, model.__class__.__name__, dataset, *k, v))

    return accum, clf_accum


if __name__ == "__main__":
    args = parser.parse_args()

    ood_tbl_rows = []
    clf_tbl_rows = []
    for checkpoint in args.checkpoint:
        checkpoint_path = Path(checkpoint)
        model_name = checkpoint_path.parent.stem

        model, config = load_checkpoint(checkpoint_path, strict=False)
        model.eval()
        model.cuda()

        if not args.ood_dataset:
            args.ood_dataset = config["test_ood_datasets"]

        if not args.dataset:
            args.dataset = config["dataset"]

        ood_rows, clf_rows = eval_model(
            model,
            args.dataset,
            args.ood_dataset,
            args.eval_classification,
            model_name=model_name,
            batch_size=128,
            max_items=args.max_eval,
        )
        ood_tbl_rows.extend(ood_rows)
        clf_tbl_rows.append(clf_rows)

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        for model_dir in checkpoint_dir.glob("**/version_*"):
            try:
                model, config = load_model(model_dir, last=True, strict=False)
            except:
                continue
            model.eval()
            model.cuda()

            if not args.ood_dataset:
                args.ood_dataset = config["test_ood_datasets"]

            if not args.dataset:
                args.dataset = config["dataset"]

            ood_rows, clf_rows = eval_model(
                model,
                args.dataset,
                args.ood_dataset,
                args.eval_classification,
                model_name=model_dir.parent.stem,
                batch_size=128,
                max_items=args.max_eval,
            )
            ood_tbl_rows.extend(ood_rows)
            clf_tbl_rows.extend(clf_rows)

    if args.output_folder:
        output_folder = Path(args.output_folder)
        ood_df = pd.DataFrame(
            ood_tbl_rows,
            columns=(
                "Model",
                "Model Type",
                "ID dataset",
                "OOD dataset",
                "Score",
                "Metric",
                "Value",
            ),
        )
        ood_df.to_csv(output_folder / f"ood-{args.name}.csv", index=False)

        if args.eval_classification:
            clf_df = pd.DataFrame(
                clf_tbl_rows,
                columns=(
                    "Model",
                    "Model Type",
                    "ID dataset",
                    "Metric",
                    "Value",
                ),
            )
            clf_df.to_csv(output_folder / f"clf-{args.name}.csv", index=False)
