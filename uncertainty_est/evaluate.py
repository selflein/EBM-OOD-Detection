import os
import sys

sys.path.insert(0, os.getcwd())

import logging
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from uncertainty_eval.metrics.brier import brier_score, brier_decomposition
from uncertainty_eval.metrics.calibration_error import classification_calibration
from uncertainty_eval.vis import draw_reliability_graph, plot_score_hist

from uncertainty_est.utils.utils import to_np
from uncertainty_est.data.dataloaders import get_dataloader
from uncertainty_est.utils.metrics import accuracy
from uncertainty_est.models import load_checkpoint


parser = ArgumentParser()
parser.add_argument("--checkpoint", type=str, action="append")
parser.add_argument("--dataset", type=str)
parser.add_argument("--ood_dataset", type=str, action="append")
parser.add_argument("--eval-classification", action="store_true")
parser.add_argument("--output_csv", type=str)


logger = logging.getLogger()
logger.setLevel(logging.INFO)


def eval_model(
    model,
    dataset,
    ood_datasets,
    eval_classification=False,
    output_folder=None,
    model_name="",
):
    # Reset logger handlers
    logger.handlers = []
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(stdout_handler)

    if output_folder is not None:
        logger.addHandler(logging.FileHandler(output_folder / "out.log", mode="w"))

    id_test_loader = get_dataloader(dataset, "test", batch_size=128)

    if eval_classification:
        y, logits = model.get_gt_preds(id_test_loader)

        # Compute accuracy
        probs = torch.softmax(logits, dim=1)
        acc = accuracy(y, probs)
        logger.info(f"Accuracy: {acc * 100.:.02f}")

        # Compute calibration
        y_np, probs_np = to_np(y), to_np(probs)
        ece, mce = classification_calibration(y_np, probs_np)
        brier = brier_score(y_np, probs_np)
        uncertainty, resolution, reliability = brier_decomposition(y_np, probs_np)

        fig, ax = plt.subplots(figsize=(10, 10))
        draw_reliability_graph(y_np, probs_np, 10, ax=ax)
        fig.savefig(output_folder / "calibration.png", dpi=200)

        logger.info(f"ECE: {ece * 100:.02f}")
        logger.info(f"Brier: {brier * 100:.02f}")
        logger.info(f"Brier uncertainty: {uncertainty * 100:.02f}")
        logger.info(f"Brier resolution: {resolution * 100:.02f}")
        logger.info(f"Brier reliability: {reliability * 100:.02f}")
        logger.info(
            f"Brier (via decomposition): {(reliability - resolution + uncertainty) * 100:.02f}"
        )

    # Compute ID OOD scores
    id_scores_dict = model.ood_detect(id_test_loader)

    accum = []
    # Compute OOD detection metrics
    for ood_ds in ood_datasets:
        logger.info(f"\n\n{ood_ds}")
        ood_loader = get_dataloader(
            ood_ds,
            "test",
            data_shape=id_test_loader.dataset.data_shape,
            batch_size=128,
            num_workers=4,
        )
        ood_scores_dict = model.ood_detect(ood_loader)

        for score_name, id_scores in id_scores_dict.items():
            ood_scores = ood_scores_dict[score_name]

            labels = np.concatenate(
                [np.zeros_like(ood_scores), np.ones_like(id_scores)]
            )
            preds = np.concatenate([ood_scores, id_scores])

            if output_folder is not None:
                try:
                    ax = plot_score_hist(
                        id_scores,
                        ood_scores,
                        title="",
                    )
                    ax.figure.savefig(output_folder / f"{ood_ds}_{score_name}.png")
                    plt.close()
                except:
                    pass

            auroc = roc_auc_score(labels, preds) * 100.0
            aupr = average_precision_score(labels, preds) * 100.0

            logger.info(score_name)
            logger.info(f"AUROC: {auroc:.02f}")
            logger.info(f"AUPR: {aupr:.02f}")

            accum.append(
                (
                    model_name,
                    type(model),
                    dataset,
                    ood_ds,
                    score_name,
                    auroc,
                    aupr,
                )
            )

    return accum


if __name__ == "__main__":
    args = parser.parse_args()

    rows = []
    for checkpoint in args.checkpoint:
        checkpoint_path = Path(checkpoint)
        output_folder = checkpoint_path.parent
        model_name = output_folder.stem

        model, config = load_checkpoint(checkpoint_path)
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
            )
        )

    if args.output_csv:
        ood_df = pd.DataFrame(
            rows,
            columns=(
                "model",
                "model_type",
                "id_dataset",
                "ood_dataset",
                "score",
                "AUROC",
                "AUPR",
            ),
        )
        ood_df.to_csv(args.output_csv, index=False)
