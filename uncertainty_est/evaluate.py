import os
import sys

sys.path.insert(0, os.getcwd())

import logging
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from uncertainty_eval.metrics.brier import brier_score, brier_decomposition
from uncertainty_eval.metrics.calibration_error import classification_calibration
from uncertainty_eval.vis import draw_reliability_graph

from uncertainty_est.utils.utils import to_np
from uncertainty_est.utils.dirichlet import dirichlet_prior_network_uncertainty
from uncertainty_est.data.dataloaders import get_dataloader
from uncertainty_est.utils.metrics import accuracy
from uncertainty_est.models import MODELS


parser = ArgumentParser()
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--ood_dataset", type=str, action="append")
parser.add_argument("--model", type=str)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)


def plot_score_hist(real_scores, fake_scores, title=None, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    if title is not None:
        ax.set_title(title)

    ax.hist(real_scores, bins=100, alpha=0.5, density=True, stacked=True)
    ax.hist(fake_scores, bins=100, alpha=0.5, density=True, stacked=True)
    ax.legend(labels=("Real", "Fake"))
    return ax


if __name__ == "__main__":
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    output_folder = checkpoint_path.parent

    model = MODELS[args.model].load_from_checkpoint(checkpoint_path)
    model.eval()
    model.cuda()

    logger.addHandler(logging.FileHandler(output_folder / "out.log", mode="w"))

    id_test_loader = get_dataloader(args.dataset, "test", 128, data_shape=(32, 32, 1))

    y, logits = model.get_gt_preds(id_test_loader)

    # Compute accuracy
    probs = torch.softmax(logits, dim=1)
    acc = accuracy(y, probs)
    logger.info(f"Accuracy: {acc * 100.:.02f}")

    # Compute calibration
    y_np, probs_np = to_np(y), to_np(probs)
    ece, mce = classification_calibration(y_np, probs_np)
    brier_score = brier_score(y_np, probs_np)
    uncertainty, resolution, reliability = brier_decomposition(y_np, probs_np)

    fig, ax = plt.subplots(figsize=(10, 10))
    draw_reliability_graph(y_np, probs_np, 10, ax=ax)
    fig.savefig(output_folder / "calibration.png", dpi=200)

    logger.info(f"ECE: {ece * 100:.02f}")
    logger.info(f"Brier: {brier_score * 100:.02f}")
    logger.info(f"Brier uncertainty: {uncertainty * 100:.02f}")
    logger.info(f"Brier resolution: {resolution * 100:.02f}")
    logger.info(f"Brier reliability: {reliability * 100:.02f}")
    logger.info(
        f"Brier (via decomposition): {(reliability - resolution + uncertainty) * 100:.02f}"
    )

    # Compute ID OOD scores
    id_scores_dict = model.ood_detect(id_test_loader)

    # Compute OOD detection metrics
    for ood_ds in args.ood_dataset:
        logger.info(f"\n\n{ood_ds}")
        ood_loader = get_dataloader(
            ood_ds,
            "test",
            128,
            data_shape=np.array(id_test_loader.dataset[0][0].shape)[[1, 2, 0]].tolist(),
        )
        ood_scores_dict = model.ood_detect(ood_loader)

        for score_name, id_scores in id_scores_dict.items():
            ood_scores = ood_scores_dict[score_name]

            ax = plot_score_hist(
                id_scores,
                ood_scores,
                title="",  # f"{ood_ds}, {score_name.replace('_', ' ').title()}",
            )
            ax.figure.savefig(str(output_folder / f"{ood_ds}_{score_name}.png"))
            plt.close()

            labels = np.concatenate(
                [np.zeros_like(ood_scores), np.ones_like(id_scores)]
            )
            preds = np.concatenate([ood_scores, id_scores])

            auroc = roc_auc_score(labels, preds)
            aupr = average_precision_score(labels, preds)

            logger.info(score_name)
            logger.info(f"AUROC: {auroc * 100:.02f}")
            logger.info(f"AUPR: {aupr * 100:.02f}")
