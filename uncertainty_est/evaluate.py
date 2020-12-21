import os
import sys

sys.path.insert(0, os.getcwd())

import logging
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

import torch
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from uncertainty_eval.metrics.brier import brier_score, brier_decomposition
from uncertainty_eval.metrics.calibration_error import classification_calibration

from uncertainty_est.utils.utils import to_np
from uncertainty_est.utils.dirichlet import dirichlet_prior_network_uncertainty
from uncertainty_est.data.dataloaders import get_dataloader
from uncertainty_est.models.ce_baseline import CEBaseline
from uncertainty_est.models.priornet.priornet import PriorNet
from uncertainty_est.utils.metrics import accuracy


parser = ArgumentParser()
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--ood_dataset", type=str, action="append")
parser.add_argument("--log_file", type=str)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)


if __name__ == "__main__":
    args = parser.parse_args()

    if args.log_file is not None:
        logger.addHandler(logging.FileHandler(args.log_file, mode="w"))

    model = PriorNet.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.cuda()

    id_test_loader = get_dataloader(args.dataset, "test", 128, img_size=32)
    y, logits = model.get_gt_preds(id_test_loader)

    # Compute accuracy
    probs = torch.softmax(logits, dim=1)
    logger.info(f"Accuracy: {accuracy(y, probs) * 100.:.02f}")

    # Compute calibration
    y_np, probs_np = to_np(y), to_np(probs)
    ece, mce = classification_calibration(y_np, probs_np)
    brier_score = brier_score(y_np, probs_np)
    uncertainty, resolution, reliability = brier_decomposition(y_np, probs_np)

    # fig, ax = plt.subplots(figsize=(10, 10))
    # draw_reliability_graph(y_np, probs_np, 10, ax=ax)
    # fig.savefig("logs/calibration.png", dpi=200)

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
        ood_loader = get_dataloader(ood_ds, "test", 128, img_size=32)
        ood_scores_dict = model.ood_detect(ood_loader)

        for score_name, id_scores in id_scores_dict.items():
            ood_scores = ood_scores_dict[score_name]
            labels = np.concatenate(
                [np.ones_like(ood_scores), np.zeros_like(id_scores)]
            )
            preds = np.concatenate([ood_scores, id_scores])

            auroc = roc_auc_score(labels, preds)
            aupr = average_precision_score(labels, preds)

            logger.info(score_name)
            logger.info(f"AUROC: {auroc * 100:.02f}")
            logger.info(f"AUPR: {aupr * 100:.02f}")
