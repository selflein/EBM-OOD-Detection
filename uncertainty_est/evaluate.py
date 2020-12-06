import os
import sys

sys.path.insert(0, os.getcwd())

import logging
from pathlib import Path
from argparse import ArgumentParser

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from uncertainty_est.utils.utils import to_np
from uncertainty_est.data.dataloaders import get_dataloader
from uncertainty_est.models.ce_baseline import CEBaseline
from uncertainty_est.utils.metrics import (
    accuracy,
    classification_calibration,
    brier_score,
)


logger = logging.Logger("eval")

parser = ArgumentParser()
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--dataset", type=str)
parser.add_argument("--ood_dataset", type=str, action="append")


if __name__ == "__main__":
    args = parser.parse_args()

    model = CEBaseline.load_from_checkpoint(args.checkpoint)
    model.eval()

    id_test_loader = get_dataloader(args.dataset, "test", 128, img_size=32)

    # Compute accuracy
    y, logits = model.get_gt_preds(id_test_loader)
    probs = torch.softmax(logits, dim=1)
    logger.info(f"Accuracy: {accuracy(y, probs) * 100.:.02f}")

    # Compute calibration
    y_np, probs_np = to_np(y), to_np(probs)
    ece, mce = classification_calibration(y_np, probs_np)
    brier_score = brier_score(y_np, probs_np)

    logger.info(f"ECE: {ece * 100:.02f}")
    logger.info(f"Brier: {brier_score * 100:.02f}")

    # Compute ID OOD scores
    id_scores = to_np(model.ood_detect(id_test_loader, "max"))

    # Compute OOD detection metrics
    for ood_ds in args.ood_dataset:
        logger.info(ood_ds)
        ood_loader = get_dataloader(ood_ds, "test", 128, img_size=32)
        ood_scores = to_np(model.ood_detect(ood_loader, "max"))

        labels = np.concatenate([np.ones_like(ood_scores), np.zeros_like(id_scores)])
        preds = np.concatenate([ood_scores, id_scores])

        auroc = roc_auc_score(labels, preds)
        aupr = average_precision_score(labels, preds)

        logger.info(f"AUROC: {auroc * 100:.02f}")
        logger.info(f"AUPR: {aupr * 100:.02f}")
