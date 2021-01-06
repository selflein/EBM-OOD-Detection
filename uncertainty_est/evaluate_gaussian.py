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
from torch.utils.data import TensorDataset, DataLoader

from uncertainty_est.utils.utils import to_np
from uncertainty_est.utils.metrics import accuracy
from uncertainty_est.models import MODELS


parser = ArgumentParser()
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--model", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    assert checkpoint.is_file()

    model = MODELS[args.model].load_from_checkpoint(args.checkpoint)
    model.eval()
    model.cuda()

    interp = torch.linspace(-10, 10, 500)
    x, y = torch.meshgrid(interp, interp)
    data = torch.stack((x.reshape(-1), y.reshape(-1)), 1)

    ds = TensorDataset(data, torch.empty(len(data)).fill_(-1))
    dl = DataLoader(ds, 128)

    ood_scores = model.ood_detect(dl)
    for score_name, ood_score in ood_scores.items():
        ood_grid = ood_score.reshape(*x.shape)

        fig, ax = plt.subplots()
        ax.set_title(f"{args.model}: {score_name.replace('_', ' ').title()}")
        mesh = ax.pcolormesh(to_np(x), to_np(y), ood_grid)
        fig.colorbar(mesh)

        fig.savefig(
            checkpoint.parent / f"{score_name}_heatmap.png", bbox_inches="tight"
        )
