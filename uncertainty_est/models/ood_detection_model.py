import abc
import inspect
from os import path
from typing import Dict
import copy

import torch
import numpy as np
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
from pytorch_lightning.utilities.parsing import get_init_args

from uncertainty_eval.vis import plot_score_hist


class OODDetectionModel(pl.LightningModule):
    def __init__(self, test_ood_dataloaders=[]):
        super().__init__()
        self.test_ood_dataloaders = test_ood_dataloaders

    @abc.abstractmethod
    def ood_detect(self, loader) -> Dict[str, np.array]:
        pass

    def test_epoch_end(self, *args, **kwargs):
        if len(self.test_ood_dataloaders) < 1:
            return

        # Compute ID OOD scores
        id_scores_dict = self.ood_detect(self.test_dataloader.dataloader)

        # Compute OOD detection metrics
        for dataset_name, loader in self.test_ood_dataloaders:
            ood_scores_dict = self.ood_detect(loader)

            for score_name, id_scores in id_scores_dict.items():
                ood_scores = ood_scores_dict[score_name]

                if self.logger is not None and self.logger.save_dir is not None:
                    ax = plot_score_hist(
                        id_scores,
                        ood_scores,
                        title="",
                    )
                    ax.figure.savefig(
                        path.join(
                            self.logger.save_dir, f"{dataset_name}_{score_name}.png"
                        )
                    )
                    plt.close()

                labels = np.concatenate(
                    [np.zeros_like(ood_scores), np.ones_like(id_scores)]
                )
                preds = np.concatenate([ood_scores, id_scores])

                self.log(f"{dataset_name}: AUROC", roc_auc_score(labels, preds))
                self.log(
                    f"{dataset_name}: AUPR", average_precision_score(labels, preds)
                )

    def optimizer_step(
        self,
        epoch: int = None,
        batch_idx: int = None,
        optimizer=None,
        optimizer_idx: int = None,
        optimizer_closure=None,
        on_tpu: bool = None,
        using_native_amp: bool = None,
        using_lbfgs: bool = None,
        **kwargs,
    ):
        # learning rate warm-up
        if (
            optimizer is not None
            and hasattr(self, "warmup_steps")
            and self.trainer.global_step < self.warmup_steps
        ):
            lr_scale = min(
                1.0, float(self.trainer.global_step + 1) / float(self.warmup_steps)
            )
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate

        optimizer.step(closure=optimizer_closure)

    # TODO: Remove when https://github.com/PyTorchLightning/pytorch-lightning/pull/6056 is merged
    def save_hyperparameters(
        self, *args, ignore=["test_ood_dataloaders"], frame=None
    ) -> None:
        if not frame:
            frame = inspect.currentframe().f_back
        init_args = get_init_args(frame)
        assert init_args, "failed to inspect the self init"

        if ignore is not None:
            if isinstance(ignore, str):
                ignore = [ignore]
            if isinstance(ignore, (list, tuple)):
                ignore = [arg for arg in ignore if isinstance(arg, str)]
            init_args = {k: v for k, v in init_args.items() if k not in ignore}

        if not args:
            # take all arguments
            hp = init_args
            self._hparams_name = "kwargs" if hp else None
        else:
            # take only listed arguments in `save_hparams`
            isx_non_str = [i for i, arg in enumerate(args) if not isinstance(arg, str)]
            if len(isx_non_str) == 1:
                hp = args[isx_non_str[0]]
                cand_names = [k for k, v in init_args.items() if v == hp]
                self._hparams_name = cand_names[0] if cand_names else None
            else:
                hp = {arg: init_args[arg] for arg in args if isinstance(arg, str)}
                self._hparams_name = "kwargs"

        # `hparams` are expected here
        if hp:
            self._set_hparams(hp)
        # make deep copy so  there is not other runtime changes reflected
        self._hparams_initial = copy.deepcopy(self._hparams)
