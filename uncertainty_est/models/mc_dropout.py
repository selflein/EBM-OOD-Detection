import torch
from torch import nn

from uncertainty_est.models.ce_baseline import CEBaseline


class MCDropout(CEBaseline):
    def __init__(
        self,
        arch_name,
        arch_config,
        learning_rate,
        momentum,
        weight_decay,
        num_samples=10,
        **kwargs
    ):
        super().__init__(
            arch_name=arch_name,
            arch_config=arch_config,
            learning_rate=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            **kwargs
        )
        self.__dict__.update(locals())
        self.save_hyperparameters()
        self.num_samples = num_samples

    def __set_dropout(self, mode=True):
        for m in self.backbone.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                m.train(mode)

    def get_ood_scores(self, x):
        self.__set_dropout(True)

        samples = []
        for _ in range(self.num_samples):
            samples.append(self(x).cpu())

        variance = torch.stack(samples, 1).var(1).mean(1)
        dir_uncert = {"Variance": variance}
        return dir_uncert
