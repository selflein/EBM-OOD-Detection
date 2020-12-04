import torch
import pytorch_lightning as pl
import torch.nn.functional as F


class CEBaseline(pl.LightningModule):
    def __init__(self, backbone, learning_rate, momentum, weight_decay):
        super().__init__()
        self.backbone = backbone
        self.lr = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.backbone(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        loss = F.cross_entropy(y_hat, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", loss)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("val_acc", acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.backbone(x)

        acc = (y == y_hat.argmax(1)).float().mean(0).item()
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(),
            betas=(self.momentum, 0.999),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=30, gamma=0.5)
        return [optim], [scheduler]

    def get_gt_preds(self, loader):
        gt, preds = [], []
        for x, y in loader:
            y_hat = self(x)
            gt.append(y)
            preds.append(y_hat)
        return torch.cat(gt), torch.cat(preds)

    def ood_detect(self, loader, method):
        ood_scores = []
        for x, _ in loader:
            y_hat = self(x)
            probs = torch.softmax(y_hat, dim=1)
            ood_scores.append(torch.max(probs, dim=1)[0])
        return torch.cat(ood_scores)
