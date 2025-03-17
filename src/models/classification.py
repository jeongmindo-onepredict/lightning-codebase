import torch
from torch import nn

import lightning as L
from lightning.pytorch.loggers import WandbLogger

from torchmetrics import Accuracy

try:
    import wandb
finally:
    pass


class ClassificationModel(L.LightningModule):
    def __init__(
        self,
        encoder: nn.Module,
        predictor: nn.Module,
        criterion: nn.Module,
        vis_per_batch: int,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["encoder", "predictor", "criterion"])

        self.encoder = encoder
        self.predictor = predictor
        self.criterion = criterion
        self.vis_per_batch = vis_per_batch

        self.accuracy = Accuracy(task="multiclass", num_classes=10)

    def forward(self, x):
        features = self.encoder(x)
        return self.predictor(features)

    def on_fit_start(self) -> None:
        pass

    def training_step(self, batch, batch_idx):
        x, y, motor_id = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        # 로깅
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def on_validation_epoch_start(self) -> None:
        pass

    def validation_step(self, batch, batch_idx):
        x, y, motor_id = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        acc = self.accuracy(logits, y)

        self.log_dict(
            {
                "val/loss": loss.item(),
                "val/acc": acc,
            },
            on_epoch=True,
            on_step=False,
        )

    def on_validation_epoch_end(self) -> None:
        pass

    def test_step(self, batch, batch_idx):
        img, labels = batch
        pred = self(img)

        acc = self.accuracy(pred, labels)
        self.log("test/acc", acc)
