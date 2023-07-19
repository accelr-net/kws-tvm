import torch
import torchvision
from pytorch_lightning import LightningModule
from torchmetrics.classification import MulticlassAccuracy



class KWSModel(LightningModule):
    def __init__(self, num_classes=35, epochs=30, lr=0.001, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.model = torchvision.models.resnet18(num_classes=num_classes)
        self.model.conv1 = torch.nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        mels, labels, _ = batch
        preds = self.model(mels)
        loss = self.hparams.criterion(preds, labels)
        return {'loss': loss}

    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("train_loss", avg_loss, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        return self.test_epoch_end(outputs)

    def test_step(self, batch, batch_idx):
        mels, labels, wavs = batch
        preds = self.model(mels)
        loss = self.hparams.criterion(preds, labels)
        metric = MulticlassAccuracy(num_classes=35, top_k=1 )
        
        acc = metric(preds, labels) * 100.
        return {"preds": preds, 'test_loss': loss, 'test_acc': acc}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        self.log("test_loss", avg_loss, on_epoch=True, prog_bar=True)
        self.log("test_acc", avg_acc, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=self.hparams.epochs)
        return [optimizer], [lr_scheduler]

    def setup(self, stage=None):
        self.hparams.criterion = torch.nn.CrossEntropyLoss()






       