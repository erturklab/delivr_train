import lightning as L
import torch
import torchmetrics

from torch.nn import BCELoss
import torch.nn.functional as F

from monai.networks.nets import BasicUNet

from ranger21 import Ranger21


class LightningUnet(L.LightningModule):

    def __init__(self, learning_rate=1e-4, batch_size=4, epochs=100, len_dataloader=1):
        super().__init__()
        self.network = BasicUNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=1,
                features=(32, 32, 64, 128, 256, 32),
                # features=(8, 8, 16, 16, 32, 8),
                # features=(8, 8, 8, 16, 16, 8),
                dropout=0.1,
                act="mish",
            )

        # self.example_input_array = {"volume":torch.rand((1,1,100,100,100)),"segmentation":torch.rand((1,1,100,100,100))}
    
        self.epochs = epochs
        self.len_dataloader = len_dataloader

        # self.conf_matrix = torchmetrics.classification.BinaryConfusionMatrix(threshold=0.5)
        self.val_dice       = torchmetrics.Dice(zero_division=1)
        self.val_acc        = torchmetrics.classification.BinaryAccuracy()
        self.val_precision  = torchmetrics.classification.BinaryPrecision()
        self.val_recall     = torchmetrics.classification.BinaryRecall()

        self.validation_step_outputs = []

        self.batch_size = batch_size
        self.log("batch_size",self.batch_size)
        self.learning_rate = learning_rate
        self.loss = BCELoss()
        self.automatic_optimization = False

        self.test_dice      = torchmetrics.Dice(zero_division=1)
        self.test_acc       = torchmetrics.classification.BinaryAccuracy()
        self.test_precision = torchmetrics.classification.BinaryPrecision()
        self.test_recall    = torchmetrics.classification.BinaryRecall()
        self.save_hyperparameters()

    def forward(self, x):
        logits = self.network(x)
        result = F.sigmoid(logits)
        return result

    def validation_step(self, batch, batch_idx):
        x, y = batch["volume"], batch["segmentation"]
        z = self.forward(x)
        # z = F.sigmoid(z)

        self.val_dice(z, y.int())
        self.val_acc(z, y.int())
        self.val_precision(z, y.int())
        self.val_recall(z, y.int())

        self.log("val_acc", self.val_acc)
        self.log("val_precision", self.val_precision)
        self.log("val_rec", self.val_recall)
        self.log("val_dice", self.val_dice, on_epoch=True)

        loss = self.loss(z, y)
        self.log("val_loss", loss)

    
    def test_step(self, batch, batch_idx):
        x, y = batch["volume"], batch["segmentation"]
        z = self.forward(x)
        # z = F.sigmoid(z)
        
        self.test_dice(z, y.int())
        self.test_acc(z, y.int())
        self.test_precision(z, y.int())
        self.test_recall(z, y.int())

        loss = self.loss(z, y)
        self.log("test_loss", loss)
        self.validation_step_outputs.clear() 
        # dice = self.dice(z, y.int())
        # self.log("test_dice", dice)

    def on_validation_epoch_end(self):
        self.log("val_acc", self.val_acc, on_epoch=True)
        self.log("val_precision", self.val_precision, on_epoch=True)
        self.log("val_rec", self.val_recall, on_epoch=True)
        self.log("val_dice", self.val_dice, on_epoch=True)

        self.validation_step_outputs.append({"val_dice":self.val_dice})
        self.validation_step_outputs.append({"val_acc":self.val_acc})
        self.validation_step_outputs.append({"val_precision":self.val_precision})
        self.validation_step_outputs.append({"val_recall":self.val_recall})

        return {"val_dice" : self.val_dice,
                "val_acc" : self.val_acc, 
                "val_precision" : self.val_precision, 
                "val_recall" : self.val_recall}

    def on_test_epoch_end(self):
        self.log("test_dice", self.test_dice)
        self.log("test_acc", self.test_acc, on_epoch=True)
        self.log("test_precision", self.test_precision, on_epoch=True)
        self.log("test_recall", self.test_recall, on_epoch=True)
        
        return {"test_dice": self.test_dice,
                "test_acc" : self.test_acc, 
                "test_precision" : self.test_precision, 
                "test_recall" : self.test_recall}

       


    # def on_validation_epoch_end(self):
    #     all_preds = torch.stack(self.validation_step_outputs)

    #     self.validation_step_outputs.clear()

    def training_step(self, batch, batch_idx):
        x, y = batch["volume"], batch["segmentation"]
        z = self.forward(x)
        z = F.sigmoid(z)
        loss = self.loss(z, y)

        self.log("train_loss", loss)

        self.manual_backward(loss)

        optimizer= self.optimizers()
        # scheduler = self.lr_schedulers()
        optimizer.step()
        optimizer.zero_grad()
        # scheduler.step()
    
    def configure_optimizers(self):
        optimizer = Ranger21(self.parameters(),
                             lr                     = self.learning_rate,
                             num_epochs             = self.epochs,
                             num_batches_per_epoch  = self.len_dataloader)
        return optimizer

