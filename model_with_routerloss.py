import time

import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn.functional as F
from _TTMoE_with_routerloss_multitask_wrapper import MoEsparseRouting, SharedState

class CustomLightningModule(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.learning_rate = config["learning_rate"]
        self.model = model
        self.data = config["dataset_name"]
        self.val_f1= torchmetrics.F1Score(task="multiclass",num_classes=2, average = 'micro')
        self.test_f1 = torchmetrics.F1Score(task="multiclass",num_classes=2, average = 'micro')
        self.val_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)
        self.test_acc = torchmetrics.Accuracy(task="multiclass", num_classes=2)

    # def forward(self, input_ids, attention_mask, labels):
    #     return self.model(input_ids, attention_mask=attention_mask, labels=labels)

    def forward(self, input_ids, attention_mask, labels, expert_label=None):
        if isinstance(self.model, MoEsparseRouting):
            return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, expert_label=expert_label)
    
        # If it's a regular model (like LlamaForSequenceClassification), don't pass expert_label
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


    def training_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["label"], expert_label=batch["expert_label"])
        total_loss = outputs["loss"] + SharedState.router_loss_weight * SharedState.routerloss
        self.log("train_loss", outputs["loss"], prog_bar=True)
        self.log("router_loss", SharedState.routerloss, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["label"], expert_label=batch["expert_label"])
        total_loss = outputs["loss"] + SharedState.router_loss_weight * SharedState.routerloss
        self.log("val_loss", outputs["loss"], prog_bar=False)
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.val_acc(predicted_labels, batch["label"])
        self.log("router_loss", SharedState.routerloss, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)
        self.log("val_acc", self.val_acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(batch["input_ids"], attention_mask=batch["attention_mask"],
                    labels=batch["label"], expert_label=batch["expert_label"])
        total_loss = outputs["loss"] + SharedState.router_loss_weight * SharedState.routerloss
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        self.test_acc(predicted_labels, batch["label"])
        self.log("router_loss", SharedState.routerloss, prog_bar=True)
        self.log("total_loss", total_loss, prog_bar=True)
        self.log("accuracy", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
