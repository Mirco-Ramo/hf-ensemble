from typing import *
from functools import partial
import os
import sys

import torch
import evaluate
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from lightning import LightningDataModule
import pytorch_lightning as pl
import torch.distributed

from datasets import DatasetDict

from dataset.dataset import ParallelFilesDataset


class ParallelDataModule(LightningDataModule):

    def __init__(
        self,
        base_dir: str,
        tokenizer: AutoTokenizer = None,
        max_seq_length: int = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.num_workers = kwargs.get("num_workers", 1)
        self.drop_last = kwargs.get("drop_last", False)
        self.pin_memory = kwargs.get("pin_memory", False)
        self.truncation = kwargs.get("truncation", True)
        self.fast_dev_run = kwargs.get("fast_dev_run", False)
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size 
        self.eval_batch_size = eval_batch_size

    def setup(self, stage: str):
        raw_train = ParallelFilesDataset(os.path.join(self.base_dir,"train.sl"), os.path.join(self.base_dir,"train.tl"), None).to_hf_dataset() if os.path.exists(os.path.join(self.base_dir,"train.sl")) else None
        raw_valid = ParallelFilesDataset(os.path.join(self.base_dir,"valid.sl"), os.path.join(self.base_dir,"valid.tl"), None).to_hf_dataset() if os.path.exists(os.path.join(self.base_dir,"valid.sl")) else None

        self.dataset = DatasetDict({"train": raw_train, "validation": raw_valid})

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=self.dataset[split].column_names,
                load_from_cache_file=True,
                cache_file_name=os.path.join(self.base_dir, f"{split}_cache_file")
            )

            self.columns = [c for c in self.dataset[split].column_names if c in ["input_ids", "attention_mask", "labels"]]
            self.dataset[split].set_format(type="torch", columns=self.columns)


    def prepare_data(self):
        # download, IO, etc. Useful with shared filesystems
        # only called on 1 GPU/TPU in distributed
        ...

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=self.num_workers, drop_last=self.drop_last, pin_memory=self.pin_memory) if "test" in self.dataset.keys() else None

    def convert_to_features(self, example_batch, indices=None):
        src = [ex["sl"] for ex in example_batch["translation"]]
        tgt = [ex["tl"] for ex in example_batch["translation"]]
        model_inputs = self.tokenizer(src, max_length=self.max_seq_length, truncation=True, padding="max_length")
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(tgt, max_length=self.max_seq_length, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def teardown(self, stage: str) -> None:
        # clean up state after the trainer stops, delete files...
        # called on every process in DDP
        ...


class LightningTransformer(pl.LightningModule):
    def __init__(self, model, dm, *args, **kwargs) -> None:
        super().__init__()
        self.model = model
        self.lr = kwargs.pop("lr") if lr in kwargs else None
        self.save_hyperparameters(ignore=["model"])
        self.metric = evaluate.load("sacrebleu")
        self.dm=dm
        self.val_step_preds = []        # save outputs in each batch to compute metric overall epoch
        self.val_step_labels = []
        self.val_step_losses = []
        self.loss = nn.CrossEntropyLoss()

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs["loss"]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def train_dataloader(self) -> DataLoader:
        return self.dm.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.dm.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.dm.test_dataloader()

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs["loss"], outputs["logits"]
        preds = torch.argmax(logits, dim=-1)
        labels = batch["labels"]
        self.log('val_loss', val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.val_step_losses.append(val_loss)
        self.val_step_preds.extend(preds)
        self.val_step_labels.extend(labels)
        return {"loss": val_loss, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx: int):
        outputs = self(**batch)
        test_loss, logits = outputs[:2]
        preds = torch.argmax(logits, dim=1)
        labels = batch["labels"]
        return {"loss": test_loss, "preds": preds, "labels": labels}

    def on_validation_epoch_end(self):
        val_all_preds = self.val_step_preds
        val_all_losses = self.val_step_losses
        val_all_labels = self.val_step_labels
        preds = torch.cat([x for x in val_all_preds]).detach().cpu().numpy()
        labels = torch.cat([x for x in val_all_labels]).detach().cpu().numpy()
        loss = torch.stack([x.unsqueeze(0) for x in val_all_losses]).squeeze(0).mean()
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        bleu = self.metric.compute(predictions=self.dm.tokenizer.batch_decode(preds), references=self.dm.tokenizer.batch_decode(labels))
        self.log("bleu_score", bleu["score"], prog_bar=True, on_step=False, on_epoch=True)
        self.val_step_preds.clear()
        self.val_step_losses.clear()
        self.val_step_labels.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=32000,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]
