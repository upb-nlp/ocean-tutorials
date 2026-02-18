import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AdamW,
)

from datasets import Dataset

from sklearn.metrics import f1_score, accuracy_score


# -------------------------
# Config
# -------------------------
@dataclass
class Config:
    model_name: str = "google-bert/bert-base-uncased"
    num_epochs: int = 5
    batch_size: int = 8
    grad_accum: int = 2
    lr: float = 5e-5
    max_length: int = 512
    num_labels1: int = 5
    num_labels2: int = 2
    early_stop_patience: int = 3
    num_workers: int = 0


# -------------------------
# Version 1 model (2 heads)
# -------------------------
class BertWithTwoHeads(torch.nn.Module):
    def __init__(self, model_name: str, num_labels1: int = 5, num_labels2: int = 2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.classifier1 = torch.nn.Linear(hidden_size, num_labels1)
        self.classifier2 = torch.nn.Linear(hidden_size, num_labels2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [B, H]
        logits1 = self.classifier1(pooled)
        logits2 = self.classifier2(pooled)
        return logits1, logits2


# -------------------------
# DataModules
# -------------------------
class EncoderDataModuleV1(pl.LightningDataModule):
    def __init__(
        self,
        cfg: Config,
        dataset_df: pd.DataFrame,
        dataset_train: List[Dict[str, Any]],
        dataset_val: List[Dict[str, Any]],
        dataset_test: List[Dict[str, Any]],
    ):
        super().__init__()
        self.cfg = cfg
        self.dataset_df = dataset_df
        self.dataset_train_raw = dataset_train
        self.dataset_val_raw = dataset_val
        self.dataset_test_raw = dataset_test

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        # label maps
        self.condition_map = {label: i for i, label in enumerate(pd.unique(dataset_df["condition"]))}
        self.reverse_condition_map = {v: k for k, v in self.condition_map.items()}
        self.record_type_map = {label: i for i, label in enumerate(pd.unique(dataset_df["record_type"]))}
        self.reverse_record_type_map = {v: k for k, v in self.record_type_map.items()}

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        inputs = self.tokenizer(
            f"Choose the condition and the record type of the following medical note: {example['text']}",
            truncation=True,
            max_length=self.cfg.max_length,
        )
        inputs["label1"] = self.condition_map[example["condition"]]
        inputs["label2"] = self.record_type_map[example["record_type"]]
        return inputs

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch],
            batch_first=True,
            padding_value=0,
        )
        label1 = torch.tensor([ex["label1"] for ex in batch], dtype=torch.long)
        label2 = torch.tensor([ex["label2"] for ex in batch], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label1": label1,
            "label2": label2,
        }

    def setup(self, stage: Optional[str] = None):
        train = Dataset.from_list(self.dataset_train_raw).map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text", "condition", "record_type"],
        )
        val = Dataset.from_list(self.dataset_val_raw).map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text", "condition", "record_type"],
        )
        test = Dataset.from_list(self.dataset_test_raw)  # keep raw for inference later
        self.train_ds = train
        self.val_ds = val
        self.test_ds = test

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.num_workers,
        )


class EncoderDataModuleV2(pl.LightningDataModule):
    """
    MLM prompt formulation. We also compute max_tokens from label strings (condition + record_type).
    """
    def __init__(
        self,
        cfg: Config,
        dataset_df: pd.DataFrame,
        dataset_train: List[Dict[str, Any]],
        dataset_val: List[Dict[str, Any]],
        dataset_test: List[Dict[str, Any]],
    ):
        super().__init__()
        self.cfg = cfg
        self.dataset_df = dataset_df
        self.dataset_train_raw = dataset_train
        self.dataset_val_raw = dataset_val
        self.dataset_test_raw = dataset_test

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        self.condition_map = {label: i for i, label in enumerate(pd.unique(dataset_df["condition"]))}
        self.reverse_condition_map = {v: k for k, v in self.condition_map.items()}
        self.record_type_map = {label: i for i, label in enumerate(pd.unique(dataset_df["record_type"]))}
        self.reverse_record_type_map = {v: k for k, v in self.record_type_map.items()}

        self.max_tokens = self._compute_max_tokens()

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def _compute_max_tokens(self) -> int:
        max_tokens = 0
        for label in list(self.condition_map.keys()) + list(self.record_type_map.keys()):
            toks = self.tokenizer(label, add_special_tokens=False)["input_ids"]
            max_tokens = max(max_tokens, len(toks))
        return max_tokens

    def tokenize_function(self, example: Dict[str, Any]) -> Dict[str, Any]:
        condition_tokens = self.tokenizer(example["condition"], add_special_tokens=False)["input_ids"]
        record_type_tokens = self.tokenizer(example["record_type"], add_special_tokens=False)["input_ids"]

        prompt = (
            f"Choose the condition and the record type of the following medical note: {example['text']}."
            f" The condition is: {' '.join([self.tokenizer.mask_token] * self.max_tokens)};"
            f" The record type is: {' '.join([self.tokenizer.mask_token] * self.max_tokens)}"
        )

        inputs = self.tokenizer(prompt, truncation=True, max_length=self.cfg.max_length)

        # Create labels: -100 for non-mask positions, and token IDs for mask positions
        input_ids = inputs["input_ids"]
        labels = [-100] * len(input_ids)

        mask_id = self.tokenizer.mask_token_id
        mask_positions = [i for i, tid in enumerate(input_ids) if tid == mask_id]

        # First max_tokens masks -> condition, next max_tokens -> record_type
        for j, pos in enumerate(mask_positions):
            if j < self.max_tokens:
                labels[pos] = condition_tokens[j] if j < len(condition_tokens) else self.tokenizer.pad_token_id
            else:
                k = j - self.max_tokens
                labels[pos] = record_type_tokens[k] if k < len(record_type_tokens) else self.tokenizer.pad_token_id

        inputs["labels"] = labels
        return inputs

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence(
            [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        attention_mask = pad_sequence(
            [torch.tensor(ex["attention_mask"], dtype=torch.long) for ex in batch],
            batch_first=True,
            padding_value=0,
        )
        labels = pad_sequence(
            [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch],
            batch_first=True,
            padding_value=-100,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def setup(self, stage: Optional[str] = None):
        train = Dataset.from_list(self.dataset_train_raw).map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text", "condition", "record_type"],
        )
        val = Dataset.from_list(self.dataset_val_raw).map(
            self.tokenize_function,
            batched=False,
            remove_columns=["text", "condition", "record_type"],
        )
        test = Dataset.from_list(self.dataset_test_raw)  # keep raw for inference later
        self.train_ds = train
        self.val_ds = val
        self.test_ds = test

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=self.cfg.num_workers,
        )


# -------------------------
# LightningModules
# -------------------------
class EncoderLightningModuleV1(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.model = BertWithTwoHeads(cfg.model_name, cfg.num_labels1, cfg.num_labels2)
        self.loss_fn1 = torch.nn.CrossEntropyLoss()
        self.loss_fn2 = torch.nn.CrossEntropyLoss()

        # for epoch-end metrics
        self.val_preds1 = []
        self.val_preds2 = []
        self.val_true1 = []
        self.val_true2 = []

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    def training_step(self, batch, batch_idx):
        logits1, logits2 = self(batch["input_ids"], batch["attention_mask"])
        loss1 = self.loss_fn1(logits1, batch["label1"])
        loss2 = self.loss_fn2(logits2, batch["label2"])
        loss = loss1 + loss2

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/loss1", loss1, on_step=True, on_epoch=True)
        self.log("train/loss2", loss2, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits1, logits2 = self(batch["input_ids"], batch["attention_mask"])
        loss1 = self.loss_fn1(logits1, batch["label1"])
        loss2 = self.loss_fn2(logits2, batch["label2"])
        loss = loss1 + loss2

        pred1 = torch.argmax(logits1, dim=1)
        pred2 = torch.argmax(logits2, dim=1)

        self.val_preds1.append(pred1.detach().cpu())
        self.val_preds2.append(pred2.detach().cpu())
        self.val_true1.append(batch["label1"].detach().cpu())
        self.val_true2.append(batch["label2"].detach().cpu())

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        p1 = torch.cat(self.val_preds1).numpy() if self.val_preds1 else np.array([])
        p2 = torch.cat(self.val_preds2).numpy() if self.val_preds2 else np.array([])
        t1 = torch.cat(self.val_true1).numpy() if self.val_true1 else np.array([])
        t2 = torch.cat(self.val_true2).numpy() if self.val_true2 else np.array([])

        self.val_preds1.clear()
        self.val_preds2.clear()
        self.val_true1.clear()
        self.val_true2.clear()

        if len(t1) > 0:
            f1_1 = f1_score(t1, p1, average="weighted")
            acc_1 = accuracy_score(t1, p1)
            self.log("val/f1_condition", f1_1, prog_bar=True)
            self.log("val/acc_condition", acc_1, prog_bar=True)

        if len(t2) > 0:
            f1_2 = f1_score(t2, p2, average="weighted")
            acc_2 = accuracy_score(t2, p2)
            self.log("val/f1_record_type", f1_2, prog_bar=True)
            self.log("val/acc_record_type", acc_2, prog_bar=True)

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.cfg.lr)


class EncoderLightningModuleV2(pl.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()
        self.model = AutoModelForMaskedLM.from_pretrained(cfg.model_name)

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.cfg.lr)


# -------------------------
# Evaluation helpers (reload best ckpt easily)
# -------------------------
@torch.no_grad()
def evaluate_best_v1(
    ckpt_path: str,
    datamodule: EncoderDataModuleV1,
    cfg: Config,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    model = EncoderLightningModuleV1.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.to(device)
    model.eval()

    # run on raw test set with tokenizer prompt (as your original)
    tokenizer = datamodule.tokenizer
    reverse_condition_map = datamodule.reverse_condition_map
    reverse_record_type_map = datamodule.reverse_record_type_map

    preds_cond, preds_rec = [], []
    true_cond, true_rec = [], []

    for ex in datamodule.dataset_test_raw:
        inputs = tokenizer(
            f"Choose the condition and the record type of the following medical note: {ex['text']}",
            return_tensors="pt",
            truncation=True,
            max_length=cfg.max_length,
        ).to(device)

        logits1, logits2 = model.model(**inputs)
        c = torch.argmax(logits1, dim=1).item()
        r = torch.argmax(logits2, dim=1).item()

        preds_cond.append(reverse_condition_map[c])
        preds_rec.append(reverse_record_type_map[r])

        true_cond.append(ex["condition"])
        true_rec.append(ex["record_type"])

    f1_cond = f1_score(true_cond, preds_cond, average="weighted")
    f1_rec = f1_score(true_rec, preds_rec, average="weighted")
    acc_cond = accuracy_score(true_cond, preds_cond)
    acc_rec = accuracy_score(true_rec, preds_rec)

    return {
        "test/f1_condition": f1_cond,
        "test/f1_record_type": f1_rec,
        "test/acc_condition": acc_cond,
        "test/acc_record_type": acc_rec,
    }


@torch.no_grad()
def evaluate_best_v2(
    ckpt_path: str,
    datamodule: EncoderDataModuleV2,
    cfg: Config,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> Dict[str, float]:
    model = EncoderLightningModuleV2.load_from_checkpoint(ckpt_path, cfg=cfg)
    model.to(device)
    model.eval()

    tokenizer = datamodule.tokenizer
    max_tokens = datamodule.max_tokens

    preds_cond, preds_rec = [], []
    true_cond, true_rec = [], []

    for ex in datamodule.dataset_test_raw:
        prompt = (
            f"Choose the condition and the record type of the following medical note: {ex['text']}."
            f" The condition is: {' '.join([tokenizer.mask_token] * max_tokens)};"
            f" The record type is: {' '.join([tokenizer.mask_token] * max_tokens)}"
        )
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=cfg.max_length).to(device)
        outputs = model.model(**inputs)
        logits = outputs.logits  # [1, T, V]

        mask_positions = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
        pred_ids = inputs["input_ids"].clone().squeeze(0)

        for pos in mask_positions:
            token_logits = logits[0, pos, :]
            pred_ids[pos] = torch.argmax(token_logits, dim=-1)

        decoded = tokenizer.decode(pred_ids, skip_special_tokens=True).lower()

        # robust-ish parsing (still heuristic, like your original)
        try:
            cond = decoded.split("the condition is : ")[1].split(";")[0].strip()
            rec = decoded.split("the record type is : ")[1].strip()
        except Exception:
            cond, rec = "", ""

        preds_cond.append(cond)
        preds_rec.append(rec)
        true_cond.append(ex["condition"].lower())
        true_rec.append(ex["record_type"].lower())

    f1_cond = f1_score(true_cond, preds_cond, average="weighted")
    f1_rec = f1_score(true_rec, preds_rec, average="weighted")
    acc_cond = accuracy_score(true_cond, preds_cond)
    acc_rec = accuracy_score(true_rec, preds_rec)

    return {
        "test/f1_condition": f1_cond,
        "test/f1_record_type": f1_rec,
        "test/acc_condition": acc_cond,
        "test/acc_record_type": acc_rec,
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, choices=[1, 2], required=True, help="Encoder version: 1 or 2")
    parser.add_argument("--save_dir", type=str, default="./checkpoints", help="Root checkpoint directory")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name")
    parser.add_argument("--eval_only", action="store_true", help="Skip training; evaluate best checkpoint")
    args = parser.parse_args()

    cfg = Config()

    # ------------------------------------------------------------------
    # IMPORTANT:
    # You must provide these in your environment/script before running:
    #   dataset (pandas df with columns: condition, record_type)
    #   dataset_train / dataset_val / dataset_test (lists of dicts)
    # Each dict must have: text, condition, record_type
    # ------------------------------------------------------------------
    global dataset, dataset_train, dataset_val, dataset_test

    if args.version == 1:
        saved_model_name = "saved_encoder_model_v1"
        datamodule = EncoderDataModuleV1(cfg, dataset, dataset_train, dataset_val, dataset_test)
        lit_model = EncoderLightningModuleV1(cfg)
        monitor_metric = "val/loss"
        eval_fn = evaluate_best_v1
    else:
        saved_model_name = "saved_encoder_model_v2"
        datamodule = EncoderDataModuleV2(cfg, dataset, dataset_train, dataset_val, dataset_test)
        lit_model = EncoderLightningModuleV2(cfg)
        monitor_metric = "val/loss"
        eval_fn = evaluate_best_v2

    run_name = args.run_name or saved_model_name
    ckpt_dir = os.path.join(args.save_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        save_top_k=1,
        monitor=monitor_metric,
        mode="min",
        save_last=True,
    )
    earlystop_cb = EarlyStopping(monitor=monitor_metric, mode="min", patience=cfg.early_stop_patience)

    logger = TensorBoardLogger(save_dir=os.path.join(args.save_dir, "tb_logs"), name=run_name)

    trainer = pl.Trainer(
        max_epochs=cfg.num_epochs,
        accumulate_grad_batches=cfg.grad_accum,
        callbacks=[checkpoint_cb, earlystop_cb],
        logger=logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    datamodule.setup()

    if not args.eval_only:
        trainer.fit(lit_model, datamodule=datamodule)

    best_ckpt_path = checkpoint_cb.best_model_path
    if not best_ckpt_path:
        # If eval_only and callback didn't run, try to find best.ckpt
        candidate = os.path.join(ckpt_dir, "best.ckpt")
        if os.path.exists(candidate):
            best_ckpt_path = candidate

    if not best_ckpt_path or not os.path.exists(best_ckpt_path):
        raise FileNotFoundError(f"Could not find best checkpoint in {ckpt_dir}")

    # Save tokenizer too (easy deployment)
    datamodule.tokenizer.save_pretrained(ckpt_dir)

    # Evaluate best checkpoint on test set (simple + explicit)
    metrics = eval_fn(best_ckpt_path, datamodule, cfg)
    print("\n=== Best checkpoint test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\nBest checkpoint: {best_ckpt_path}")
    print(f"Tokenizer saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
