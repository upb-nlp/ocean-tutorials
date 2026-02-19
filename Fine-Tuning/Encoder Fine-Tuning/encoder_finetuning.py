import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type, Callable, Union

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForMaskedLM,
    AdamW,
)

from datasets import Dataset, load_dataset
from sklearn.metrics import f1_score, accuracy_score
from wandb import login


# -----------------------------
# Data
# -----------------------------
@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

dataclass(frozen=True)
class EncoderSpec:
    saved_model_name: str
    datamodule_cls: Type[Any]
    lit_model_cls: Type[Any]
    eval_fn: Callable[..., Dict[str, float]]
    monitor_metric: str = "val/loss"
    monitor_mode: str = "min"


def get_encoder_spec(version: int) -> EncoderSpec:
    specs: Dict[int, EncoderSpec] = {
        1: EncoderSpec(
            saved_model_name="saved_encoder_model_v1",
            datamodule_cls=EncoderDataModuleV1,
            lit_model_cls=EncoderLightningModuleV1,
            eval_fn=evaluate_best_v1,
        ),
        2: EncoderSpec(
            saved_model_name="saved_encoder_model_v2",
            datamodule_cls=EncoderDataModuleV2,
            lit_model_cls=EncoderLightningModuleV2,
            eval_fn=evaluate_best_v2,
        ),
    }
    if version not in specs:
        raise ValueError(f"Unknown VERSION={version}. Supported: {list(specs)}")
    return specs[version]


def resolve_best_ckpt(checkpoint_cb: ModelCheckpoint, ckpt_dir: str) -> str:
    best = checkpoint_cb.best_model_path

    # If eval_only, callback may not run — fall back to conventional names.
    if not best:
        for candidate in (
            os.path.join(ckpt_dir, "best.ckpt"),
            os.path.join(ckpt_dir, "best"),
            os.path.join(ckpt_dir, "last.ckpt"),
        ):
            if os.path.exists(candidate):
                best = candidate
                break

    if not best or not os.path.exists(best):
        raise FileNotFoundError(f"Could not find checkpoint in {ckpt_dir}")
    return best


# -------------------------
# Version 1 model (2 heads)
# -------------------------
class EncoderWithTwoHeads(torch.nn.Module):
    def __init__(self, model_name: str, num_labels1: int = 5, num_labels2: int = 2):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size
        self.classifier1 = torch.nn.Linear(hidden_size, num_labels1)
        self.classifier2 = torch.nn.Linear(hidden_size, num_labels2)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [B, H]
        logits1 = self.classifier1(pooled)
        logits2 = self.classifier2(pooled)
        return logits1, logits2


# -------------------------
# DataModules
# -------------------------
class EncoderDataModuleV1(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str,
        tokenizer_name: str,
        max_length: int,
        batch_size: int,
        num_workers: int,
        seed: int,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self):
        # Load dataset and tokenizer
        load_dataset(self.dataset_name, split=self.train_split)
        AutoTokenizer.from_pretrained(self.tokenizer_name)

    # Continue from here
    def setup(self, stage: Optional[str] = None):

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.dataset = load_dataset(self.dataset_name)
        self.train = self.dataset["train"]
        self.val = self.dataset["validation"]
        self.test = self.dataset["test"]

        # label maps
        self.condition_map = {label: i for i, label in enumerate(pd.unique(self.train["condition"]))}
        self.reverse_condition_map = {v: k for k, v in self.condition_map.items()}
        self.record_type_map = {label: i for i, label in enumerate(pd.unique(self.train["record_type"]))}
        self.reverse_record_type_map = {v: k for k, v in self.record_type_map.items()}

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


class EncoderDataModuleV2(L.LightningDataModule):
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
class EncoderLightningModuleV1(L.LightningModule):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        self.model = EncoderWithTwoHeads(cfg.model_name, cfg.num_labels1, cfg.num_labels2)
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


class EncoderLightningModuleV2(L.LightningModule):
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


# -----------------------------
# Training loop (called by main)
# -----------------------------
def run_encoder_training(
    *,
    seed: int,
    version: int,
    model_name: str,
    num_epochs: int,
    grad_accum: int,
    early_stop_patience: int,
    max_length: int,
    batch_size: int,
    learning_rate: float,
    log_every_n_steps: int,
    save_dir: str,
    accelerator: str = "auto",
    devices: Union[str, int] = "auto",
    precision: str = "32-true",
    eval_only: bool = False,
    run_name: Optional[str] = None,
):
    L.seed_everything(seed, workers=True)

    spec = get_encoder_spec(version)

    run_name = spec.saved_model_name
    ckpt_dir = os.path.join(save_dir, run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ---- DataModule / Model ----
    dm = spec.datamodule_cls(cfg, dataset, dataset_train, dataset_val, dataset_test)
    lit_model = spec.lit_model_cls(cfg)

    # ---- Logger ----
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(args.save_dir, "tb_logs"),
        name=run_name,
    )

    # ---- Checkpointing / Early stopping ----
    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best",
        save_top_k=1,
        monitor=spec.monitor_metric,
        mode=spec.monitor_mode,
        save_last=True,
    )
    earlystop_cb = EarlyStopping(
        monitor=spec.monitor_metric,
        mode=spec.monitor_mode,
        patience=cfg.early_stop_patience,
    )

    # ---- Trainer ----
    trainer = L.Trainer(
        max_epochs=cfg.num_epochs,
        accumulate_grad_batches=cfg.grad_accum,
        callbacks=[checkpoint_cb, earlystop_cb],
        logger=tb_logger,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=log_every_n_steps,
        default_root_dir=ckpt_dir,
    )

    dm.setup()

    if not eval_only:
        trainer.fit(lit_model, datamodule=dm)

    # ---- Resolve best checkpoint, save tokenizer, eval ----
    best_ckpt_path = resolve_best_ckpt(checkpoint_cb, ckpt_dir)

    # Save tokenizer too (easy deployment)
    dm.tokenizer.save_pretrained(ckpt_dir)

    metrics = spec.eval_fn(best_ckpt_path, dm, cfg)
    print("\n=== Best checkpoint test metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    print(f"\nBest checkpoint: {best_ckpt_path}")
    print(f"Tokenizer saved to: {ckpt_dir}")


# -------------------------
# Main
# -------------------------
def main():
    # -------------------------
    # Constants (all here)
    # -------------------------
    SEED = 42

    # Model
    MODEL_NAME = "Qwen/Qwen3-8B"

    # Tokenization / batching
    MAX_LENGTH = 1028
    BATCH_SIZE = 1              # per GPU
    GRAD_ACCUM = 16                   # effective batch = BATCH_SIZE * GRAD_ACCUM

    # Training
    LR = 2e-5
    LOG_EVERY_N_STEPS = 10
    NUM_EPOCHS = 10
    EARLY_STOP_PATIENCE = 3

    # Precision
    PRECISION = "bf16-mixed"  # if your GPUs support bf16; otherwise use "16-mixed"

    # Output
    SAVE_DIR = "./encoder_finetuning"

    # Version: 
    # 1: Encoder backbone with classification heads
    # 2: Encoder with masked token prediction-style classification
    VERSION = 1

    wandb_key = os.getenv("WANDB_KEY")
    if wandb_key:
        # Authenticate using the key
        login(key=wandb_key)
        print("Successfully logged into wandb.")
    else:
        print("Error: WANDB_API_KEY environment variable not found.")

    run_encoder_training(
        seed=SEED,
        version=VERSION,
        model_name=MODEL_NAME,
        num_epochs=NUM_EPOCHS,
        grad_accum=GRAD_ACCUM,
        early_stop_patience=EARLY_STOP_PATIENCE,
        max_length=MAX_LENGTH,
        batch_size=BATCH_SIZE,
        learning_rate=LR,
        log_every_n_steps=LOG_EVERY_N_STEPS,
        save_dir=SAVE_DIR,
        accelerator="auto",
        devices="auto",
        precision=PRECISION,
        eval_only=False,  # True to skip training and only eval best ckpt
        run_name=None,  # or "my_run_name"
    )



    


if __name__ == "__main__":
    main()
