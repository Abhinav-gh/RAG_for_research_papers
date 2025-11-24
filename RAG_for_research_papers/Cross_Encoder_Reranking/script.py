#!/usr/bin/env python3
"""
Cross-Encoder Reranker with LoRA fine-tuning

- Uses LoRA adapters (injected into all nn.Linear layers of the encoder)
- Freezes base pretrained weights; trains only LoRA adapters + classifier head
- Uses 90% train, 10% test split
- Saves trained model (weights include LoRA adapters) and tokenizer to output_dir
"""

import os
import math
import random
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

# ------------------- Dataset -------------------

class PairDataset(Dataset):
    def __init__(self, queries: List[str], chunks: List[str], labels: List[int]):
        assert len(queries) == len(chunks) == len(labels)
        self.queries = queries
        self.chunks = chunks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "query": self.queries[idx],
            "chunk": self.chunks[idx],
            "label": float(self.labels[idx]),
        }

def collate_fn(batch: List[Dict], tokenizer, max_length: int):
    queries = [b["query"] for b in batch]
    chunks = [b["chunk"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float)

    enc = tokenizer(
        queries, chunks, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
    )
    if "token_type_ids" not in enc:
        enc["token_type_ids"] = torch.zeros_like(enc["input_ids"])

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "token_type_ids": enc["token_type_ids"],
        "labels": labels,
    }

# ------------------- LoRA Implementation -------------------

class LoRALinear(nn.Module):
    def __init__(self, orig_linear: nn.Linear, r: int = 8, alpha: float = 32.0, dropout: float = 0.0):
        super().__init__()
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / max(1, r)
        self.dropout_p = dropout

        self.weight = nn.Parameter(orig_linear.weight.data.clone(), requires_grad=False)
        if orig_linear.bias is not None:
            self.bias = nn.Parameter(orig_linear.bias.data.clone(), requires_grad=False)
        else:
            self.bias = None

        if r > 0:
            self.A = nn.Parameter(torch.randn(r, self.in_features) * 0.01)
            self.B = nn.Parameter(torch.zeros(self.out_features, r))
        else:
            self.A = None
            self.B = None

        self.dropout = nn.Dropout(self.dropout_p) if self.dropout_p > 0.0 else None

    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        if self.r > 0:
            orig_shape = x.shape
            if x.dim() == 3:
                N, S, D = x.shape
                x_flat = x.reshape(-1, D)
            else:
                x_flat = x
            x_drop = self.dropout(x_flat) if self.dropout is not None else x_flat
            low_rank = (x_drop @ self.A.t()) @ self.B.t()
            low_rank = low_rank * self.scaling
            if x.dim() == 3:
                low_rank = low_rank.view(N, S, self.out_features)
                out = base + low_rank
            else:
                out = base + low_rank
            return out
        else:
            return base

def replace_linear_with_lora(module: nn.Module, r: int, alpha: float, dropout: float):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            lora_linear = LoRALinear(child, r=r, alpha=alpha, dropout=dropout)
            setattr(module, name, lora_linear)
        else:
            replace_linear_with_lora(child, r=r, alpha=alpha, dropout=dropout)

def inject_lora(model: nn.Module, r: int, alpha: float, dropout: float):
    encoder = model.encoder
    replace_linear_with_lora(encoder, r=r, alpha=alpha, dropout=dropout)
    for name, p in encoder.named_parameters():
        p.requires_grad = False
    for name, module in encoder.named_modules():
        if isinstance(module, LoRALinear):
            if module.A is not None:
                module.A.requires_grad = True
            if module.B is not None:
                module.B.requires_grad = True
    return model

# ------------------- Model -------------------

class CrossEncoderLoRA(nn.Module):
    def __init__(self, model_name_or_path: str, dropout_prob: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=self.encoder.config.initializer_range)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        cls_emb = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_emb)
        logits = self.classifier(x).squeeze(-1)
        return logits, cls_emb

# ------------------- Helper functions -------------------

def load_csv_dataset(path: str):
    df = pd.read_csv(path)
    assert {"query", "chunk", "label"} <= set(df.columns)
    return df

def evaluate(model, dataloader, device):
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)
            logits, _ = model(input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
            loss = loss_fn(logits, labels)
            total_loss += loss.item() * len(labels)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-logits))
    auc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    return {"auc": auc, "acc": acc, "loss": total_loss / len(labels)}

def train_one_epoch(train_loader, model, optimizer, scheduler, device, max_grad_norm):
    model.train()
    loss_fn = nn.BCEWithLogitsLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        logits, _ = model(input_ids, attention_mask=attn_mask, token_type_ids=token_type_ids)
        loss = loss_fn(logits, labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# ------------------- Main -------------------

def main(args):
    df = load_csv_dataset(args.data_csv)

    # Split 90% train, 10% test
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42, stratify=df["label"])

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    train_ds = PairDataset(train_df["query"].tolist(), train_df["chunk"].tolist(), train_df["label"].tolist())
    test_ds = PairDataset(test_df["query"].tolist(), test_df["chunk"].tolist(), test_df["label"].tolist())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length))
    test_loader = DataLoader(test_ds, batch_size=args.eval_batch_size, shuffle=False,
                             collate_fn=lambda b: collate_fn(b, tokenizer, args.max_length))

    # Build model and inject LoRA
    model = CrossEncoderLoRA(args.model_name_or_path, dropout_prob=args.dropout)
    model = inject_lora(model, r=args.lora_rank, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.to(args.device)

    # Ensure classifier is trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    # Optimizer + scheduler
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_steps_ratio * total_steps),
        num_training_steps=total_steps,
    )

    # Train
    for epoch in range(args.epochs):
        train_one_epoch(train_loader, model, optimizer, scheduler, args.device, args.max_grad_norm)
        metrics = evaluate(model, train_loader, args.device)
        print(f"Epoch {epoch+1}/{args.epochs} - Train AUC: {metrics['auc']:.4f}, Acc: {metrics['acc']:.4f}, Loss: {metrics['loss']:.4f}")

    # Evaluate on test set
    test_metrics = evaluate(model, test_loader, args.device)
    print("\nTest Set Results:", test_metrics)

    # Save model and tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "model_with_lora.pt"))
    tokenizer.save_pretrained(args.output_dir)
    print("Saved model (with LoRA adapters) and tokenizer to", args.output_dir)

# ------------------- Args -------------------

class Args:
    data_csv = "./training_pairs.csv"
    model_name_or_path = "bert-base-uncased"
    output_dir = "./crossenc_lora_out"

    epochs = 5
    batch_size = 16
    eval_batch_size = 64
    max_length = 256
    lr = 3e-5
    warmup_steps_ratio = 0.06
    max_grad_norm = 1.0
    dropout = 0.1

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    lora_rank = 32
    lora_alpha = 64
    lora_dropout = 0.0

if __name__ == "__main__":
    args = Args()
    args.device = torch.device(args.device)
    main(args)