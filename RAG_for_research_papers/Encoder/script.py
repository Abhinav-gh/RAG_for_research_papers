# bert_encoder_from_scratch_with_pooling_multitype_allpairs.py
# Modified version that supports:
# - 'C' = Chunk (uses MLM + NSP) → uses ALL possible positive & negative pairs
# - 'Q' = Query (uses MLM only)
# - Added model saving and evaluation on test subset
# - Added Mixture-of-Experts (MoE) in feedforward with Top-K=2 routing and 5 experts
# - Modified to use mask-aware mean pooling instead of CLS token pooling

import random
import math
import os
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from gensim.models import Word2Vec
import chromadb
from pathlib import Path

# -------------------------
# Config
# -------------------------
VOCAB_MIN_FREQ = 1
MAX_SEQ_LEN = 1024
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
FFN_DIM = 3072
DROPOUT = 0.1
WORD2VEC_SIZE = HIDDEN_SIZE
WORD2VEC_WINDOW = 5
WORD2VEC_MIN_COUNT = 1
MLM_MASK_PROB = 0.15
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100

# -------------------------
# Special tokens
# -------------------------
PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
UNK_TOKEN = "[UNK]"
SPECIAL_TOKENS = [PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN, UNK_TOKEN]

# -------------------------
# Utility: Vocab builder
# -------------------------
def build_vocab(sentences: List[str], min_freq: int = VOCAB_MIN_FREQ):
    from collections import Counter
    token_counts = Counter()
    for s in sentences:
        tokens = s.strip().split()
        token_counts.update(tokens)
    stoi, itos = {}, []
    for t in SPECIAL_TOKENS:
        stoi[t] = len(itos)
        itos.append(t)
    for token, cnt in token_counts.items():
        if cnt >= min_freq and token not in stoi:
            stoi[token] = len(itos)
            itos.append(token)
    return stoi, itos

# -------------------------
# Train or load Word2Vec
# -------------------------
def train_word2vec(sentences: List[str], vector_size=WORD2VEC_SIZE, window=WORD2VEC_WINDOW, min_count=WORD2VEC_MIN_COUNT, epochs=5):
    tokenized = [s.strip().split() for s in sentences]
    w2v = Word2Vec(sentences=tokenized, vector_size=vector_size, window=window, min_count=min_count, epochs=epochs, sg=0)
    return w2v

def build_embedding_matrix(w2v: Word2Vec, itos: List[str], hidden_size: int):
    vocab_size = len(itos)
    embeddings = np.random.normal(scale=0.02, size=(vocab_size, hidden_size)).astype(np.float32)
    for idx, tok in enumerate(itos):
        if tok in w2v.wv:
            vec = w2v.wv[tok]
            if vec.shape[0] != hidden_size:
                vec = vec[:hidden_size] if vec.shape[0] >= hidden_size else np.pad(vec, (0, hidden_size - vec.shape[0]))
            embeddings[idx] = vec
    pad_idx = itos.index(PAD_TOKEN)
    embeddings[pad_idx] = np.zeros(hidden_size, dtype=np.float32)
    return torch.tensor(embeddings)

# -------------------------
# Dataset (supports queries and chunks)
# -------------------------
class BertPretrainingDataset(Dataset):
    def __init__(self, data: List[Tuple[str, str]], stoi: dict, max_seq_len=MAX_SEQ_LEN):
        """
        data: list of tuples [(text, discriminator)], where discriminator ∈ {'Q', 'C'}
        """
        self.stoi = stoi
        self.max_seq_len = max_seq_len
        self.samples = []

        for text, dtype in data:
            if dtype == "Q":
                # Single-sentence query (MLM only)
                self.samples.append((text, dtype, None, None))
            elif dtype == "C":
                # Split chunk into sentences
                sents = [s.strip() for s in text.strip().split('.') if s.strip()]
                if len(sents) < 2:
                    sents = sents + sents  # duplicate if only one sentence
                # Positive pairs: consecutive sentences
                for i in range(len(sents) - 1):
                    self.samples.append((sents[i], "C", sents[i + 1], 1))
                # Negative pairs: non-consecutive
                for i in range(len(sents)):
                    for j in range(len(sents)):
                        if abs(i - j) > 1:  # skip consecutive
                            self.samples.append((sents[i], "C", sents[j], 0))

    def __len__(self):
        return len(self.samples)

    def _tokenize_to_ids(self, text: str) -> List[int]:
        toks = text.strip().split()
        return [self.stoi.get(t, self.stoi[UNK_TOKEN]) for t in toks]

    def __getitem__(self, idx):
        sent_a, dtype, sent_b, nsp_label = self.samples[idx]

        # -------------------------------
        # Case 1: Query (MLM only)
        # -------------------------------
        if dtype == 'Q':
            ids = self._tokenize_to_ids(sent_a)
            ids = ids[:self.max_seq_len - 2]
            input_ids = [self.stoi[CLS_TOKEN]] + ids + [self.stoi[SEP_TOKEN]]
            token_type_ids = [0] * len(input_ids)
            nsp_label = -100  # dummy
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "nsp_label": torch.tensor(nsp_label, dtype=torch.long),
                "batch_type": "Q"
            }

        # -------------------------------
        # Case 2: Chunk (MLM + NSP)
        # -------------------------------
        elif dtype == 'C':
            ids_a = self._tokenize_to_ids(sent_a)
            ids_b = self._tokenize_to_ids(sent_b)
            while len(ids_a) + len(ids_b) > self.max_seq_len - 3:
                if len(ids_a) > len(ids_b):
                    ids_a.pop()
                else:
                    ids_b.pop()
            input_ids = [self.stoi[CLS_TOKEN]] + ids_a + [self.stoi[SEP_TOKEN]] + ids_b + [self.stoi[SEP_TOKEN]]
            token_type_ids = [0] * (len(ids_a) + 2) + [1] * (len(ids_b) + 1)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
                "nsp_label": torch.tensor(nsp_label, dtype=torch.long),
                "batch_type": "C"
            }

def collate_fn(batch, pad_id):
    input_ids_list = [b["input_ids"] for b in batch]
    token_type_list = [b["token_type_ids"] for b in batch]
    nsp_labels = torch.stack([b["nsp_label"] for b in batch]).long()
    batch_types = [b["batch_type"] for b in batch]

    max_len = max([x.size(0) for x in input_ids_list])
    padded_input_ids, padded_token_types, attention_masks = [], [], []
    for ids, tt in zip(input_ids_list, token_type_list):
        pad_len = max_len - ids.size(0)
        padded_input_ids.append(F.pad(ids, (0, pad_len), value=pad_id))
        padded_token_types.append(F.pad(tt, (0, pad_len), value=0))
        attention_masks.append((F.pad(ids, (0, pad_len), value=pad_id) != pad_id).long())

    return {
        "input_ids": torch.stack(padded_input_ids),
        "token_type_ids": torch.stack(padded_token_types),
        "attention_mask": torch.stack(attention_masks),
        "nsp_labels": nsp_labels,
        "batch_type": batch_types
    }

# -------------------------
# MLM Masking
# -------------------------
def create_mlm_labels_and_masked_input(input_ids, pad_id, mask_token_id, vocab_size, mask_prob=MLM_MASK_PROB):
    batch_size, seq_len = input_ids.shape
    mlm_labels = torch.full_like(input_ids, -100)
    prob_matrix = torch.full((batch_size, seq_len), mask_prob, device=input_ids.device)
    prob_matrix[input_ids == pad_id] = 0.0
    special_upper = len(SPECIAL_TOKENS)
    prob_matrix[input_ids < special_upper] = 0.0
    masked_positions = torch.bernoulli(prob_matrix).bool()
    mlm_labels[masked_positions] = input_ids[masked_positions]
    input_ids_masked = input_ids.clone()
    rand_for_replace = torch.rand_like(input_ids, dtype=torch.float, device=input_ids.device)
    mask_replace = masked_positions & (rand_for_replace < 0.8)
    random_replace = masked_positions & (rand_for_replace >= 0.8) & (rand_for_replace < 0.9)
    input_ids_masked[mask_replace] = mask_token_id
    if random_replace.any():
        count = int(random_replace.sum().item())
        rand_tokens = torch.randint(len(SPECIAL_TOKENS), vocab_size, (count,), device=input_ids.device)
        input_ids_masked[random_replace] = rand_tokens
    return input_ids_masked, mlm_labels

# -------------------------
# Mixture-of-Experts Module
# -------------------------
class MoE(nn.Module):
    def __init__(self, hidden_size, ffn_dim, num_experts=5, k=2, noise_std=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std

        # experts: each expert is a small Feed-Forward Network (H -> ffn_dim -> H)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_size)
            ) for _ in range(num_experts)
        ])

        # router: maps hidden vector to expert logits
        self.router = nn.Linear(hidden_size, num_experts)

    def forward(self, x, mask=None):
        """
        x: (B, S, H)
        returns: out (B, S, H), aux_loss (scalar)
        """
        B, S, H = x.size()
        # ---- router logits (noiseless, for load-balancing) ----
        logits = self.router(x)  # (B, S, E)
        # soft probabilities for load balancing (use non-noisy softmax)
        probs_all = F.softmax(logits, dim=-1)  # (B, S, E)
        # importance per expert:
        importance = probs_all.sum(dim=(0, 1))  # (E,)
        total_tokens = float(B * S)
        # aux_loss encourages balanced importance across experts
        aux_loss = (self.num_experts * (importance / total_tokens).pow(2).sum())

        # ---- noisy logits for selection (only add noise during training) ----
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits_noisy = logits + noise
        else:
            logits_noisy = logits

        # top-k selection on noisy logits
        topk_vals, topk_idx = torch.topk(logits_noisy, self.k, dim=-1)  # shapes (B,S,k)
        # convert topk vals to normalized weights via softmax over k
        topk_weights = F.softmax(topk_vals, dim=-1)  # (B,S,k)

        # Compute each expert's output on the full x (inefficient but simple)
        expert_outs = []
        for e in range(self.num_experts):
            expert_outs.append(self.experts[e](x))  # (B,S,H)
        expert_stack = torch.stack(expert_outs, dim=2)  # (B,S,E,H)

        # Build a gating tensor of shape (B,S,E) with nonzero entries only at topk indices
        device = x.device
        gating = torch.zeros(B, S, self.num_experts, device=device, dtype=x.dtype)  # float
        # scatter the topk_weights into gating at positions topk_idx
        # topk_idx: (B,S,k), topk_weights: (B,S,k)
        # We can flatten and scatter
        flat_idx = topk_idx.view(-1, self.k)  # (B*S, k)
        flat_w = topk_weights.view(-1, self.k)  # (B*S, k)
        # For each row r in [0..B*S-1], scatter into gating_flat[r, idx] = weight
        gating_flat = gating.view(-1, self.num_experts)  # (B*S, E)
        rows = torch.arange(gating_flat.size(0), device=device).unsqueeze(1).expand(-1, self.k)  # (B*S, k)
        gating_flat.scatter_(1, flat_idx, flat_w)
        gating = gating_flat.view(B, S, self.num_experts)  # (B,S,E)

        # Combine experts: out[b,s,:] = sum_e gating[b,s,e] * expert_stack[b,s,e,:]
        out = torch.einsum('bse,bseh->bsh', gating, expert_stack)  # (B,S,H)

        return out, aux_loss

# -------------------------
# Transformer encoder
# -------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout=0.1, moe_experts=5, moe_k=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        # Replace ffn with MoE module
        self.ffn_moe = MoE(hidden_size, ffn_dim, num_experts=moe_experts, k=moe_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        key_padding_mask = (mask == 0)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.dropout(attn_out))
        # MoE FFN
        ffn_out, aux_loss = self.ffn_moe(x, mask)
        x = self.ln2(x + self.dropout(ffn_out))
        return x, aux_loss

class BertEncoderModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, ffn_dim=FFN_DIM, max_position_embeddings=512, pad_token_id=0, embedding_weights=None, moe_experts=5, moe_k=2):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        if embedding_weights is not None:
            self.token_embeddings.weight.data.copy_(embedding_weights)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(2, hidden_size)
        self.emb_ln = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(0.1)
        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_size, num_heads, ffn_dim, dropout=DROPOUT, moe_experts=moe_experts, moe_k=moe_k) for _ in range(num_layers)])
        self.nsp_classifier = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Tanh(), nn.Linear(hidden_size, 2))
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

    def encode(self, ids, tt=None, mask=None):
        if tt is None:
            tt = torch.zeros_like(ids)
        if mask is None:
            mask = (ids != self.pad_token_id).long()
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        x = self.token_embeddings(ids) + self.position_embeddings(pos) + self.segment_embeddings(tt)
        x = self.emb_dropout(self.emb_ln(x))
        total_aux = 0.0
        for layer in self.layers:
            x, aux = layer(x, mask)
            total_aux = total_aux + aux
        return x, total_aux

    def forward(self, ids, tt=None, mask=None):
        """
        MASK-AWARE MEAN POOLING (replaces CLS pooling):
        mask: shape (B, S) with 1 for real tokens and 0 for padding.
        We compute mean over token dimension only across valid tokens.
        """
        seq_out, total_aux = self.encode(ids, tt, mask)
        # build mask if not provided
        if mask is None:
            mask = (ids != self.pad_token_id).long()
        # Ensure float and proper shape
        mask_float = mask.unsqueeze(-1).to(seq_out.dtype)  # (B, S, 1)
        # Sum representations over token dimension, masked
        summed = (seq_out * mask_float).sum(dim=1)  # (B, H)
        # denom = number of valid tokens per example (shape (B,1))
        denom = mask_float.sum(dim=1).clamp(min=1e-9)  # avoid div by zero
        pooled = summed / denom  # (B, H)  mask-aware mean pooling
        nsp_logits = self.nsp_classifier(pooled)
        mlm_logits = F.linear(seq_out, self.token_embeddings.weight, self.mlm_bias)
        return mlm_logits, nsp_logits, total_aux

# -------------------------
# Learning Rate Scheduler with Warmup
# -------------------------
def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# -------------------------
# Training and Evaluation
# -------------------------
def main():
    # Load corpus from ChromaDB
    ABSOLUTE_DB_PATH = "../VectorDB/chroma_Data"
    COLLECTION_NAME = "harry_potter_collection"
    
    print(f"Loading ChromaDB collection from {ABSOLUTE_DB_PATH}...")
    chroma_path = Path(ABSOLUTE_DB_PATH).resolve()
    client = chromadb.PersistentClient(path=str(chroma_path))
    print(f"[INFO] ChromaDB client initialized at: {chroma_path}")
    
    # Get the collection
    collection = client.get_collection(name=COLLECTION_NAME)
    print(f"[INFO] Loaded collection '{COLLECTION_NAME}' with {collection.count()} documents")
    
    # Get all documents from the collection
    results = collection.get()
    documents = results['documents']
    print(f"[INFO] Retrieved {len(documents)} documents from collection")
    
    # Create corpus: all documents as chunks ('C' type)
    corpus = [(doc, "C") for doc in documents]
    print(f"[INFO] Created corpus with {len(corpus)} chunks")
    
    stoi, itos = build_vocab([x[0] for x in corpus])
    vocab_size = len(itos)
    print(f"[INFO] Vocab size: {vocab_size}")
    
    print("[INFO] Training Word2Vec...")
    w2v = train_word2vec([x[0] for x in corpus])
    print("[INFO] Building embedding matrix...")
    emb = build_embedding_matrix(w2v, itos, HIDDEN_SIZE)
    # Move embedding weights to device to avoid device mismatch
    emb = emb.to(DEVICE)
    
    pad_id = stoi[PAD_TOKEN]
    mask_id = stoi[MASK_TOKEN]
    
    print("[INFO] Creating dataset...")
    ds = BertPretrainingDataset(corpus, stoi)

    # Split train/test
    total_len = len(ds)
    test_len = max(1, total_len // 5)
    train_len = total_len - test_len
    train_ds, test_ds = random_split(ds, [train_len, test_len])
    print(f"[INFO] Train samples: {train_len}, Test samples: {test_len}")
    
    dl_train = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=lambda b: collate_fn(b, pad_id))
    dl_test = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=lambda b: collate_fn(b, pad_id))

    # instantiate model with MoE: 5 experts, top-k=2
    print(f"[INFO] Initializing model on device: {DEVICE}")
    model = BertEncoderModel(vocab_size, embedding_weights=emb, moe_experts=5, moe_k=2).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Calculate total training steps and create LR scheduler
    total_steps = len(dl_train) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(opt, WARMUP_STEPS, total_steps)
    print(f"[INFO] LR Scheduler: warmup_steps={WARMUP_STEPS}, total_steps={total_steps}")
    
    mlm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    nsp_loss_fct = nn.CrossEntropyLoss()
    
    print(f"[INFO] Starting training for {NUM_EPOCHS} epochs...")
    model.train()
    global_step = 0
    for epoch in range(NUM_EPOCHS):
        print(f"\n[EPOCH {epoch + 1}/{NUM_EPOCHS}]")
        epoch_loss = 0.0
        for batch_idx, batch in enumerate(dl_train):
            ids = batch["input_ids"].to(DEVICE)
            tts = batch["token_type_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            nsp_labels = batch["nsp_labels"].to(DEVICE)
            btypes = batch["batch_type"]
            ids_masked, mlm_labels = create_mlm_labels_and_masked_input(ids, pad_id, mask_id, vocab_size)
            ids_masked, mlm_labels = ids_masked.to(DEVICE), mlm_labels.to(DEVICE)
            mlm_logits, nsp_logits, aux_loss = model(ids_masked, tts, mask)
            mlm_loss = mlm_loss_fct(mlm_logits.view(-1, vocab_size), mlm_labels.view(-1))
            if all(bt == "C" for bt in btypes):
                nsp_loss = nsp_loss_fct(nsp_logits.view(-1, 2), nsp_labels.view(-1))
            else:
                nsp_loss = torch.tensor(0.0, device=DEVICE)
            # auxiliary load-balancing loss scaled down
            aux_coeff = 0.01
            loss = mlm_loss + nsp_loss + aux_coeff * aux_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()  # Update learning rate
            
            epoch_loss += loss.item()
            global_step += 1
            
            if batch_idx % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Batch {batch_idx}: Loss {loss.item():.4f} (MLM {mlm_loss.item():.4f}, NSP {nsp_loss.item():.4f}, AUX {aux_coeff * aux_loss.item():.6f}) LR: {current_lr:.2e}")
        
        avg_epoch_loss = epoch_loss / len(dl_train)
        print(f"[EPOCH {epoch + 1}] Average Loss: {avg_epoch_loss:.4f}")

    # -------------------------
    # Save model and vocab
    # -------------------------
    save_dir = "saved_bert_encoder_moe_pooling"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "bert_encoder_moe_pooling.pt"))
    import json
    with open(os.path.join(save_dir, "vocab.json"), "w") as f:
        json.dump({"stoi": stoi, "itos": itos}, f)
    print(f"[INFO] Model and vocab saved to {save_dir}")

    # -------------------------
    # Evaluation
    # -------------------------
    print("\n[INFO] Starting evaluation...")
    model.eval()
    total_mlm_correct = 0
    total_mlm_count = 0
    total_nsp_correct = 0
    total_nsp_count = 0

    with torch.no_grad():
        for batch in dl_test:
            ids = batch["input_ids"].to(DEVICE)
            tts = batch["token_type_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            nsp_labels = batch["nsp_labels"].to(DEVICE)
            btypes = batch["batch_type"]
            ids_masked, mlm_labels = create_mlm_labels_and_masked_input(ids, pad_id, mask_id, vocab_size)
            ids_masked, mlm_labels = ids_masked.to(DEVICE), mlm_labels.to(DEVICE)
            mlm_logits, nsp_logits, aux_loss = model(ids_masked, tts, mask)
            # MLM accuracy
            mlm_preds = mlm_logits.argmax(-1)
            mask_positions = mlm_labels != -100
            total_mlm_correct += (mlm_preds[mask_positions] == mlm_labels[mask_positions]).sum().item()
            total_mlm_count += mask_positions.sum().item()
            # NSP accuracy
            if all(bt == "C" for bt in btypes):
                nsp_preds = nsp_logits.argmax(-1)
                total_nsp_correct += (nsp_preds == nsp_labels).sum().item()
                total_nsp_count += nsp_labels.numel()

    mlm_acc = total_mlm_correct / max(1, total_mlm_count)
    nsp_acc = total_nsp_correct / max(1, total_nsp_count)
    print(f"\n[RESULTS] MLM Accuracy: {mlm_acc:.4f}, NSP Accuracy: {nsp_acc:.4f}")
    print("[INFO] Training and evaluation complete!")

if __name__ == "__main__":
    main()