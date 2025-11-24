import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import csv
import chromadb
import math
import bitsandbytes as bnb
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple
from dataclasses import dataclass
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# ============================================================
# 1. LOAD YOUR PRETRAINED BERT ENCODER MODEL WITH MoE
#    (This must match your previously saved architecture)
# ============================================================

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
UNK_TOKEN = "[UNK]"

SPECIAL_TOKENS = [PAD_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN, UNK_TOKEN]

# ------------------------
# Mixture of Experts (MoE)
# ------------------------
class MoE(nn.Module):
    def __init__(self, hidden_size, ffn_dim, num_experts=5, k=2, noise_std=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.ffn_dim = ffn_dim
        self.num_experts = num_experts
        self.k = k
        self.noise_std = noise_std
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, hidden_size)
            ) for _ in range(num_experts)
        ])
        
        self.router = nn.Linear(hidden_size, num_experts)
    
    def forward(self, x, mask=None):
        B, S, H = x.size()
        logits = self.router(x)
        probs_all = F.softmax(logits, dim=-1)
        importance = probs_all.sum(dim=(0, 1))
        total_tokens = float(B * S)
        aux_loss = (self.num_experts * (importance / total_tokens).pow(2).sum())
        
        if self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits_noisy = logits + noise
        else:
            logits_noisy = logits
        
        topk_vals, topk_idx = torch.topk(logits_noisy, self.k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)
        
        expert_outs = []
        for e in range(self.num_experts):
            expert_outs.append(self.experts[e](x))
        expert_stack = torch.stack(expert_outs, dim=2)
        
        device = x.device
        gating = torch.zeros(B, S, self.num_experts, device=device, dtype=x.dtype)
        flat_idx = topk_idx.view(-1, self.k)
        flat_w = topk_weights.view(-1, self.k)
        gating_flat = gating.view(-1, self.num_experts)
        rows = torch.arange(gating_flat.size(0), device=device).unsqueeze(1).expand(-1, self.k)
        gating_flat.scatter_(1, flat_idx, flat_w)
        gating = gating_flat.view(B, S, self.num_experts)
        
        out = torch.einsum('bse,bseh->bsh', gating, expert_stack)
        return out, aux_loss

# ------------------------
# Transformer Layer with MoE
# ------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ffn_dim, dropout=0.1, moe_experts=5, moe_k=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads,
                                               dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn_moe = MoE(hidden_size, ffn_dim, num_experts=moe_experts, k=moe_k)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Ensure mask is boolean and inputs are floating-point to avoid dtype issues
        key_padding_mask = (mask == 0)
        if key_padding_mask.dtype != torch.bool:
            try:
                key_padding_mask = key_padding_mask.bool()
            except Exception:
                key_padding_mask = (mask == 0)

        if not x.is_floating_point():
            x_attn = x.float()
        else:
            x_attn = x

        attn_out, _ = self.self_attn(x_attn, x_attn, x_attn, key_padding_mask=key_padding_mask)
        x = self.ln1(x + self.dropout(attn_out))
        ffn_out, aux_loss = self.ffn_moe(x, mask)
        x = self.ln2(x + self.dropout(ffn_out))
        return x, aux_loss

# ------------------------
# Base BERT Encoder with MoE
# ------------------------
class BertEncoderModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=768,
                 num_layers=12, num_heads=12, ffn_dim=3072,
                 max_position_embeddings=512, pad_token_id=0,
                 moe_experts=5, moe_k=2, embedding_weights=None):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.hidden_size = hidden_size

        self.token_embeddings = nn.Embedding(vocab_size, hidden_size,
                                             padding_idx=pad_token_id)
        if embedding_weights is not None:
            self.token_embeddings.weight.data.copy_(embedding_weights)

        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.segment_embeddings = nn.Embedding(2, hidden_size)

        self.emb_ln = nn.LayerNorm(hidden_size)
        self.emb_dropout = nn.Dropout(0.1)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_heads, ffn_dim, dropout=0.1,
                                   moe_experts=moe_experts, moe_k=moe_k)
            for _ in range(num_layers)
        ])
        
        # These are in the saved model but not used for fine-tuning
        self.nsp_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2)
        )
        self.mlm_bias = nn.Parameter(torch.zeros(vocab_size))

    def encode(self, ids, tt=None, mask=None):
        if tt is None:
            tt = torch.zeros_like(ids)
        if mask is None:
            mask = (ids != self.pad_token_id).long()

        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        x = (self.token_embeddings(ids) +
             self.position_embeddings(pos) +
             self.segment_embeddings(tt))

        x = self.emb_dropout(self.emb_ln(x))

        total_aux = 0.0
        for layer in self.layers:
            x, aux_loss = layer(x, mask)
            total_aux = total_aux + aux_loss

        return x[:, 0]   # CLS embedding

# ============================================================
# 2. APPLY LoRA TO THE BERT ATTENTION MODULES (QLoRA, bitsandbytes NF4)
# ============================================================

class LoRALinear(nn.Module):
    """A LoRA wrapper for linear layers. Works with a frozen base layer (quantized or otherwise)."""
    def __init__(self, base_layer, r=8, alpha=8, dropout=0.0):
        super().__init__()
        self.base = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        in_features = getattr(base_layer, "in_features", None)
        out_features = getattr(base_layer, "out_features", None)
        if in_features is None or out_features is None:
            # try infer from weight if available
            if hasattr(base_layer, "weight"):
                in_features = base_layer.weight.shape[1]
                out_features = base_layer.weight.shape[0]
            else:
                raise ValueError("Base layer must expose in_features and out_features")

        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        self.dropout = nn.Dropout(dropout)

        self.train_lora_only()

    # Provide attribute accessors so wrapped modules behave like nn.Linear
    @property
    def weight(self):
        return getattr(self.base, "weight", None)

    @property
    def bias(self):
        return getattr(self.base, "bias", None)

    @property
    def in_features(self):
        return getattr(self.base, "in_features", None) or getattr(self.lora_A, "in_features", None)

    @property
    def out_features(self):
        return getattr(self.base, "out_features", None) or getattr(self.lora_B, "out_features", None)

    def train_lora_only(self):
        """Freeze base layer, train LoRA params only."""
        # Freeze any parameters on the base (bnb modules have parameters that must be frozen)
        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.lora_A.parameters():
            p.requires_grad = True
        for p in self.lora_B.parameters():
            p.requires_grad = True

    def forward(self, x):
        # Ensure input is floating and match dtypes between base and LoRA paths.
        # Some upstream tensors (e.g., masks or ids) can be integer/byte — cast safely.
        if not x.is_floating_point():
            x_float = x.float()
        else:
            x_float = x

        # Determine base dtype (if available) to run quantized bases correctly
        base_dtype = None
        try:
            if hasattr(self.base, "weight") and self.base.weight is not None:
                base_dtype = self.base.weight.dtype
        except Exception:
            base_dtype = None

        # Run base module in its preferred dtype
        if base_dtype is not None and base_dtype != x_float.dtype:
            base_in = x_float.to(base_dtype)
        else:
            base_in = x_float

        base_out = self.base(base_in)

        # Run LoRA adapters in float32 for stability, then cast to base dtype
        lora_in = x_float
        lora_out = self.dropout(self.lora_B(self.lora_A(lora_in))) * self.scaling

        if base_out.dtype != lora_out.dtype:
            try:
                lora_out = lora_out.to(base_out.dtype)
            except Exception:
                lora_out = lora_out.type_as(base_out)

        return base_out + lora_out

def apply_lora(model, r=8, alpha=8):
    """Wraps all MHA Q,K,V projection layers with LoRA using bitsandbytes 4-bit NF4 base layers."""
    for layer in model.layers:
        attn = layer.self_attn

        # If the MHA exposes in_proj_weight/in_proj_bias, keep them as frozen parameters (as before)
        if hasattr(attn, "in_proj_weight") and getattr(attn, "in_proj_weight") is not None:
            try:
                attn.in_proj_weight = nn.Parameter(attn.in_proj_weight, requires_grad=False)
            except Exception:
                pass
        if hasattr(attn, "in_proj_bias") and getattr(attn, "in_proj_bias") is not None:
            try:
                attn.in_proj_bias = nn.Parameter(attn.in_proj_bias, requires_grad=False)
            except Exception:
                pass

        hidden = attn.embed_dim

        # Create 4-bit quantized projection layers (bitsandbytes NF4)
        # compute_dtype set to float16 for faster computation on GPU
        q_proj = bnb.nn.Linear4bit(hidden, hidden, bias=True, compute_dtype=torch.float16, quant_type="nf4")
        k_proj = bnb.nn.Linear4bit(hidden, hidden, bias=True, compute_dtype=torch.float16, quant_type="nf4")
        v_proj = bnb.nn.Linear4bit(hidden, hidden, bias=True, compute_dtype=torch.float16, quant_type="nf4")

        # If in_proj_weight exists we can try to copy float weights into these bnb layers where possible.
        # bnb's Linear4bit exposes a `.weight` buffer or parameter depending on version; attempt best-effort copy.
        try:
            if hasattr(attn, "in_proj_weight") and attn.in_proj_weight is not None:
                w = attn.in_proj_weight.detach().clone()
                # in_proj_weight layout: (3*hidden, hidden)
                q_w, k_w, v_w = w.chunk(3, dim=0)
                # Attempt to set float data before quantization if bnb supports set_state_dict style assignment
                try:
                    # Many versions of bnb allow direct access to .weight (float) for copying before quantization
                    q_proj.weight.data.copy_(q_w.to(q_proj.weight.data.dtype))
                    k_proj.weight.data.copy_(k_w.to(k_proj.weight.data.dtype))
                    v_proj.weight.data.copy_(v_w.to(v_proj.weight.data.dtype))
                except Exception:
                    # If direct copy not available, ignore — the modules will remain initialized randomly but frozen.
                    pass
            if hasattr(attn, "in_proj_bias") and attn.in_proj_bias is not None:
                b = attn.in_proj_bias.detach().clone()
                try:
                    qb, kb, vb = b.chunk(3, dim=0)
                    if hasattr(q_proj, "bias") and q_proj.bias is not None:
                        q_proj.bias.data.copy_(qb.to(q_proj.bias.data.dtype))
                        k_proj.bias.data.copy_(kb.to(k_proj.bias.data.dtype))
                        v_proj.bias.data.copy_(vb.to(v_proj.bias.data.dtype))
                except Exception:
                    pass
        except Exception:
            pass

        # Attach references (for debugging or inspection)
        attn.q_proj_weight = getattr(q_proj, "weight", None)
        attn.k_proj_weight = getattr(k_proj, "weight", None)
        attn.v_proj_weight = getattr(v_proj, "weight", None)

        # Wrap quantized bases with LoRA adapters
        attn.q_proj = LoRALinear(q_proj, r=r, alpha=alpha)
        attn.k_proj = LoRALinear(k_proj, r=r, alpha=alpha)
        attn.v_proj = LoRALinear(v_proj, r=r, alpha=alpha)

    return model


# ------------------------
# QLoRA helpers: quantize & wrap linears across the model
# ------------------------
def replace_linears_with_4bit_and_lora(module, r=8, alpha=8, device=DEVICE):
    """Recursively replace nn.Linear children with bitsandbytes 4-bit + LoRA wrapper.
    If bitsandbytes replacement fails, fallback to wrapping the original Linear with LoRA.
    """
    for name, child in list(module.named_children()):
        # If this module is a MultiheadAttention, do NOT replace its internal linears
        # (e.g., out_proj) because `multi_head_attention_forward` expects real float
        # weight tensors for functional calls. We handle attention LoRA separately
        # in `apply_lora`.
        if isinstance(module, nn.MultiheadAttention):
            return

        # skip embeddings
        if isinstance(child, nn.Embedding):
            continue


        if isinstance(child, nn.Linear):
            try:
                # create a 4-bit bnb linear and copy weights (best-effort)
                if hasattr(bnb, 'nn') and hasattr(bnb.nn, 'Linear4bit'):
                    new_lin = bnb.nn.Linear4bit(child.in_features, child.out_features,
                                                 bias=(child.bias is not None),
                                                 compute_dtype=torch.float16,
                                                 quant_type='nf4')
                    try:
                        # attempt to copy weights into new module
                        if hasattr(new_lin, 'weight') and child.weight is not None:
                            new_lin.weight.data.copy_(child.weight.data.to(new_lin.weight.data.dtype).to(device))
                        if child.bias is not None and hasattr(new_lin, 'bias') and new_lin.bias is not None:
                            new_lin.bias.data.copy_(child.bias.data.to(new_lin.bias.data.dtype).to(device))
                    except Exception:
                        pass
                    base = new_lin
                else:
                    base = child

                # wrap base with LoRA adapter
                wrapped = LoRALinear(base, r=r, alpha=alpha)
                setattr(module, name, wrapped)
            except Exception as e:
                print(f"[WARN] Failed to replace Linear {name} with 4-bit: {e}; falling back to LoRA on original")
                setattr(module, name, LoRALinear(child, r=r, alpha=alpha))
        else:
            replace_linears_with_4bit_and_lora(child, r=r, alpha=alpha, device=device)


# ============================================================
# 3. CHROMADB LOADING
# ============================================================
client = chromadb.PersistentClient(path="../VectorDB/chroma_Data_with_BERT_embeddings")
collection = client.get_collection("HP_Chunks_BERT_Embeddings_collection")

# ============================================================
# 4. TRAINING DATASET (CSV WITH query, chunk_ID)
# ============================================================

@dataclass
class QueryChunkPair:
    query: str
    positive_id: str

class ContrastiveDataset(Dataset):
    def __init__(self, csv_path):
        self.pairs: List[QueryChunkPair] = []
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self.pairs.append(QueryChunkPair(query=row[0], positive_id=row[1]))

        all_chunks = collection.get(
            where={"ischunk": True}
        )
        self.chunk_ids = all_chunks["ids"]

    def __len__(self):
        return len(self.pairs)

    def sample_negatives(self, positive_id: str, k=5):
        negs = set()
        while len(negs) < k:
            cid = random.choice(self.chunk_ids)
            if cid != positive_id:
                negs.add(cid)
        return list(negs)


    def __getitem__(self, idx):
        item = self.pairs[idx]

        pos_chunk = collection.get(ids=[item.positive_id])
        neg_ids = self.sample_negatives(item.positive_id, k=5)
        neg_chunks = collection.get(ids=neg_ids)

        return {
            "query": item.query,
            "positive_text": pos_chunk["documents"][0],
            "negative_texts": neg_chunks["documents"]
        }

# ============================================================
# 5. TOKENIZER (simple whitespace tokenizer)
# ============================================================

def tokenize(text, stoi, max_len=64):
    tokens = text.strip().split()

    # truncate
    ids = [stoi.get(t, stoi[UNK_TOKEN]) for t in tokens][: max_len-2]

    # add CLS and SEP
    ids = [stoi[CLS_TOKEN]] + ids + [stoi[SEP_TOKEN]]

    # CASE 1: longer than max_len → truncate + NO padding needed
    if len(ids) > max_len:
        ids = ids[:max_len]
        pad_amount = 0
    else:
        # CASE 2: shorter → pad
        pad_id = stoi[PAD_TOKEN]
        pad_amount = max_len - len(ids)
        ids = ids + [pad_id] * pad_amount

    tt = [0] * max_len
    mask = [1] * (max_len - pad_amount) + [0] * pad_amount

    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(tt, dtype=torch.long),
        torch.tensor(mask, dtype=torch.long)
    )

def collate(batch):
    queries, q_tt, q_mask = [], [], []
    pos, pt_tt, pt_mask = [], [], []
    negs, nt_tt, nt_mask = [], [], []

    MAX_LEN = 128  # <<<<< HARD CAP

    for b in batch:
        q_ids, qtt, qmask = tokenize(b["query"], stoi, MAX_LEN)
        p_ids, ptt, pmask = tokenize(b["positive_text"], stoi, MAX_LEN)

        neg_ids_list = []
        neg_tt_list = []
        neg_mask_list = []

        for nt in b["negative_texts"]:
            ids, tti, msk = tokenize(nt, stoi, MAX_LEN)
            neg_ids_list.append(ids)
            neg_tt_list.append(tti)
            neg_mask_list.append(msk)

        queries.append(q_ids)
        q_tt.append(qtt)
        q_mask.append(qmask)

        pos.append(p_ids)
        pt_tt.append(ptt)
        pt_mask.append(pmask)

        negs.append(torch.stack(neg_ids_list))
        nt_tt.append(torch.stack(neg_tt_list))
        nt_mask.append(torch.stack(neg_mask_list))

    return {
        "queries": torch.stack(queries),
        "queries_tt": torch.stack(q_tt),
        "queries_mask": torch.stack(q_mask),
        "pos": torch.stack(pos),
        "pos_tt": torch.stack(pt_tt),
        "pos_mask": torch.stack(pt_mask),
        "neg": torch.stack(negs),
        "neg_tt": torch.stack(nt_tt),
        "neg_mask": torch.stack(nt_mask),
    }

# ============================================================
# 6. CONTRASTIVE LOSS (softmax over exp(cos()))
# ============================================================

def contrastive_loss(q, pos, negs):
    """
    q: (B, H)
    pos: (B, H)
    negs: (B, 5, H)
    """

    pos_sim = F.cosine_similarity(q, pos)      # (B,)
    neg_sim = F.cosine_similarity(
        q.unsqueeze(1).repeat(1, negs.size(1), 1),
        negs,
        dim=-1
    )  # (B, 5)

    sims = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (B, 6)
    exp_sims = torch.exp(sims)
    probs = exp_sims / exp_sims.sum(dim=1, keepdim=True)

    loss = -torch.log(probs[:, 0]).mean()
    return loss

# ============================================================
# 7. MAIN TRAINING LOOP
# ============================================================

def train_lora(
    model_path="../Encoder/saved_bert_encoder_moe_pooling/bert_encoder_moe_pooling.pt",
    vocab_path="../Encoder/saved_bert_encoder_moe_pooling/vocab.json",
    csv_path="../LLM Caller/generated_pairs_without_commas.csv",
    batch_size=1,
    lr=1e-4,
    epochs=5,
):

    import json
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    global stoi, itos
    stoi, itos = vocab["stoi"], vocab["itos"]
    vocab_size = len(itos)

    # Create model with MoE architecture matching the saved model
    model = BertEncoderModel(vocab_size, max_position_embeddings=512, 
                            moe_experts=5, moe_k=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Convert heavy linear layers to 4-bit where possible and wrap with LoRA adapters
    print("[INFO] Replacing Linear layers with 4-bit + LoRA wrappers (QLoRA)...")
    replace_linears_with_4bit_and_lora(model, r=8, alpha=8, device=DEVICE)

    # Also apply attention-specific LoRA projections (q/k/v) using bitsandbytes
    apply_lora(model)

    model.to(DEVICE)

    ds = ContrastiveDataset(csv_path)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate)

    # Prefer bitsandbytes 8-bit optimizer when available
    try:
        from bitsandbytes import optim as bnb_optim
        opt = bnb_optim.AdamW8bit(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        print("[INFO] Using bitsandbytes AdamW8bit optimizer")
    except Exception:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        print("[WARN] bitsandbytes 8-bit optimizer not available; using torch AdamW")

    # Use AMP scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    model.train()
    for epoch in range(epochs):
        for batch in dl:
            q = batch["queries"].to(DEVICE)
            q_tt = batch["queries_tt"].to(DEVICE)
            q_mask = batch["queries_mask"].to(DEVICE)

            p = batch["pos"].to(DEVICE)
            p_tt = batch["pos_tt"].to(DEVICE)
            p_mask = batch["pos_mask"].to(DEVICE)

            n = batch["neg"].to(DEVICE)
            n_tt = batch["neg_tt"].to(DEVICE)
            n_mask = batch["neg_mask"].to(DEVICE)

            # Forward with AMP
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    q_emb = model.encode(q, q_tt, q_mask)
                    p_emb = model.encode(p, p_tt, p_mask)

                    B, K, L = n.size()
                    n = n.view(B*K, L)
                    n_tt = n_tt.view(B*K, L)
                    n_mask = n_mask.view(B*K, L)

                    # micro-batch to prevent OOM
                    chunks = []
                    MB = 1
                    for i in range(0, B*K, MB):
                        part = n[i:i+MB]
                        part_tt = n_tt[i:i+MB]
                        part_mask = n_mask[i:i+MB]
                        chunks.append(model.encode(part, part_tt, part_mask))
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.ipc_collect()
                        del part, part_tt, part_mask

                    n_emb = torch.cat(chunks, dim=0).view(B, K, -1)
                    loss = contrastive_loss(q_emb, p_emb, n_emb)

                opt.zero_grad()
                scaler.scale(loss).backward()
                try:
                    scaler.step(opt)
                except Exception:
                    opt.step()
                scaler.update()
            else:
                q_emb = model.encode(q, q_tt, q_mask)
                p_emb = model.encode(p, p_tt, p_mask)

                B, K, L = n.size()
                n = n.view(B*K, L)
                n_tt = n_tt.view(B*K, L)
                n_mask = n_mask.view(B*K, L)

                chunks = []
                MB = 1
                for i in range(0, B*K, MB):
                    part = n[i:i+MB]
                    part_tt = n_tt[i:i+MB]
                    part_mask = n_mask[i:i+MB]
                    chunks.append(model.encode(part, part_tt, part_mask))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                    del part, part_tt, part_mask

                n_emb = torch.cat(chunks, dim=0).view(B, K, -1)
                loss = contrastive_loss(q_emb, p_emb, n_emb)

                opt.zero_grad()
                loss.backward()
                opt.step()

            print(f"Epoch {epoch} Loss {loss.item():.4f}")

    os.makedirs("lora_finetuned", exist_ok=True)
    torch.save(model.state_dict(), "lora_finetuned/lora_bert.pt")
    print("LoRA fine-tuned model saved.")

# ============================================================
# RUN TRAINING
# ============================================================

if __name__ == "__main__":
    train_lora()