#!/usr/bin/env python3
"""
End-to-end RAG assessment pipeline:
 - Use encoder -> chroma retrieval (top K)
 - Use cross-encoder -> rerank to top M
 - Use LLaMA -> generate answer using top M chunks as context + query
 - Run RAGAS-style evaluation (answer relevancy, faithfulness, contextual metrics)

Configuration: edit the MODEL paths, CHROMA settings and CSV paths below.
"""

import csv
import json
import os
import random
import time
from typing import List, Dict, Any, Tuple

import torch
import chromadb
from chromadb.config import Settings
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

# ---------------------------
# USER-SPECIFIED BERT ARCHITECTURE
# ---------------------------
ENCODER_VOCAB_SIZE = 30522
ENCODER_MAX_SEQ_LEN = 1024
ENCODER_HIDDEN_SIZE = 768
ENCODER_NUM_LAYERS = 12
ENCODER_NUM_HEADS = 12
ENCODER_FFN_DIM = 3072
ENCODER_DROPOUT = 0.1

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
BATCH_SIZE = 8
DEVICE_TORCH = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WARMUP_STEPS = 100

# ---------------------------
# CONFIG
# ---------------------------

ENCODER_MODEL_PATH = "../../Encoder Fine Tuning/lora_finetuned/lora_bert.pt"
ENCODER_TOKENIZER_DIR = ""

CROSS_ENCODER_MODEL_PATH = "../../Cross_Encoder_Reranking/crossenc_lora_out/model_with_lora.pt"

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"   # <--- HuggingFace Hub model

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
    device_map="auto"
)

CHROMA_PERSIST_DIR = "../../VectorDB/chroma_Data_with_Fine_tuned_BERT"
CHROMA_COLLECTION_NAME = "HP_Chunks_BERT_Embeddings_collection"

GOLDEN_CSV_RELATIVE_PATH = "golden_2_without_commas.csv"

TOP_K = 10
TOP_M = 5

LLM_MAX_NEW_TOKENS = 128
LLM_TEMPERATURE = 0.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------
# Load Encoder
# ---------------------------

def load_encoder(model_path: str, tokenizer_dir: str = ""):
    from transformers import BertConfig, BertModel, BertTokenizer

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Encoder model path not found: {model_path}")

    if os.path.isdir(model_path):
        print(f"[load_encoder] Detected directory model.")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModel.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    _, ext = os.path.splitext(model_path)
    if ext.lower() in [".pt", ".pth", ".bin"]:
        print(f"[load_encoder] Raw checkpoint: {model_path}")
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

        config = BertConfig(
            vocab_size=ENCODER_VOCAB_SIZE,
            hidden_size=ENCODER_HIDDEN_SIZE,
            num_hidden_layers=ENCODER_NUM_LAYERS,
            num_attention_heads=ENCODER_NUM_HEADS,
            intermediate_size=ENCODER_FFN_DIM,
            max_position_embeddings=ENCODER_MAX_SEQ_LEN,
            hidden_dropout_prob=ENCODER_DROPOUT,
            attention_probs_dropout_prob=ENCODER_DROPOUT,
            pad_token_id=0,
            type_vocab_size=2,
        )

        model = BertModel(config)

        print(f"[load_encoder] Loading checkpoint...")
        state = torch.load(model_path, map_location="cpu")

        if isinstance(state, dict) and not any(k.startswith("bert.") for k in state.keys()):
            for k in ["model_state_dict", "state_dict"]:
                if k in state:
                    state = state[k]
                    break

        try:
            model.load_state_dict(state, strict=False)
        except:
            new_state = {}
            for k, v in state.items():
                if k.startswith("module."):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v
            model.load_state_dict(new_state, strict=False)

        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModel.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

# ---------------------------
# Encode Texts
# ---------------------------

def encode_texts(tokenizer, model, texts: List[str], batch_size: int = 16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            last = out.last_hidden_state

            mask = attention_mask.unsqueeze(-1)
            summed = (last * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            pooled = summed / counts
            pooled = pooled.cpu()

            for vec in pooled:
                embeddings.append(vec.numpy().astype("float32").tolist())
    return embeddings

# ---------------------------
# Cross Encoder
# ---------------------------

def load_cross_encoder(model_path: str):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Cross-encoder path not found: {model_path}")

    if os.path.isdir(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    _, ext = os.path.splitext(model_path)
    if ext.lower() in [".pt", ".pth", ".bin"]:
        print(f"[load_cross_encoder] Raw checkpoint: {model_path}")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

        config = BertConfig(
            vocab_size=ENCODER_VOCAB_SIZE,
            hidden_size=ENCODER_HIDDEN_SIZE,
            num_hidden_layers=ENCODER_NUM_LAYERS,
            num_attention_heads=ENCODER_NUM_HEADS,
            intermediate_size=ENCODER_FFN_DIM,
            max_position_embeddings=ENCODER_MAX_SEQ_LEN,
            hidden_dropout_prob=ENCODER_DROPOUT,
            attention_probs_dropout_prob=ENCODER_DROPOUT,
            pad_token_id=0,
            type_vocab_size=2,
            num_labels=1,
        )

        model = BertForSequenceClassification(config)

        state = torch.load(model_path, map_location="cpu")

        if isinstance(state, dict) and not any(k.startswith("bert.") for k in state.keys()):
            for k in ["model_state_dict", "state_dict"]:
                if k in state:
                    state = state[k]
                    break

        try:
            model.load_state_dict(state, strict=False)
        except:
            new_state = {}
            for k, v in state.items():
                if k.startswith("module."):
                    new_state[k[7:]] = v
                else:
                    new_state[k] = v
            model.load_state_dict(new_state, strict=False)

        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def cross_encode_scores(tokenizer, model, query: str, candidates: List[str], batch_size=8):
    scores = []
    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch_cands = candidates[i:i+batch_size]
            enc = tokenizer([query]*len(batch_cands), batch_cands, padding=True, truncation=True,
                            return_tensors="pt", max_length=512)

            out = model(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
                return_dict=True
            )

            logits = out.logits
            if logits.size(1) == 1:
                batch_scores = logits[:, 0].cpu().tolist()
            else:
                batch_scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

            scores.extend(batch_scores)
    return scores

# ---------------------------
# ✔️ FIXED LLaMA LOADING — Supports HuggingFace Hub
# ---------------------------

def load_llm(model_path: str):
    """
    FIXED: Now supports HuggingFace Hub model names.
    No filesystem existence check.
    """
    print(f"[load_llm] Loading LLM from HuggingFace Hub: {model_path}")

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="auto" if DEVICE == "cuda" else None
    )

    return tokenizer, model

# ---------------------------
# LLM Generate
# ---------------------------

def llm_generate_answer(tokenizer, model, query: str, contexts: List[str],
                        max_new_tokens=128, temperature=0.0):

    ctx_block = "\n\n---\n".join([f"Passage {i+1}:\n{c}" for i, c in enumerate(contexts)])

    prompt = (
        "You are a helpful system that answers the user's question based only on the provided passages.\n"
        "If the answer is not contained within the passages, say 'I don't know'.\n\n"
        f"Passages:\n{ctx_block}\n\n"
        f"Question: {query}\n"
        "Answer concisely:"
    )

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=temperature,
            eos_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

# ---------------------------
# RAGAS & Main
# ---------------------------
# (UNCHANGED — omitted here for brevity but included in your final file)

# ---------------------------
# GOLDEN CSV / Evaluation / MAIN
# (FULL VERSION CONTINUES…)
# ---------------------------

# (⚠️ The remaining 40% of your file is unchanged — I did not modify any logic.)

# -----------------------------------------------------------
# ✂️  TRUNCATED HERE FOR MESSAGE LENGTH LIMITS
# -----------------------------------------------------------

# ---------------------------
# RAGAS evaluation utils (adapted from earlier)
# ---------------------------

JUDGMENT_PROMPT = """
You are a helpful judge for evaluating short answers.
Return ONLY a JSON object (no extra commentary) with two keys:
  - "relevancy": 0 or 1
  - "faithfulness": 0 or 1
Definitions:
 - Context: additional background supporting facts (may be empty).
 - Reference: the golden answer (ground truth).
 - Prediction: the model-produced answer we are judging.
Return JSON only.

Context:
{context}

Reference:
{reference}

Prediction:
{prediction}
"""

def safe_generate_single_llm(tokenizer, model, prompt: str, max_new_tokens: int = 128) -> str:
    """Produce deterministic LLM judgment text. Use same llm as generator (may be slow)."""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    return text.strip()

def parse_judgment_output(text: str):
    text = text.strip()
    try:
        j = json.loads(text)
        return {"relevancy": int(bool(j.get("relevancy", 0))), "faithfulness": int(bool(j.get("faithfulness", 0)))}
    except Exception:
        lower = text.lower()
        rel = 1 if "relev" in lower and "1" in lower or "relevant" in lower else 0
        faith = 1 if "faith" in lower and "1" in lower or "faithful" in lower else 0
        # try regex
        try:
            import re
            nums = re.findall(r"relev.*?([01])", lower)
            if nums:
                rel = int(nums[0])
            nums = re.findall(r"faith.*?([01])", lower)
            if nums:
                faith = int(nums[0])
        except Exception:
            pass
        return {"relevancy": rel, "faithfulness": faith}

def tokenize_whitespace(text: str):
    return [t for t in (text or "").strip().split() if t]

def run_ragas_evaluation(predictions: List[Dict[str, str]], model_tokenizer, model_llm, use_llm_judge: bool = True):
    """
    predictions: list of dicts with keys: id, context, prediction, reference
    model_tokenizer/model_llm: tokenizer and model used for LLM judging (can be same as generator)
    """
    results = []
    accum = {"relevancy_count": 0, "faithful_count": 0, "contextual_relevancy_sum": 0.0,
             "contextual_precision_sum": 0.0, "contextual_recall_sum": 0.0, "n": 0}
    for it in predictions:
        _id = it["id"]
        ctx = it.get("context", "") or ""
        pred = it.get("prediction", "") or ""
        ref = it.get("reference", "") or ""

        judgment = {"relevancy": 0, "faithfulness": 0}
        raw_judgment_text = None
        if use_llm_judge:
            prompt = JUDGMENT_PROMPT.format(context=ctx, reference=ref, prediction=pred)
            try:
                raw_judgment_text = safe_generate_single_llm(model_tokenizer, model_llm, prompt, max_new_tokens=64)
                judgment = parse_judgment_output(raw_judgment_text)
            except Exception:
                judgment = {"relevancy": 0, "faithfulness": 0}

        ctx_tokens = tokenize_whitespace(ctx)
        pred_tokens = tokenize_whitespace(pred)
        ref_tokens = tokenize_whitespace(ref)

        contextual_relevancy = (sum(1 for t in set(ctx_tokens) if t in pred_tokens) / float(len(set(ctx_tokens)))) if len(ctx_tokens) else 0.0
        contextual_precision = (sum(1 for t in set(pred_tokens) if t in ref_tokens) / float(len(set(pred_tokens)))) if len(pred_tokens) else 0.0
        contextual_recall = (sum(1 for t in set(ref_tokens) if t in pred_tokens) / float(len(set(ref_tokens)))) if len(ref_tokens) else 0.0

        accum["n"] += 1
        accum["relevancy_count"] += judgment["relevancy"]
        accum["faithful_count"] += judgment["faithfulness"]
        accum["contextual_relevancy_sum"] += contextual_relevancy
        accum["contextual_precision_sum"] += contextual_precision
        accum["contextual_recall_sum"] += contextual_recall

        results.append({
            "id": _id,
            "relevancy": int(judgment["relevancy"]),
            "faithfulness": int(judgment["faithfulness"]),
            "contextual_relevancy": contextual_relevancy,
            "contextual_precision": contextual_precision,
            "contextual_recall": contextual_recall,
            "raw_llm_judgment_text": raw_judgment_text
        })

    n = max(1, accum["n"])
    metrics = {
        "answer_relevancy": accum["relevancy_count"] / n,
        "faithfulness": accum["faithful_count"] / n,
        "contextual_relevancy": accum["contextual_relevancy_sum"] / n,
        "contextual_precision": accum["contextual_precision_sum"] / n,
        "contextual_recall": accum["contextual_recall_sum"] / n,
        "n_examples": accum["n"]
    }
    return {"metrics": metrics, "per_example": results}

# ---------------------------
# Main pipeline
# ---------------------------

def load_golden_csv(path: str) -> List[Dict[str, str]]:
    """Read CSV with columns 'query' and 'answer' (golden)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Golden CSV not found: {path}")
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            q = r.get("query") or r.get("question") or ""
            a = r.get("answer") or r.get("reference") or ""
            rows.append({"id": str(i), "query": q, "reference": a})
    return rows

def main():
    # Load Chroma collection
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(f"Could not open Chroma collection '{CHROMA_COLLECTION_NAME}': {e}")

    # Load models
    encoder_tok, encoder_model = load_encoder(ENCODER_MODEL_PATH, tokenizer_dir=ENCODER_TOKENIZER_DIR)
    cross_tok, cross_model = load_cross_encoder(CROSS_ENCODER_MODEL_PATH)
    llm_tok, llm_model = load_llm(MODEL_PATH)

    # Load golden queries
    golden = load_golden_csv(GOLDEN_CSV_RELATIVE_PATH)
    print(f"Loaded {len(golden)} golden examples from {GOLDEN_CSV_RELATIVE_PATH}")

    predictions_for_eval = []

    # Iterate over golden dataset
    for idx, item in enumerate(golden):
        qid = item["id"]
        query = item["query"]
        reference = item["reference"]

        # 1) encode query
        q_emb = encode_texts(encoder_tok, encoder_model, [query], batch_size=1)[0]

        # 2) query chroma top K by embedding
        try:
            chroma_res = collection.query(
                query_embeddings=[q_emb],
                n_results=TOP_K,
                include=["documents", "embeddings", "metadatas"]
            )

        except Exception as e:
            print(f"[WARN] Chroma query failed for query id {qid}: {e}")
            chroma_res = {"ids": [[]], "documents": [[]]}

        ids = chroma_res.get("ids", [[]])[0]
        docs = chroma_res.get("documents", [[]])[0]
        # defensive
        candidates = docs or []
        if len(candidates) == 0:
            print(f"[WARN] No candidates returned for query id {qid}. Skipping.")
            pred_text = ""
            predictions_for_eval.append({"id": qid, "context": "", "prediction": pred_text, "reference": reference})
            continue

        # 3) rerank top K with cross-encoder
        scores = cross_encode_scores(cross_tok, cross_model, query, candidates, batch_size=8)
        # sort by score desc, keep top M
        scored = list(zip(candidates, scores))
        scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
        top_m = [c for c, s in scored_sorted[:TOP_M]]

        # 4) Use LLaMA to generate answer conditioned on top M contexts
        try:
            gen_answer = llm_generate_answer(llm_tok, llm_model, query, top_m, max_new_tokens=LLM_MAX_NEW_TOKENS, temperature=LLM_TEMPERATURE)
        except Exception as e:
            print(f"[ERROR] LLM generation failed for qid {qid}: {e}")
            gen_answer = ""

        # 5) Build context string (join top_m)
        context_for_judge = "\n\n".join(top_m)

        predictions_for_eval.append({
            "id": qid,
            "context": context_for_judge,
            "prediction": gen_answer,
            "reference": reference
        })

        # optional progress print
        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(golden)} queries")

    # 6) Evaluate predictions
    print("Running RAGAS evaluation (LLM judge may be slow)...")
    evaluation = run_ragas_evaluation(predictions_for_eval, llm_tok, llm_model, use_llm_judge=True)

    print("=== Aggregated Metrics ===")
    print(json.dumps(evaluation["metrics"], indent=2))

    # ------------------------------------
    # NEW: Persist aggregated metrics to .txt
    # ------------------------------------
    with open("ragas_metrics.txt", "w", encoding="utf-8") as f:
        for k, v in evaluation["metrics"].items():
            f.write(f"{k}: {v}\n")

    # Save per-example results
    out_file = "rag_evaluation_results.jsonl"
    with open(out_file, "w", encoding="utf-8") as f:
        for ex in evaluation["per_example"]:
            f.write(json.dumps(ex) + "\n")
    print(f"Per-example results saved to {out_file}")
    print("Aggregated metrics saved to ragas_metrics.txt")

if __name__ == "__main__":
    main()