#!/usr/bin/env python3
"""
Corrected end-to-end RAG + RAGAS evaluation script (compatible with ragas 0.3.9)

- Uses your encoder + cross-encoder + Chroma retrieval pipeline
- Uses HuggingFace LLaMA as generator and as judge (wrapped)
- Instantiates ragas metrics correctly for ragas 0.3.9
- Adds a safe fallback for LLaMA loading (GPU -> CPU on OOM)
"""

import csv
import json
import os
import time
from typing import List, Dict

import torch
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM
import numpy as np
import logging

# Ragas (0.3.9) metric classes
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    AnswerCorrectness,
    ContextPrecision,
    ContextRecall,
    # ContextRelevancy,
)
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.embeddings.base import HuggingfaceEmbeddings

# Dataset
from datasets import Dataset as HFDataset

# langchain_core wrappers expected by ragas
from langchain_core.prompt_values import PromptValue
from langchain_core.outputs import Generation, LLMResult

# sentence-transformers
from sentence_transformers import SentenceTransformer

# Optional backend modules you referenced (kept as imports; ensure they exist in your project)
# If they are not needed, remove or comment the following imports.
# from backend.encoder_model import BertEncoder
# from backend.chromadb_utils import ChromaDBClient
# from backend.model import CrossEncoderLoRAWrapper
# from backend.utils import health_check, get_model_info

# ---------------------------
# Config (paths and models)
# ---------------------------
ENCODER_MODEL_PATH = "../../Encoder_Fine_Tuning/lora_finetuned/lora_bert.pt"
ENCODER_TOKENIZER_DIR = ""
CROSS_ENCODER_MODEL_PATH = "../../Cross_Encoder_Reranking/crossenc_lora_out/model_with_lora.pt"

# Main generator/judge model (user-selected)
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"

CHROMA_PERSIST_DIR = "../../VectorDB/chroma_Data_with_Fine_tuned_BERT"
CHROMA_COLLECTION_NAME = "HP_Chunks_BERT_Embeddings_collection"

GOLDEN_CSV_RELATIVE_PATH = "temp.csv"

TOP_K = 10
TOP_M = 5

LLM_MAX_NEW_TOKENS = 128
LLM_TEMPERATURE = 0.0

# Allow forcing CPU-only loading via env var for low-memory machines
LLM_LOAD_MODE = os.environ.get("LLM_LOAD_MODE", "").lower()  # values: "", "cpu-only"

DEVICE = "cuda" if torch.cuda.is_available() and LLM_LOAD_MODE != "cpu-only" else "cpu"
print(f"Using device: {DEVICE}")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Encoder loader (unchanged logic)
# ---------------------------
def load_encoder(model_path: str, tokenizer_dir: str = ""):
    from transformers import BertConfig, BertModel, BertTokenizer

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Encoder model path not found: {model_path}")

    if os.path.isdir(model_path):
        logger.info("[load_encoder] Detected directory model.")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModel.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model

    _, ext = os.path.splitext(model_path)
    if ext.lower() in [".pt", ".pth", ".bin"]:
        logger.info(f"[load_encoder] Raw checkpoint: {model_path}")
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=1024,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            pad_token_id=0,
            type_vocab_size=2,
        )

        model = BertModel(config)
        logger.info("[load_encoder] Loading checkpoint...")
        state = torch.load(model_path, map_location="cpu")

        # try common key shapes
        if isinstance(state, dict) and not any(k.startswith("bert.") for k in state.keys()):
            for k in ["model_state_dict", "state_dict"]:
                if k in state:
                    state = state[k]
                    break

        try:
            model.load_state_dict(state, strict=False)
        except Exception:
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
# Text encoder helper (unchanged)
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
# Cross-encoder loader + scoring (unchanged)
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
        logger.info(f"[load_cross_encoder] Raw checkpoint: {model_path}")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            max_position_embeddings=1024,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
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
        except Exception:
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
            enc = tokenizer([query] * len(batch_cands), batch_cands, padding=True, truncation=True,
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
# Load LLaMA with safe fallback on OOM
# ---------------------------
def load_llm(model_path: str):
    """
    Attempt to load HF causal LM using GPU (auto) + fp16.
    On OOM, fall back to CPU (float32) to allow evaluation to proceed (slowly).
    """
    logger.info(f"[load_llm] Loading LLM from HuggingFace Hub: {model_path} (device={DEVICE})")

    # prefer fp16 on GPU
    preferred_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    # initial attempt: try device_map="auto" (offload if possible) - may OOM
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=preferred_dtype,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        # ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("[load_llm] Loaded model (auto device_map).")
        return tokenizer, model

    except RuntimeError as e:
        # Catch CUDA OOM or other runtime errors while attempting GPU load
        logger.warning(f"[load_llm] Initial HF load failed: {e}. Falling back to CPU.")
        # attempt CPU-only load (slower, but avoids crashes)
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("[load_llm] Loaded model on CPU (device_map={'': 'cpu'}).")
        return tokenizer, model

# ---------------------------
# LLM generate helper
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
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
    return generated

# ---------------------------
# Ragas LLM wrapper (implements BaseRagasLLM)
# ---------------------------
class HFRagasLLM(BaseRagasLLM):
    def __init__(self, tokenizer, model, run_config: RunConfig = None):
        super().__init__(run_config=run_config or RunConfig())
        self.tokenizer = tokenizer
        self.model = model
        self.device = DEVICE

    def _prompt_to_text(self, prompt: PromptValue):
        if hasattr(prompt, "to_string"):
            return prompt.to_string()
        return str(prompt)

    def generate_text(self, prompt: PromptValue, n: int = 1, temperature: float = 0.0, stop=None, callbacks=None):
        text_prompt = self._prompt_to_text(prompt)
        gens = []
        for i in range(n):
            inputs = self.tokenizer(text_prompt, return_tensors="pt", truncation=True, max_length=2048).to(DEVICE)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False if temperature == 0.0 else True,
                    temperature=temperature,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            out_text = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True).strip()
            gens.append(Generation(text=out_text))
        # LLMResult expects list-of-list-of-Generation
        return LLMResult(generations=[[g] for g in gens])

    async def agenerate_text(self, prompt: PromptValue, n: int = 1, temperature: float = 0.0, stop=None, callbacks=None):
        return self.generate_text(prompt, n=n, temperature=temperature, stop=stop, callbacks=callbacks)

    def is_finished(self, response):
        return True

# ---------------------------
# Embeddings wrapper for ragas
# ---------------------------
class CustomHuggingfaceEmbeddings(HuggingfaceEmbeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text], show_progress_bar=False)[0].tolist()

    async def aembed_documents(self, texts: List[str]):
        return self.embed_documents(texts)

    async def aembed_query(self, text: str):
        return self.embed_query(text)

# ---------------------------
# Utility: load golden CSV
# ---------------------------
def load_golden_csv(path: str) -> List[Dict[str, str]]:
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

# ---------------------------
# Main orchestration
# ---------------------------
def run_ragas_evaluation():
    # 1) open chroma collection
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(f"Could not open Chroma collection '{CHROMA_COLLECTION_NAME}': {e}")

    # 2) load encoder and cross-encoder
    logger.info("[INFO] Loading encoder and cross-encoder ...")
    encoder_tok, encoder_model = load_encoder(ENCODER_MODEL_PATH, tokenizer_dir=ENCODER_TOKENIZER_DIR)
    cross_tok, cross_model = load_cross_encoder(CROSS_ENCODER_MODEL_PATH)

    # 3) load or attempt to load LLaMA (generator + judge)
    logger.info("[INFO] Loading generator/judge LLaMA model ...")
    llm_tok, llm_model = load_llm(MODEL_PATH)

    # 4) load golden csv
    golden = load_golden_csv(GOLDEN_CSV_RELATIVE_PATH)
    logger.info(f"[INFO] Loaded {len(golden)} golden examples from {GOLDEN_CSV_RELATIVE_PATH}")

    # 5) produce RAG answers by retrieval -> rerank -> generate
    records = []
    for idx, item in enumerate(golden):
        qid = item["id"]
        query = item["query"]
        reference = item["reference"]

        # encode query
        try:
            q_emb = encode_texts(encoder_tok, encoder_model, [query], batch_size=1)[0]
        except Exception as e:
            logger.warning(f"[WARN] Encoding query failed for id {qid}: {e}")
            q_emb = None

        # query chroma
        try:
            if q_emb is not None:
                chroma_res = collection.query(
                    query_embeddings=[q_emb],
                    n_results=TOP_K,
                    include=["documents", "embeddings", "metadatas"]
                )
                docs = chroma_res.get("documents", [[]])[0]
            else:
                chroma_res = collection.query(query_texts=[query], n_results=TOP_K, include=["documents", "metadatas"])
                docs = chroma_res.get("documents", [[]])[0]
        except Exception as e:
            logger.warning(f"[WARN] Chroma query failed for query id {qid}: {e}")
            docs = []

        candidates = docs or []
        if len(candidates) == 0:
            logger.warning(f"[WARN] No candidates returned for query id {qid}. Producing empty answer.")
            gen_answer = ""
            contexts = []
        else:
            try:
                scores = cross_encode_scores(cross_tok, cross_model, query, candidates, batch_size=8)
                scored = list(zip(candidates, scores))
                scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
                top_m = [c for c, s in scored_sorted[:TOP_M]]
            except Exception as e:
                logger.warning(f"[WARN] Cross-encoder reranking failed for qid {qid}: {e}")
                top_m = candidates[:TOP_M]

            try:
                gen_answer = llm_generate_answer(llm_tok, llm_model, query, top_m, max_new_tokens=LLM_MAX_NEW_TOKENS, temperature=LLM_TEMPERATURE)
            except Exception as e:
                logger.error(f"[ERROR] LLM generation failed for qid {qid}: {e}")
                gen_answer = ""

            contexts = top_m

        records.append({
            "id": qid,
            "question": query,
            "answer": gen_answer,
            "contexts": contexts,
            "ground_truth": reference
        })

        if (idx + 1) % 10 == 0:
            logger.info(f"Processed {idx+1}/{len(golden)} queries")

    # 6) Prepare HF Dataset for ragas
    logger.info("[INFO] Preparing HF Dataset for ragas...")
    hf_dataset = HFDataset.from_list(records)

    # 7) ragas LLM wrapper + embeddings
    logger.info("[INFO] Wrapping LLaMA for ragas (LLM & embeddings)...")
    ragas_llm = HFRagasLLM(tokenizer=llm_tok, model=llm_model, run_config=RunConfig())
    ragas_embeddings = CustomHuggingfaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

    # 8) Metrics: instantiate metric objects (required by ragas 0.3.9)
    metrics_to_run = [
        Faithfulness(),
        AnswerRelevancy(),
        AnswerCorrectness(),
        ContextPrecision(),
        ContextRecall(),
        # ContextRelevancy(),
    ]
    logger.info("[INFO] Running ragas.evaluate() - this will call the wrapped LLaMA many times for judge prompts...")
    start_time = time.time()
    results = evaluate(
        dataset=hf_dataset,
        metrics=metrics_to_run,
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )
    end_time = time.time()
    elapsed_min = (end_time - start_time) / 60.0

    # # 9) Aggregate & persist results
    # aggregated = {}
    # try:
    #     for metric_name in results:
    #         vals = results[metric_name]
    #         try:
    #             arr = np.array(vals, dtype=float)
    #             aggregated[metric_name] = float(np.nanmean(arr))
    #         except Exception:
    #             aggregated[metric_name] = vals
    # except Exception as e:
    #     logger.warning(f"[WARN] Error aggregating ragas results: {e}")
    #     aggregated = results
    # 9) RAGAS 0.3.9 returns EvaluationResult, not a dict
    aggregated = results.summaries          # dict: metric → float
    per_example = results.details           # list of dicts with per-example scores


    out_metrics_file = "ragas_metrics.txt"
    out_json_file = "ragas_full_results.json"
    out_per_example = "ragas_per_example.jsonl"

    # logger.info("[INFO] Writing results to disk...")
    # try:
    #     with open(out_metrics_file, "w", encoding="utf-8") as f:
    #         f.write("=== RAGAS aggregated metrics ===\n")
    #         f.write(f"Time: {time.ctime()}\n")
    #         f.write(f"Duration (minutes): {elapsed_min:.2f}\n\n")
    #         f.write(json.dumps(aggregated, indent=2))
    #         f.write("\n\n=== Full raw results object ===\n")
    #         f.write(json.dumps(results, indent=2, default=str))
    #     logger.info(f"Aggregated metrics saved to {out_metrics_file}")
    # except Exception as e:
    #     logger.error(f"[ERROR] Could not write aggregated metrics: {e}")

    # try:
    #     with open(out_json_file, "w", encoding="utf-8") as f:
    #         json.dump(results, f, indent=2, default=str)
    #     logger.info(f"Full results saved to {out_json_file}")
    # except Exception as e:
    #     logger.error(f"[ERROR] Could not write full results JSON: {e}")

    # # Per-example output
    # try:
    #     per_example = []
    #     for i, rec in enumerate(records):
    #         entry = dict(rec)
    #         for metric_name, metric_vals in results.items():
    #             try:
    #                 entry[metric_name] = metric_vals[i]
    #             except Exception:
    #                 pass
    #         per_example.append(entry)

    #     with open(out_per_example, "w", encoding="utf-8") as f:
    #         for ex in per_example:
    #             f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    #     logger.info(f"Per-example results saved to {out_per_example}")
    # except Exception as e:
    #     logger.error(f"[ERROR] Could not write per-example results: {e}")

    # logger.info("Done. Summary:")
    # logger.info("Elapsed (minutes): %.2f", elapsed_min)
    # logger.info("Aggregated metrics (sample): %s", json.dumps(aggregated, indent=2))
    logger.info("[INFO] Writing results to disk...")

    # aggregated metrics → ragas_metrics.txt
    with open(out_metrics_file, "w", encoding="utf-8") as f:
        f.write("=== RAGAS aggregated metrics ===\n")
        f.write(f"Time: {time.ctime()}\n")
        f.write(f"Duration (minutes): {elapsed_min:.2f}\n\n")
        for k, v in aggregated.items():
            f.write(f"{k}: {v}\n")

    logger.info(f"Aggregated metrics saved to {out_metrics_file}")

    # full results (converted manually)
    with open(out_json_file, "w", encoding="utf-8") as f:
        json.dump({
            "aggregated": aggregated,
            "per_example": per_example
        }, f, indent=2)

    logger.info(f"Full results saved to {out_json_file}")

    # per-example JSONL
    with open(out_per_example, "w", encoding="utf-8") as f:
        for ex in per_example:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    logger.info(f"Per-example results saved to {out_per_example}")

    logger.info("Done. Summary:")
    logger.info("Elapsed (minutes): %.2f", elapsed_min)
    logger.info("Aggregated metrics (sample): %s", json.dumps(aggregated, indent=2))


if __name__ == "__main__":
    run_ragas_evaluation()