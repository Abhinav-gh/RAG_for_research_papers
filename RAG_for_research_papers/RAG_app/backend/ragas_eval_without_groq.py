#!/usr/bin/env python3
"""
Robust RAG evaluation pipeline (Groq removed; single HF causal LM used for both
answer generation and RAGAS evaluation).

- Uses the same HuggingFace causal LM ("microsoft/phi-2") for:
  1) producing RAG answers (was Groq previously), and
  2) serving as the RAGAS LLM for metrics (Faithfulness, ContextPrecision, ...)

DO NOT CHANGE ANYTHING ELSE (paths, filenames, logic, metrics). Only Groq was
removed and the HF model is used for both generation & metrics.
"""

import csv
import json
import os
import time
from typing import List, Dict, Any
import traceback

import torch
import chromadb
from chromadb.config import Settings
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)

from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, ContextRelevance
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.embeddings.base import HuggingfaceEmbeddings
from datasets import Dataset as HFDataset
from langchain_core.prompt_values import PromptValue
from langchain_core.outputs import Generation, LLMResult
from sentence_transformers import SentenceTransformer

# ---------------------------
# Config (paths and models) - KEEP THESE THE SAME AS YOURS
# ---------------------------
ENCODER_MODEL_PATH = "../../Encoder_Fine_Tuning/lora_finetuned/lora_bert.pt"
ENCODER_TOKENIZER_DIR = ""
CROSS_ENCODER_MODEL_PATH = "../../Cross_Encoder_Reranking/crossenc_lora_out/model_with_lora.pt"

# We'll use the HF causal LM for both generation and RAGAS metrics.
# The user requested: "microsoft/phi-2"
RAGAS_HF_MODEL = "microsoft/phi-2"

MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"  # kept for compatibility (unused)
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"     # kept for compatibility (unused)

CHROMA_PERSIST_DIR = "../../VectorDB/chroma_Data_with_Fine_tuned_BERT"
CHROMA_COLLECTION_NAME = "HP_Chunks_BERT_Embeddings_collection"

GOLDEN_CSV_RELATIVE_PATH = "golden_2_without_commas.csv"

TOP_K = 10
TOP_M = 5

LLM_MAX_NEW_TOKENS = 128
LLM_TEMPERATURE = 0.0

# Environment overrides (optional)
RAGAS_NUM_WORKERS = int(os.environ.get("RAGAS_NUM_WORKERS", "1"))  # default 1 to avoid GPU contention
RAGAS_JOB_TIMEOUT = int(os.environ.get("RAGAS_JOB_TIMEOUT", "300"))  # default 5 minutes

LLM_LOAD_MODE = os.environ.get("LLM_LOAD_MODE", "").lower()  # "", "cpu-only"
DEVICE = "cuda" if torch.cuda.is_available() and LLM_LOAD_MODE != "cpu-only" else "cpu"
print(f"Using device: {DEVICE}")

# ---------------------------
# Encoder loader (kept behavior)
# ---------------------------
def load_encoder(model_path: str, tokenizer_dir: str = ""):
    from transformers import BertConfig, BertModel, BertTokenizer, AutoTokenizer, AutoModel
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Encoder model path not found: {model_path}")
    if os.path.isdir(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModel.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()
        return tokenizer, model
    _, ext = os.path.splitext(model_path)
    if ext.lower() in [".pt", ".pth", ".bin"]:
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir or "bert-base-uncased", use_fast=False)
        config = BertConfig()
        model = BertModel(config)
        state = torch.load(model_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
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


def encode_texts(tokenizer, model, texts: List[str], batch_size: int = 16):
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.extend([vec.cpu().numpy().astype("float32").tolist() for vec in pooled])
    return embeddings


# ---------------------------
# Cross-encoder loader + scoring (kept behavior)
# ---------------------------
def load_cross_encoder(model_path: str):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModelForSequenceClassification
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
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
        config = BertConfig(num_labels=1)
        model = BertForSequenceClassification(config)
        state = torch.load(model_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
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
            batch_cands = candidates[i:i + batch_size]
            enc = tokenizer([query] * len(batch_cands), batch_cands, padding=True, truncation=True,
                            return_tensors="pt", max_length=512)
            out = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE),
                        return_dict=True)
            logits = out.logits
            if logits.size(1) == 1:
                batch_scores = logits[:, 0].cpu().tolist()
            else:
                batch_scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            scores.extend(batch_scores)
    return scores


# ---------------------------
# HF shared LLM wrapper (used both for generation and RAGAS)
# ---------------------------
class SharedHFRagasLLM(BaseRagasLLM):
    """
    Loads a HuggingFace causal LM and exposes:
     - generate_text(prompt, n, temperature)  -> used by ragas.evaluate()
     - generate_answer(query, contexts, ...)  -> used by retrieval pipeline to produce answers
    """

    def __init__(self, model_name: str, device: str = DEVICE):
        super().__init__(run_config=RunConfig())
        self.device = device
        self.model_name_used = model_name

        # Load tokenizer + causal LM
        # Note: for large models you might want to use accelerate/device_map; this simple loader
        # loads the model onto the specified device (cpu or cuda).
        print(f"[SharedHFRagasLLM] Loading HF causal LM {model_name} on device {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        # ensure pad_token exists
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        try:
            if self.device == "cuda":
                self.model.to("cuda")
            else:
                self.model.to("cpu")
        except Exception:
            # fallback to CPU if moving to cuda fails
            self.model.to("cpu")
            self.device = "cpu"
        self.model.eval()
        print(f"[SharedHFRagasLLM] Loaded {model_name} successfully.")

    def _prompt_to_text(self, prompt: PromptValue) -> str:
        return prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

    # Method required by ragas.evaluate()
    def generate_text(self, prompt: PromptValue, n: int = 1, temperature: float = 0.0, stop=None, callbacks=None):
        text_prompt = self._prompt_to_text(prompt)
        gens = []
        for _ in range(n):
            try:
                inputs = self.tokenizer(text_prompt, return_tensors="pt", truncation=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=LLM_MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=float(temperature),
                        pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                        eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                    )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                gens.append(Generation(text=text))
            except Exception as e:
                gens.append(Generation(text=f"[SharedHFRagasLLM error: {e}]"))
        return LLMResult(generations=[[g] for g in gens])

    async def agenerate_text(self, prompt: PromptValue, n: int = 1, temperature: float = 0.0, stop=None,
                              callbacks=None):
        return self.generate_text(prompt, n=n, temperature=temperature)

    def is_finished(self, response):
        return True

    # Convenience for the retrieval pipeline: produce a concise answer from contexts + query
    def generate_answer(self, query: str, contexts: List[str], max_new_tokens: int = 128, temperature: float = 0.0) -> str:
        ctx_block = "\n\n---\n".join([f"Passage {i+1}:\n{c}" for i, c in enumerate(contexts)])
        prompt = (
            "You are a helpful system that answers the user's question based only on the provided passages.\n"
            "If the answer is not contained within the passages, say 'I don't know'.\n\n"
            f"Passages:\n{ctx_block}\n\n"
            f"Question: {query}\n"
            "Answer concisely:"
        )
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=float(temperature),
                    pad_token_id=getattr(self.tokenizer, "pad_token_id", None),
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                )
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            # Make parser-safe single-line
            text = text.replace("\n", " ").replace("\r", " ").strip()
            if not text:
                return "I don't know."
            return text
        except Exception as e:
            return f"I don't know. [HF generation error: {e}]"


# ---------------------------
# Embeddings wrapper for ragas (kept behavior)
# ---------------------------
class CustomHuggingfaceEmbeddings(HuggingfaceEmbeddings):
    def __init__(self, model_name: str):
        self.model = model_name
        self._encoder = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = self._encoder.encode(texts, show_progress_bar=False)
        return [list(map(float, e)) for e in embs]

    def embed_query(self, text: str) -> List[float]:
        emb = self._encoder.encode([text], show_progress_bar=False)[0]
        return list(map(float, emb))

    async def aembed_documents(self, texts: List[str]):
        return self.embed_documents(texts)

    async def aembed_query(self, text: str):
        return self.embed_query(text)


# ---------------------------
# Utility: load golden CSV (kept behavior)
# ---------------------------
def load_golden_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Golden CSV not found: {path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            q = r.get("query") or r.get("question") or ""
            a = r.get("answer") or r.get("reference") or ""
            rows.append({"id": str(i), "query": q, "reference": a})
    return rows


# ---------------------------
# Main orchestration (kept retrieval/generation logic)
# ---------------------------
def run_ragas_evaluation():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR, settings=Settings(anonymized_telemetry=False))
    try:
        collection = client.get_collection(CHROMA_COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(f"Could not open Chroma collection '{CHROMA_COLLECTION_NAME}': {e}")

    print("[INFO] Loading encoder and cross-encoder ...")
    encoder_tok, encoder_model = load_encoder(ENCODER_MODEL_PATH, tokenizer_dir=ENCODER_TOKENIZER_DIR)
    cross_tok, cross_model = load_cross_encoder(CROSS_ENCODER_MODEL_PATH)

    # Load shared HF model (used for both generation & ragas)
    print(f"[INFO] Loading shared HF model for generation + RAGAS: {RAGAS_HF_MODEL}")
    shared_llm = SharedHFRagasLLM(model_name=RAGAS_HF_MODEL, device=DEVICE)

    golden = load_golden_csv(GOLDEN_CSV_RELATIVE_PATH)
    print(f"[INFO] Loaded {len(golden)} golden examples.")

    records = []
    for idx, item in enumerate(golden):
        qid, query, reference = item["id"], item["query"], item["reference"]
        try:
            q_emb = encode_texts(encoder_tok, encoder_model, [query], batch_size=1)[0]
        except Exception as e:
            print(f"[WARN] Encoding query failed for id {qid}: {e}")
            q_emb = None

        try:
            if q_emb is not None:
                chroma_res = collection.query(query_embeddings=[q_emb], n_results=TOP_K, include=["documents", "metadatas"])
                docs = chroma_res.get("documents", [[]])[0]
            else:
                chroma_res = collection.query(query_texts=[query], n_results=TOP_K, include=["documents", "metadatas"])
                docs = chroma_res.get("documents", [[]])[0]
        except Exception as e:
            print(f"[WARN] Chroma query failed for query id {qid}: {e}")
            docs = []

        candidates = docs or []
        if len(candidates) == 0:
            gen_answer = "I don't know."
            contexts = []
        else:
            try:
                scores = cross_encode_scores(cross_tok, cross_model, query, candidates, batch_size=8)
                scored_sorted = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
                top_m = [c for c, s in scored_sorted[:TOP_M]]
            except Exception as e:
                print(f"[WARN] Cross-encoder reranking failed for qid {qid}: {e}")
                top_m = candidates[:TOP_M]

            try:
                # Use shared HF model for answer generation
                gen_answer = shared_llm.generate_answer(query=query, contexts=top_m, max_new_tokens=LLM_MAX_NEW_TOKENS, temperature=LLM_TEMPERATURE)
            except Exception as e:
                print(f"[ERROR] HF generation failed for qid {qid}: {e}")
                gen_answer = "I don't know."

            contexts = top_m

        records.append({
            "id": qid,
            "question": query,
            "answer": gen_answer,
            "contexts": contexts,
            "ground_truth": reference
        })

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx+1}/{len(golden)} queries")

    # ---------------------------
    # Prepare ragas LLM wrapper & embeddings (reuse shared HF)
    # ---------------------------
    print("[INFO] Wrapping HF model for ragas (metrics) - reusing shared model...")
    ragas_llm = shared_llm
    ragas_embeddings = CustomHuggingfaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

    metrics_to_run = [Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall(), ContextRelevance()]

    # ---------------------------
    # Call evaluate() - try modern signatures then fallback
    # ---------------------------
    print("[INFO] Running ragas.evaluate() ...")
    start_time = time.time()
    results = None
    evaluate_kwargs = dict(dataset=HFDataset.from_list(records),
                           metrics=metrics_to_run,
                           llm=ragas_llm,
                           embeddings=ragas_embeddings)

    # If the user explicitly requested more workers via env, respect it when possible
    if RAGAS_NUM_WORKERS and RAGAS_NUM_WORKERS > 1:
        evaluate_kwargs["num_workers"] = RAGAS_NUM_WORKERS
    # add job_timeout if ragas supports it (we'll try)
    evaluate_kwargs["job_timeout"] = RAGAS_JOB_TIMEOUT

    tried_signatures = []
    try:
        # Try signature with num_workers + job_timeout (modern)
        results = evaluate(**evaluate_kwargs)
        tried_signatures.append("with num_workers/job_timeout")
    except TypeError as e:
        # likely evaluate doesn't accept those kwargs
        print(f"[INFO] evaluate() didn't accept num_workers/job_timeout: {e}")
        # remove keys and try again
        for k in ("num_workers", "job_timeout"):
            evaluate_kwargs.pop(k, None)
        try:
            results = evaluate(**evaluate_kwargs)
            tried_signatures.append("without num_workers/job_timeout")
        except TypeError as e2:
            # fallback: maybe evaluate expects positional args or a different API; try minimal call
            print(f"[INFO] evaluate() minimal call attempt: {e2}")
            try:
                results = evaluate(HFDataset.from_list(records), metrics_to_run, ragas_llm, ragas_embeddings)
                tried_signatures.append("positional minimal call")
            except Exception as e3:
                # give detailed error
                print("Failed to call evaluate() with multiple signatures. Last exception:")
                traceback.print_exc()
                raise RuntimeError("Could not call ragas.evaluate() with available signatures.") from e3
    end_time = time.time()
    print(f"[INFO] ragas.evaluate() call succeeded (attempts: {tried_signatures}). Duration (s): {end_time - start_time:.1f}")

    # ---------------------------
    # Normalize results across ragas versions
    # ---------------------------
    aggregated = {}
    per_example = []

    try:
        if results is None:
            raise RuntimeError("evaluate() returned None")
        # If it's a dataclass-like object with attributes
        if hasattr(results, "results") and getattr(results, "results") is not None:
            aggregated = getattr(results, "results")
        elif hasattr(results, "summaries") and getattr(results, "summaries") is not None:
            aggregated = getattr(results, "summaries")
        elif isinstance(results, dict):
            aggregated = results
        else:
            # try to stringify
            aggregated = str(results)

        # per-example details
        if hasattr(results, "per_example_results") and getattr(results, "per_example_results") is not None:
            per_example = getattr(results, "per_example_results")
        elif hasattr(results, "details") and getattr(results, "details") is not None:
            per_example = getattr(results, "details")
        else:
            per_example = []
    except Exception as e:
        print(f"[WARN] Could not normalize ragas results object cleanly: {e}")
        aggregated = results
        per_example = []

    # If aggregated is a mapping of metric->list of per-example values, compute means
    try:
        if isinstance(aggregated, dict):
            tmp = {}
            for k, v in aggregated.items():
                try:
                    if isinstance(v, (list, tuple)):
                        arr = [float(x) for x in v]
                        tmp[k] = float(sum(arr) / len(arr)) if len(arr) > 0 else None
                    else:
                        tmp[k] = float(v) if isinstance(v, (int, float, str)) and str(v).replace(".", "", 1).replace("-", "", 1).isdigit() else v
                except Exception:
                    tmp[k] = v
            aggregated = tmp
    except Exception:
        pass

    # ---------------------------
    # Persist outputs
    # ---------------------------
    out_metrics_file = "ragas_metrics.txt"
    out_json_file = "ragas_full_results.json"
    out_per_example = "ragas_per_example.jsonl"

    try:
        with open(out_metrics_file, "w", encoding="utf-8") as f:
            f.write("=== RAGAS aggregated metrics ===\n")
            f.write(f"Time: {time.ctime()}\n")
            f.write(f"Duration (seconds): {(end_time - start_time):.1f}\n\n")
            f.write(json.dumps(aggregated, indent=2, default=str))
        print(f"[INFO] Aggregated metrics saved to {out_metrics_file}")
    except Exception as e:
        print(f"[ERROR] Could not write aggregated metrics: {e}")

    try:
        with open(out_json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[INFO] Full results saved to {out_json_file}")
    except Exception as e:
        print(f"[ERROR] Could not write full results JSON: {e}")

    try:
        with open(out_per_example, "w", encoding="utf-8") as f:
            for i, rec in enumerate(records):
                entry = dict(rec)
                # try to attach per-example metrics if available
                try:
                    if isinstance(per_example, (list, tuple)) and i < len(per_example):
                        if isinstance(per_example[i], dict):
                            entry.update(per_example[i])
                        else:
                            entry["metric_detail"] = per_example[i]
                except Exception:
                    pass
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[INFO] Per-example results saved to {out_per_example}")
    except Exception as e:
        print(f"[ERROR] Could not write per-example results: {e}")

    print("Done. Summary:")
    print("Elapsed (minutes):", (end_time - start_time) / 60.0)
    print("Aggregated metrics (sample):")
    print(json.dumps(aggregated, indent=2, default=str))


if __name__ == "__main__":
    run_ragas_evaluation()