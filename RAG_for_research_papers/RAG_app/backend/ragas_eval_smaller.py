#!/usr/bin/env python3

import csv
import json
import os
import time
from typing import List, Dict

import torch
import chromadb
from chromadb.config import Settings
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
import numpy as np
import logging

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerCorrectness,
    AnswerRelevancy,
)
from ragas.llms.base import BaseRagasLLM
from ragas.run_config import RunConfig
from ragas.embeddings.base import HuggingfaceEmbeddings

from datasets import Dataset as HFDataset
from langchain_core.prompt_values import PromptValue
from langchain_core.outputs import Generation, LLMResult

from sentence_transformers import SentenceTransformer


# =========================================
# CONFIG
# =========================================

ENCODER_MODEL_PATH = "../../Encoder_Fine_Tuning/lora_finetuned/lora_bert.pt"
ENCODER_TOKENIZER_DIR = ""
CROSS_ENCODER_MODEL_PATH = "../../Cross_Encoder_Reranking/crossenc_lora_out/model_with_lora.pt"

GENERATOR_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"

CHROMA_PERSIST_DIR = "../../VectorDB/chroma_Data_with_Fine_tuned_BERT"
CHROMA_COLLECTION_NAME = "HP_Chunks_BERT_Embeddings_collection"

GOLDEN_CSV_RELATIVE_PATH = "temp.csv"

TOP_K = 10
TOP_M = 2   # REDUCED (was 5)

LLM_MAX_NEW_TOKENS = 128
LLM_TEMPERATURE = 0.0

LLM_LOAD_MODE = os.environ.get("LLM_LOAD_MODE", "").lower()
DEVICE = "cuda" if torch.cuda.is_available() and LLM_LOAD_MODE != "cpu-only" else "cpu"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =========================================
# ENCODER LOADER (unchanged)
# =========================================

def load_encoder(model_path: str, tokenizer_dir: str = ""):
    from transformers import BertConfig, BertModel, BertTokenizer

    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    if os.path.isdir(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModel.from_pretrained(model_path).to(DEVICE)
        model.eval()
        return tokenizer, model

    _, ext = os.path.splitext(model_path)
    if ext.lower() in [".pt", ".pth", ".bin"]:
        if tokenizer_dir and os.path.isdir(tokenizer_dir):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=False)
        else:
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=False)

        config = BertConfig()
        model = BertModel(config)
        state = torch.load(model_path, map_location="cpu")

        if "state_dict" in state:
            state = state["state_dict"]

        try:
            model.load_state_dict(state, strict=False)
        except Exception:
            fixed = {}
            for k, v in state.items():
                fixed[k.replace("module.", "")] = v
            model.load_state_dict(fixed, strict=False)

        model.to(DEVICE).eval()
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModel.from_pretrained(model_path).to(DEVICE)
    model.eval()
    return tokenizer, model


def encode_texts(tokenizer, model, texts: List[str], batch_size: int = 16):
    embs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in enc.items()}
            out = model(**inputs)
            last = out.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (last * mask).sum(1) / mask.sum(1).clamp(min=1)
            for vec in pooled.cpu():
                embs.append(vec.numpy().astype("float32").tolist())
    return embs


# =========================================
# CROSS ENCODER
# =========================================

def load_cross_encoder(model_path: str):
    from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

    if os.path.isdir(model_path):
        tok = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE).eval()
        return tok, model

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", use_fast=False)
    config = BertConfig(num_labels=1)
    model = BertForSequenceClassification(config)

    state = torch.load(model_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        fixed = {}
        for k, v in state.items():
            fixed[k.replace("module.", "")] = v
        model.load_state_dict(fixed, strict=False)

    model.to(DEVICE).eval()
    return tokenizer, model


def cross_encode_scores(tokenizer, model, query, candidates, batch_size=8):
    scores = []
    with torch.no_grad():
        for i in range(0, len(candidates), batch_size):
            batch = candidates[i : i + batch_size]
            enc = tokenizer(
                [query] * len(batch),
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            out = model(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
            )
            logits = out.logits
            if logits.size(1) == 1:
                sc = logits[:, 0].cpu().tolist()
            else:
                sc = torch.softmax(logits, 1)[:, 1].cpu().tolist()
            scores.extend(sc)
    return scores


# =========================================
# LLM LOADER (GENERATOR & JUDGE)
# =========================================

def load_llm(model_path: str):
    logger.info(f"[LLM] Loading: {model_path}")

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if DEVICE == "cuda" else None,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info("[LLM] GPU OK")
        return tokenizer, model

    except Exception as e:
        logger.warning(f"[LLM] GPU load failed → CPU fallback: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer, model


# =========================================
# RAG GENERATION
# =========================================

def llm_generate_answer(tok, model, query, contexts, max_new_tokens=128, temperature=0.0):
    ctx = "\n\n---\n".join([f"Passage {i+1}:\n{c}" for i, c in enumerate(contexts)])

    prompt = (
        "Answer using ONLY the given passages.\n"
        "Say 'I don't know' if not present.\n\n"
        f"Passages:\n{ctx}\n\n"
        f"Question: {query}\nAnswer:"
    )

    inputs = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,
        )
    text = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return text.strip()


# =========================================
# RAGAS LLM WRAPPER
# =========================================

class HFRagasLLM(BaseRagasLLM):
    def __init__(self, tok, model):
        super().__init__(run_config=RunConfig())
        self.tok = tok
        self.model = model

    def _prompt_to_text(self, prompt):
        return prompt.to_string() if hasattr(prompt, "to_string") else str(prompt)

    def generate_text(self, prompt, n=1, temperature=0.0, **_):
        text = self._prompt_to_text(prompt)
        gens = []
        for _ in range(n):
            inp = self.tok(text, return_tensors="pt", truncation=True).to(DEVICE)
            with torch.no_grad():
                out = self.model.generate(
                    **inp,
                    max_new_tokens=128,
                    do_sample=False,
                )
            out_text = self.tok.decode(out[0][inp["input_ids"].shape[-1]:], skip_special_tokens=True)
            gens.append(Generation(text=out_text.strip()))
        return LLMResult(generations=[[g] for g in gens])

    async def agenerate_text(self, *args, **kwargs):
        return self.generate_text(*args, **kwargs)

    def is_finished(self, response):
        return True


class CustomHuggingfaceEmbeddings(HuggingfaceEmbeddings):
    def __init__(self, mn):
        self.model = SentenceTransformer(mn)

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=False)[0].tolist()


# =========================================
# CSV LOADER
# =========================================

def load_golden_csv(path: str):
    rows = []
    with open(path, newline='', encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f)):
            q = r.get("query") or r.get("question") or ""
            a = r.get("answer") or r.get("reference") or ""
            rows.append({"id": str(i), "query": q, "reference": a})
    return rows


# =========================================
# MAIN
# =========================================

def run_ragas_evaluation():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(CHROMA_COLLECTION_NAME)

    encoder_tok, encoder_model = load_encoder(ENCODER_MODEL_PATH)
    cross_tok, cross_model = load_cross_encoder(CROSS_ENCODER_MODEL_PATH)

    gen_tok, gen_model = load_llm(GENERATOR_MODEL)
    judge_tok, judge_model = load_llm(JUDGE_MODEL)

    golden = load_golden_csv(GOLDEN_CSV_RELATIVE_PATH)

    records = []

    for idx, item in enumerate(golden):
        qid, query, ref = item["id"], item["query"], item["reference"]

        try:
            q_emb = encode_texts(encoder_tok, encoder_model, [query])[0]
            res = collection.query(
                query_embeddings=[q_emb],
                n_results=TOP_K,
                include=["documents"],
            )
            docs = res["documents"][0]
        except:
            docs = []

        if not docs:
            gen = ""
            contexts = []
        else:
            scores = cross_encode_scores(cross_tok, cross_model, query, docs)
            ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            contexts = [c for c, _ in ranked[:TOP_M]]
            gen = llm_generate_answer(gen_tok, gen_model, query, contexts)

        records.append({
            "id": qid,
            "question": query,
            "answer": gen,
            "contexts": contexts,
            "ground_truth": ref
        })

    hf_dataset = HFDataset.from_list(records)

    ragas_llm = HFRagasLLM(judge_tok, judge_model)
    ragas_emb = CustomHuggingfaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")

    metrics = [
        Faithfulness(),
        AnswerCorrectness(),
        AnswerRelevancy(),
    ]

    results = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=ragas_llm,
        embeddings=ragas_emb,
    )

    aggregated = results._repr_dict
    per_example = results._scores_dict

    with open("ragas_metrics.txt", "w") as f:
        for k, v in aggregated.items():
            f.write(f"{k}: {v}\n")

    with open("ragas_results.json", "w") as f:
        json.dump({"agg": aggregated, "per": per_example}, f, indent=2)

    print("DONE.")


if __name__ == "__main__":
    print("Here")
    run_ragas_evaluation()
