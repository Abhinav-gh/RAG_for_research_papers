#!/usr/bin/env python3
"""
RAG evaluation pipeline:
- Uses Groq LLM for generating answers
- Uses a RAGAS-compatible LLM wrapper around HuggingFace models
- Avoids RagasOutputParserException
"""

import csv
import json
import os
from typing import List, Dict

import torch
import chromadb
from chromadb.config import Settings
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM

from groq import Groq
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
# Config
# ---------------------------
ENCODER_MODEL_PATH = "../../Encoder Fine Tuning/lora_finetuned/lora_bert.pt"
ENCODER_TOKENIZER_DIR = ""
CROSS_ENCODER_MODEL_PATH = "../../Cross_Encoder_Reranking/crossenc_lora_out/model_with_lora.pt"
MODEL_PATH = "meta-llama/Llama-3.1-8B-Instruct"
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
CHROMA_PERSIST_DIR = "../../VectorDB/chroma_Data_with_Fine_tuned_BERT"
CHROMA_COLLECTION_NAME = "HP_Chunks_BERT_Embeddings_collection"
GOLDEN_CSV_RELATIVE_PATH = "golden_2_without_commas.csv"

TOP_K = 10
TOP_M = 5
LLM_MAX_NEW_TOKENS = 128
LLM_TEMPERATURE = 0.0
LLM_LOAD_MODE = os.environ.get("LLM_LOAD_MODE", "").lower()
DEVICE = "cuda" if torch.cuda.is_available() and LLM_LOAD_MODE != "cpu-only" else "cpu"
print(f"Using device: {DEVICE}")

API_KEYS = [
    "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",
    "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",
    "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",
]
_rr_index = 0

def get_next_client():
    global _rr_index
    key = API_KEYS[_rr_index % len(API_KEYS)]
    _rr_index = (_rr_index + 1) % len(API_KEYS)
    return Groq(api_key=key)

# ---------------------------
# Encoder and Cross-encoder
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
        if "state_dict" in state: state = state["state_dict"]
        new_state = {k[7:] if k.startswith("module.") else k: v for k,v in state.items()}
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
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
            input_ids = enc["input_ids"].to(DEVICE)
            attention_mask = enc["attention_mask"].to(DEVICE)
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            last = out.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            embeddings.extend([vec.cpu().numpy().astype("float32").tolist() for vec in pooled])
    return embeddings

def load_cross_encoder(model_path: str):
    from transformers import BertTokenizer, BertConfig, BertForSequenceClassification, AutoTokenizer, AutoModelForSequenceClassification
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
        if "state_dict" in state: state = state["state_dict"]
        new_state = {k[7:] if k.startswith("module.") else k: v for k,v in state.items()}
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
            enc = tokenizer([query]*len(batch_cands), batch_cands, padding=True, truncation=True, return_tensors="pt", max_length=512)
            out = model(input_ids=enc["input_ids"].to(DEVICE), attention_mask=enc["attention_mask"].to(DEVICE), return_dict=True)
            logits = out.logits
            if logits.size(1) == 1:
                batch_scores = logits[:,0].cpu().tolist()
            else:
                batch_scores = torch.softmax(logits, dim=1)[:,1].cpu().tolist()
            scores.extend(batch_scores)
    return scores

# ---------------------------
# Groq Answer Generator
# ---------------------------
class GroqLLMWrapper:
    def __init__(self, model_name: str):
        self.model_name = model_name

def llm_generate_answer(tokenizer, model, query: str, contexts: List[str],
                        max_new_tokens=128, temperature=0.0, retries=2):
    ctx_block = "\n\n---\n".join([f"Passage {i+1}:\n{c}" for i,c in enumerate(contexts)])
    prompt = f"You are a helpful system that answers the user's question based only on the passages.\nIf the answer is not contained within the passages, say 'I don't know'.\n\nPassages:\n{ctx_block}\n\nQuestion: {query}\nAnswer concisely:"

    for attempt in range(retries+1):
        try:
            client = get_next_client()
            response = client.chat.completions.create(
                model=model.model_name,
                temperature=float(temperature),
                messages=[{"role":"user","content":prompt}],
                max_tokens=max_new_tokens
            )
            choice = response.choices[0]
            text = getattr(getattr(choice,"message",None),"content",None)
            text = text.replace("\n"," ").replace("\r"," ").strip() if text else "I don't know."
            return text
        except Exception as e:
            if attempt == retries:
                return f"I don't know. [Groq error: {e}]"
            continue
    return "I don't know."

# ---------------------------
# Custom RAGAS LLM wrapper (replaces HuggingFaceLLM)
# ---------------------------
class HF_RagasLLM(BaseRagasLLM):
    def __init__(self, model_name, device=DEVICE):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.device = device

    def generate_text(self, prompt: PromptValue, n:int=1, temperature:float=0.0, stop=None, callbacks=None):
        text_prompt = prompt.to_string() if hasattr(prompt,"to_string") else str(prompt)
        inputs = self.tokenizer(text_prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return LLMResult(generations=[[Generation(text=text)]])

    async def agenerate_text(self, prompt: PromptValue, n:int=1, temperature:float=0.0, stop=None, callbacks=None):
        return self.generate_text(prompt,n=n,temperature=temperature)

    def is_finished(self,response): 
        return True

# ---------------------------
# Embeddings
# ---------------------------
class CustomHuggingfaceEmbeddings(HuggingfaceEmbeddings):
    def __init__(self, model_name:str):
        self._encoder = SentenceTransformer(model_name)

    def embed_documents(self, texts:List[str]) -> List[List[float]]:
        embs = self._encoder.encode(texts, show_progress_bar=False)
        return [list(map(float, e)) for e in embs]

    def embed_query(self, text:str) -> List[float]:
        emb = self._encoder.encode([text], show_progress_bar=False)[0]
        return list(map(float, emb))

    async def aembed_documents(self, texts:List[str]): return self.embed_documents(texts)
    async def aembed_query(self, text:str): return self.embed_query(text)

# ---------------------------
# CSV loader
# ---------------------------
def load_golden_csv(path:str) -> List[Dict[str,str]]:
    if not os.path.exists(path): raise FileNotFoundError(f"Golden CSV not found: {path}")
    rows = []
    with open(path,newline="",encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i,r in enumerate(reader):
            q = r.get("query") or r.get("question") or ""
            a = r.get("answer") or r.get("reference") or ""
            rows.append({"id":str(i),"query":q,"reference":a})
    return rows

# ---------------------------
# Main pipeline
# ---------------------------
# ---------------------------
# Main pipeline (fixed for GPU / sequential)
# ---------------------------
def run_ragas_evaluation():
    client = chromadb.PersistentClient(
        path=CHROMA_PERSIST_DIR, 
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(CHROMA_COLLECTION_NAME)

    # Load encoder / cross-encoder
    encoder_tok, encoder_model = load_encoder(ENCODER_MODEL_PATH, tokenizer_dir=ENCODER_TOKENIZER_DIR)
    cross_tok, cross_model = load_cross_encoder(CROSS_ENCODER_MODEL_PATH)

    # Load golden dataset
    golden = load_golden_csv(GOLDEN_CSV_RELATIVE_PATH)
    records = []

    for idx, item in enumerate(golden):
        qid, item_query, ref = item["id"], item["query"], item["reference"]

        # Encode query
        try:
            q_emb = encode_texts(encoder_tok, encoder_model, [item_query], batch_size=1)[0]
        except Exception:
            q_emb = None

        # Retrieve candidate documents
        try:
            if q_emb is not None:
                res = collection.query(
                    query_embeddings=[q_emb], 
                    n_results=TOP_K, 
                    include=["documents","metadatas"]
                )
                docs = res.get("documents",[[]])[0]
            else:
                res = collection.query(
                    query_texts=[item_query], 
                    n_results=TOP_K, 
                    include=["documents","metadatas"]
                )
                docs = res.get("documents",[[]])[0]
        except:
            docs = []

        candidates = docs or []
        if not candidates:
            ans = "I don't know."
            contexts = []
        else:
            try:
                scores = cross_encode_scores(cross_tok, cross_model, item_query, candidates)
                top_m = [c for c,_ in sorted(zip(candidates,scores), key=lambda x:x[1], reverse=True)[:TOP_M]]
            except:
                top_m = candidates[:TOP_M]

            ans = llm_generate_answer(None, GroqLLMWrapper(GROQ_MODEL_NAME), item_query, top_m)
            contexts = top_m

        records.append({
            "id": qid,
            "question": item_query,
            "answer": ans,
            "contexts": contexts,
            "ground_truth": ref
        })

    # ---------------------------
    # RAGAS evaluation (sequential / GPU safe)
    # ---------------------------
    hf_dataset = HFDataset.from_list(records)
    ragas_llm = HF_RagasLLM(model_name="sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
    ragas_embeddings = CustomHuggingfaceEmbeddings("sentence-transformers/all-MiniLM-L6-v2")
    metrics_to_run = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        ContextRelevance()
    ]

    results = evaluate(
        dataset=hf_dataset,
        metrics=metrics_to_run,
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        num_workers=1,       # sequential evaluation
        job_timeout=300      # 5 minutes per example
    )

    aggregated = results.results
    per_example = results.per_example_results

    # Save outputs
    with open("ragas_metrics.txt","w",encoding="utf-8") as f:
        f.write(json.dumps(aggregated, indent=2))
    with open("ragas_full_results.json","w",encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    with open("ragas_per_example.jsonl","w",encoding="utf-8") as f:
        for i, rec in enumerate(records):
            entry = dict(rec)
            if i < len(per_example):
                entry.update(per_example[i])
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print("RAGAS evaluation done. Aggregated metrics:")
    print(json.dumps(aggregated, indent=2, default=str))

if __name__=="__main__":
    run_ragas_evaluation()