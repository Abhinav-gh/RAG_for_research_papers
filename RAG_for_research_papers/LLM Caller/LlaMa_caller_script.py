# ================== INSTALL LLaMA + TRANSFORMERS + ACCELERATE ==================
# Run this in a separate cell before running the main code.

# ================== DOWNLOAD / LOAD LLaMA MODEL LOCALLY ==================
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# login to hugging face hub first using:
from huggingface_hub import login
login("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")
# Example: LLaMA-3.1 8B Instruct (open-source)
# REQUIREMENTS: ~16GB GPU or use CPU (very slow)
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
# -------------------- Device setup --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

print("[INFO] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set pad_token to eos_token to avoid warnings
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("[INFO] Loading model (this may take a while)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"     # automatically uses GPU if available
)

print("[INFO] Local LLaMA model loaded successfully.")

#!/usr/bin/env python3

import os
import time
import pandas as pd
from tqdm import tqdm
from typing import List
from datetime import datetime

import chromadb
from chromadb.api.types import Documents, Metadatas

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# -------------------- Configuration --------------------
ABSOLUTE_DB_PATH = "../VectorDB/chroma_Data"
COLLECTION_NAME = "harry_potter_collection"

BATCH_SIZE = 200
NUM_QUERIES_PER_CHUNK = 5


OUTPUT_CSV = "generated_pairs.csv"

# -------------------- Token-limit enforcement --------------------
def truncate_to_token_limit(text: str, max_tokens: int = 100000):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

# -------------------- Query generation --------------------
def ask_llama_local(chunk_text: str, chunk_id: str, num_queries: int) -> List[str]:

    safe_chunk = truncate_to_token_limit(chunk_text)

    prompt = f"""
        System instruction: You are an AI that generates realistic search queries a user might input to an LLM or search system.
        Each query should be short, relevant, and reflect what someone might actually ask.

        Now, generate {num_queries} short queries for the following chunk:
        Chunk ID: {chunk_id}
        Chunk Text: "{safe_chunk}"
        Queries:
        """

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "Queries:" in text:
        text = text.split("Queries:")[1]

    queries = [q.strip("- ").strip() for q in text.split("\n") if q.strip()]
    return queries[:num_queries]

# -------------------- Main workflow --------------------
def main():
    client_db = chromadb.PersistentClient(path=ABSOLUTE_DB_PATH)
    print(f"[INFO] ChromaDB client initialized at: {ABSOLUTE_DB_PATH}")

    collection = client_db.get_collection(name=COLLECTION_NAME)
    print(f"[INFO] Using existing collection: {COLLECTION_NAME}")

    results = collection.get(include=["documents", "metadatas"])

    # Fixed: Get id from metadata (after you re-run the ChromaDB notebook with updated metadata)
    chunks = [
        {"id": meta["id"], "text": doc}
        for doc, meta in zip(results["documents"], results["metadatas"])
        if meta.get("ischunk") is True
    ]
    print(f"[INFO] Found {len(chunks)} chunks (ischunk=True)")

    all_pairs = []

    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Processing chunk-batches"):
        batch = chunks[i: i + BATCH_SIZE]

        for chunk in batch:
            try:
                queries = ask_llama_local(
                    chunk["text"],
                    chunk["id"],
                    NUM_QUERIES_PER_CHUNK
                )
            except Exception as e:
                print(f"[ERROR] Failed to generate for chunk {chunk['id']}: {e}")
                continue

            for q in queries:
                all_pairs.append({"query": q, "chunk_id": chunk["id"]})

    df = pd.DataFrame(all_pairs)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"[INFO] Saved {len(df)} query-chunk pairs to {OUTPUT_CSV}")

# -------------------- Run --------------------
if __name__ == "__main__":
    main()
