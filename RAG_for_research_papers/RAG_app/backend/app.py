from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional

from backend.model import CrossEncoderLoRAWrapper
from backend.ragas_eval import run_ragas_evaluation
from backend.utils import health_check, get_model_info
from backend.encoder_model import BertEncoder
from backend.chromadb_utils import ChromaDBClient
from backend.gemini_answer import GeminiAnswerGenerator


import os
import sys
# Ensure repository root (two levels above this backend folder) is on sys.path so Cross_Encoder_Reranking can be imported
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
	sys.path.insert(0, repo_root)
try:
	from Cross_Encoder_Reranking.script_2 import CrossEncoderLoRA
except Exception as e:
	raise ImportError(
		"Could not import CrossEncoderLoRA from Cross_Encoder_Reranking.script_2. "
		"Make sure the Cross_Encoder_Reranking package is present in the repository root "
		"or install it as a package."
	) from e

app = FastAPI()

# Load model once at startup
model_wrapper = CrossEncoderLoRAWrapper()
encoder = BertEncoder()

# Gemini API keys (replace with your actual keys)
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY_1", "AIzaSyCpIuRgZJI9HLO01CruxSppbKjO6EhnQIc"),
    os.getenv("GEMINI_API_KEY_2", "AIzaSyCOQ5_Rk377xfh_C5yrTom-f-MTVOkh6Q4"),
]
answer_generator = GeminiAnswerGenerator(GEMINI_API_KEYS)

class QueryRequest(BaseModel):
	query: str
	top_k: int = 10
	top_m: int = 3

class EvalRequest(BaseModel):
	predictions: List[str]
	references: List[str]

@app.get("/health")
def health():
	return health_check()

@app.get("/model_info")
def model_info():
	return get_model_info()

@app.post("/query")
def query_endpoint(request: QueryRequest):
    print(">>> Query received:", request.query)

    query_emb = encoder.embed(request.query)
    print(">>> Query embedding shape:", query_emb.shape)

    # Step 2: Chroma search
    chroma = ChromaDBClient()
    chunks, chunk_ids = chroma.query(query_emb, top_k=request.top_k)

    print(">>> Retrieved chunks count:", len(chunks))

    # Print full chunks, not truncated
    for i, ch in enumerate(chunks):
        print(f"\n----- FULL CHUNK {i} (len={len(ch)}) -----")
        print(ch)
        print("----------------------------------------\n")

    print(">>> Retrieved chunk_ids:", chunk_ids)


    if len(chunks) == 0:
        print(">>> NO CHUNKS RETURNED BY CHROMA")
        return {"answer": "", "contexts": []}

    # Step 3: Reranking
    probs = model_wrapper.predict_batch(request.query, chunks)
    scores = list(zip(chunks, probs))
    print(">>> Raw reranking scores:", scores)
    scores.sort(key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, score in scores[:request.top_m]]
    print(">>> Top reranked chunks:", top_chunks)

    # Step 4: Answer generation using Gemini API
    answer = answer_generator.generate_answer(request.query, top_chunks)
    print(">>> Gemini answer:", answer)

    return {
        "answer": answer,
        "contexts": top_chunks,
        "chunk_ids": chunk_ids[:request.top_m]
    }



@app.post("/evaluate")
def evaluate_endpoint(request: EvalRequest):
	metrics = run_ragas_evaluation(request.predictions, request.references)
	return metrics

