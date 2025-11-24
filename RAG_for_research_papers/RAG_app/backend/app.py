from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional

from model import CrossEncoderLoRAWrapper
from ragas_eval import run_ragas_evaluation
from utils import health_check, get_model_info
from encoder_model import BertEncoder
from chromadb_utils import ChromaDBClient
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
model_wrapper = CrossEncoderLoRAWrapper(CrossEncoderLoRA)

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
	# Step 1: Embed query using fine-tuned BERT
	encoder = BertEncoder()
	query_emb = encoder.embed(request.query)

	# Step 2: ChromaDB search for top_k chunks
	chroma = ChromaDBClient()
	chunks, chunk_ids = chroma.query(query_emb, top_k=request.top_k)

	# Step 3: Rerank chunks using cross-encoder
	scores = []
	for chunk in chunks:
		prob = model_wrapper.predict(request.query, chunk)
		# prob is a list, take first element if needed
		score = prob[0] if isinstance(prob, list) else prob
		scores.append((chunk, score))

	# Step 4: Sort and select top_m
	scores.sort(key=lambda x: x[1], reverse=True)
	top_chunks = [chunk for chunk, score in scores[:request.top_m]]

	return {"top_chunks": top_chunks}

@app.post("/evaluate")
def evaluate_endpoint(request: EvalRequest):
	metrics = run_ragas_evaluation(request.predictions, request.references)
	return metrics

