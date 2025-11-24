# RAG — Step-by-Step README

This README gives you **complete, project blueprints**:

1. **Retrieval-Augmented Generation (RAG) for Course/Domain QA**

Both are scoped to be impressive, realistic, and demo-able. You’ll see **what you will build**, **what’s pre-trained**, a **step-by-step plan**, metrics, risks, and stretch goals.

---

## Project 1 — Retrieval-Augmented Generation (RAG)

### What you will build (your work)

* A data ingestion & cleaning pipeline for your domain PDFs/HTML/notes.
* Text chunking strategy (by headings / sliding window) with metadata.
* Vector store index (FAISS) + retriever (top-k search, MMR, or BM25+embeddings hybrid).
* RAG answer pipeline: retrieve → construct prompt with citations → generate.
* Evaluation harness (retrieval & answer quality, hallucination checks).
* Optional: a minimal **Streamlit** app for interactive Q\&A.

### What you will use pre-trained

* **Sentence embeddings**: e.g., `sentence-transformers/all-MiniLM-L6-v2` (or similar).
* **Generator LLM**: e.g., `flan-t5-base` (CPU/GPU friendly) or any open-source 7B via an API/`transformers` (if compute allows).
* Optional: pre-trained **BM25** (rule-based) as a baseline.

### Architecture (high level)

```
   Docs (PDF/HTML/Notes)
          │
   Ingest & Clean → Chunk → Embed (pre-trained) → Index (FAISS)
          │
        Query ──► Retrieve top-k ──► Rerank (optional)
          │                         │
          └──────────────► Construct RAG prompt with citations
                                   │
                               Generate (pre-trained LLM)
                                   │
                               Answer + Sources
```

### Step-by-step

#### 0) Environment

```bash
# Python >= 3.10
pip install transformers sentence-transformers faiss-cpu rank-bm25 rapidfuzz
pip install pypdf unstructured[local-inference] nltk beautifulsoup4 lxml
pip install streamlit uvicorn fastapi
```

#### 1) Data prep

* Collect PDFs/HTML/Markdown from your domain (course notes, manuals, internal docs).
* Extract text (e.g., `pypdf`, `unstructured`) + keep **source metadata** (title, page, section).
* Clean: remove headers/footers, normalize whitespace, keep tables as text if necessary.

#### 2) Chunking

* Start with **heading-aware** chunks (e.g., split by H2/H3) or **sliding window**:

  * Size: \~512–1000 tokens; Overlap: 50–100 tokens to preserve context.
* Store: `{text, source_id, page, section, chunk_id}`.

#### 3) Embeddings & Index

* Compute embeddings with a pre-trained SentenceTransformer.
* Build **FAISS** index; persist to disk.
* (Optional) Keep a **BM25** index too and use **hybrid retrieval** (BM25 + embeddings) for robustness.

#### 4) Retrieval & Reranking

* Implement `retrieve(query, k=8)` using:

  * **Embeddings** (cosine similarity) or
  * **Hybrid**: union BM25+embeddings → **MMR** rerank to diversify.

#### 5) RAG Prompt Construction

* Concatenate top chunks with **citations** (source + page).
* Keep the prompt **deterministic** and **length-limited** (truncate to model’s max tokens).

**Prompt sketch**

```
You are a helpful assistant. Answer USING ONLY the sources below. If unsure, say you don’t know.

Question: {user_query}

Sources:
[1] ({source_1_meta}): {chunk_1}
[2] ({source_2_meta}): {chunk_2}
...

Give a concise answer and cite sources like [1], [3].
```

#### 6) Generation

* Use a pre-trained seq2seq (e.g., **FLAN-T5**) via `transformers` pipeline or model class.
* Return: **answer + list of citations**.

#### 7) Evaluation

* **Retrieval**: Recall\@k, MRR, nDCG@(k).
* **Answer**: Exact Match / F1 vs. gold, **Faithfulness** (does the answer quote/align with sources?), simple **hallucination rate** check by string matching citations.
* Build a small validation set: `{query, gold_answer, gold_sources}`.

#### 8) App (Optional but great for demo)

* Streamlit page with two panels: left = query & params (k, rerank), right = answer + clickable citations.

#### 9) Reporting

* Show ablations: **BM25 vs Embeddings vs Hybrid**; **chunk size**; **rerank on/off**.
* Show error analysis: where hallucinations occur and how citations fix them.

### Minimal baseline vs improved variant

* **Baseline**: Embeddings-only retrieval (k=5) + FLAN-T5; no rerank; naive chunking.
* **Improved**: Hybrid retrieval (BM25+Embeddings) + MMR rerank; heading-aware chunking; prompt with strict “use sources only”.

### Metrics to report

* Retrieval: Recall\@5/10, MRR, nDCG.
* Answer: EM/F1; qualitative faithfulness (manual spot-checks), citation correctness rate.

### Risks & straight talk

* Garbage-in garbage-out: messy PDFs → poor retrieval. Clean aggressively.
* Overlong chunks blow up token limit: keep chunks compact.
* Generator can still hallucinate: enforce “use sources only” and **refuse when unsure**.

### Stretch goals (pick 1–2)

* **Reranker** fine-tuned on your domain (e.g., cross-encoder).
* **Context compression** (Map-Reduce, BGE summarizer) to fit more evidence.
* **Feedback loop**: collect bad answers, improve chunking/index.

---



## Quickstart Commands (copy/paste)

### RAG

```bash
# Build index
python rag/ingest.py --src data/pdfs
python rag/chunk.py --in data/raw --out data/chunks.jsonl
python rag/index_build.py --chunks data/chunks.jsonl --faiss data/index.faiss

# Query
python rag/rag_pipeline.py --query "What is X?" --k 8 --index data/index.faiss

# App
streamlit run rag/app_streamlit.py
```


## What to show in your report/demo

* **RAG**: latency per query, Recall\@k, EM/F1, examples with correct citations vs. failures.
* **Captioning**: table of BLEU/METEOR/CIDEr; side-by-side gallery: *ground truth vs baseline vs BLIP*.

## Ethics & Responsible AI (keep it crisp)

* **RAG**: Cite sources, refuse when unsure. Avoid leaking private data.
* **Captioning**: Avoid offensive terms; disclose model limitations; respect dataset licenses.

## Hardware Notes
****
* Both projects run on a single consumer GPU or Colab.
* Keep batch sizes modest; freeze encoders when memory is tight.

## Final word

These two projects cover **IR + LLM grounding** and **vision–language modeling**. They are resume-grade if you show: clear metrics, a live demo, sensible ablations, and honest failure analysis. Build the baseline fast, then polish one or two stretch goals. That’s how you win the evaluation without boiling the ocean.

