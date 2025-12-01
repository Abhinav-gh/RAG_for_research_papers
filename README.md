# RAG for Research Papers

A comprehensive **Retrieval-Augmented Generation (RAG)** system for domain-specific question answering, built with a custom BERT encoder, ChromaDB vector store, cross-encoder reranking, and LLM-based answer generation.

---

## Table of Contents

- [Introduction](#introduction)
- [Project Architecture](#project-architecture)
- [BERT Encoder Features](#bert-encoder-features)
- [BERT Encoder Variants](#bert-encoder-variants)
- [Setup & Installation](#setup--installation)
- [Project Structure](#project-structure)
- [How to Train](#how-to-train)
  - [1. Chunking](#1-chunking)
  - [2. ChromaDB Setup](#2-chromadb-setup)
  - [3. BERT Encoder Training](#3-bert-encoder-training)
  - [4. Encoder Fine-Tuning (LoRA)](#4-encoder-fine-tuning-lora)
  - [5. Cross-Encoder Reranking](#5-cross-encoder-reranking)
- [Running the RAG Application](#running-the-rag-application)
- [Evaluation](#evaluation)
- [References](#references)

---

## Introduction

This project implements a complete RAG pipeline for domain-specific question answering. The system includes:

- **Custom BERT Encoder**: Built from scratch with Word2Vec embeddings, trained using Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
- **Semantic Chunking**: Intelligent document splitting for optimal retrieval
- **ChromaDB Vector Store**: Persistent vector database for efficient similarity search
- **Cross-Encoder Reranking**: LoRA-fine-tuned reranker for improved retrieval accuracy
- **LLM Answer Generation**: Gemini API integration for generating contextual answers
- **RAGAS Evaluation**: Comprehensive evaluation metrics including faithfulness, answer relevancy, and contextual precision

---

## Project Architecture

```
   Documents (PDF)
          │
   Extract & Clean → Chunk (Semantic/Rule-based) → Embed (BERT) → Index (ChromaDB)
          │
        Query ──► Retrieve top-k ──► Cross-Encoder Rerank
          │                              │
          └──────────────► Construct RAG prompt with contexts
                                   │
                               Generate (Gemini/LLaMA)
                                   │
                               Answer + Sources
```

---

## BERT Encoder Features

The custom BERT encoder implementation includes:

### Core Architecture
- **BERT-base configuration**: 12 layers, 12 attention heads, 768 hidden size
- **Word2Vec Token Embeddings**: Custom trained on corpus
- **Learned Positional & Segment Embeddings**: Standard BERT embeddings
- **Multi-Head Self-Attention**: Full transformer encoder architecture

### Pre-training Objectives
- **Masked Language Modeling (MLM)**:
  - 15% of tokens selected for masking
  - 80% replaced with [MASK], 10% random token, 10% unchanged
  - PAD tokens excluded from random replacements
- **Next Sentence Prediction (NSP)**: Binary classification for sentence pair coherence

### Advanced Features
- **Mixture of Experts (MoE)**: Optional MoE layer in Feed-Forward Network for increased model capacity
- **Mask-Aware Mean Pooling**: Better downstream performance than CLS pooling for retrieval tasks
- **Attention Pooling**: Alternative pooling strategy available

### Special Tokens
```
[PAD], [CLS], [SEP], [MASK], [UNK]
```

---

## BERT Encoder Variants

The project includes multiple BERT encoder implementations in the `Encoder/` directory:

| Script | Description |
|--------|-------------|
| `encoder_only.ipynb` | Base BERT encoder with MLM and NSP |
| `encoder_only_with_mask_aware_mean_pooling.ipynb` | Uses mask-aware mean pooling instead of CLS token |
| `encoder_only_with_attention_pooling.ipynb` | Attention-based pooling mechanism |
| `encoder_only_with_corrected_NSP.ipynb` | Improved NSP implementation |
| `encoder_only_with_corrected_NSP_and_segment_embeddings.ipynb` | Enhanced segment embedding handling |
| `bert_encoder_with_MoE_without_hyperparameter_tuning.ipynb` | BERT with Mixture of Experts in FFN |
| `bert_encoder_with_MoE_without_hyperparameter_tuning_with_mask_aware_pooling.ipynb` | MoE + mask-aware pooling |
| `bert_encoder_with_MoE_with_BO_CV_hyperparameter_tuning.ipynb` | MoE with Bayesian Optimization & Cross-Validation |
| `bert_encoder_with_MoE_with_BO_CV_hyperparameter_tuning_mask_aware_mean_pooling.ipynb` | Full-featured with hyperparameter tuning |
| `encoder_final_with_query_and_chunk_handling.ipynb` | Final encoder optimized for RAG query/chunk processing |

### Cross-Encoder Reranking Variants

Located in `Cross_Encoder_Reranking/`:

| Script | Description |
|--------|-------------|
| `cross_encoder_reranking.ipynb` | Base cross-encoder with Grid Search and K-Fold CV |
| `cross_encoder_reranking_lora_fine_tuning_without_hyperparameter_tuning.ipynb` | LoRA fine-tuning without HP search |
| `cross_encoder_reranking_lora_finetuning_with_hyperparameter_tuning.ipynb` | LoRA with hyperparameter optimization |

---

## Setup & Installation

### Prerequisites
- Python >= 3.10
- CUDA-capable GPU (recommended) or CPU

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/Abhinav-gh/RAG_for_research_papers.git
cd RAG_for_research_papers/RAG_for_research_papers
```

2. **Create a virtual environment** (recommended):
```bash
# Using conda
conda env create -f env_fixed.yaml
conda activate <env_name>

# Or using pip
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

Or install core dependencies manually:
```bash
pip install torch transformers sentence-transformers chromadb
pip install langchain langchain-community pymupdf
pip install fastapi uvicorn pydantic
pip install gensim scikit-learn pandas numpy
pip install python-dotenv
```

4. **Set up environment variables**:
```bash
# Create .env file with your API keys
echo "GEMINI_API_KEY_1=your_api_key_here" >> .env
echo "GEMINI_API_KEY_2=your_backup_key_here" >> .env
```

---

## Project Structure

```
RAG_for_research_papers/
├── README.md                          # This file
├── group_2_ANLP_Project_All_Codes_Design_Document.pdf  # Project report
└── RAG_for_research_papers/
    ├── Chunking/                      # Document chunking strategies
    │   ├── First_chunking_attempt.ipynb
    │   └── Chunk_files/               # Saved chunk files (.pkl)
    │
    ├── Encoder/                       # BERT encoder implementations
    │   ├── encoder_only.ipynb
    │   ├── encoder_only_with_mask_aware_mean_pooling.ipynb
    │   ├── bert_encoder_with_MoE_*.ipynb
    │   └── script.py
    │
    ├── Encoder Fine Tuning/           # LoRA fine-tuning for encoder
    │   ├── lora_fine_tuning.ipynb
    │   └── script.py
    │
    ├── VectorDB/                      # ChromaDB setup and management
    │   ├── ChromaDB_quickstart.ipynb
    │   ├── ChromaDB_HP_with_BERT_embeddings.ipynb
    │   ├── ChromaDB_HP_with_Fine_tuned_BERT.ipynb
    │   ├── script.py
    │   └── chroma_Data*/              # Persistent ChromaDB storage
    │
    ├── Cross_Encoder_Reranking/       # Cross-encoder reranker
    │   ├── cross_encoder_reranking.ipynb
    │   ├── cross_encoder_reranking_lora_*.ipynb
    │   ├── dataset_generator_for_cross_encoder_reranking.ipynb
    │   └── crossenc_lora_out/         # Trained model weights
    │
    ├── LLM Caller/                    # LLM integration
    │   ├── gemini_API_caller_for_query_creation.ipynb
    │   ├── LLaMA_caller_for_query_creation.ipynb
    │   └── LlaMa_caller_script.py
    │
    ├── RAG_app/                       # Web application
    │   ├── backend/                   # FastAPI backend
    │   │   ├── app.py
    │   │   ├── chromadb_utils.py
    │   │   ├── encoder_model.py
    │   │   ├── gemini_answer.py
    │   │   └── requirements.txt
    │   ├── frontend/                  # Frontend application
    │   │   └── app.py
    │   └── docker-compose.yaml
    │
    ├── RAG Results/                   # Evaluation results
    │   └── multiquery_rag_results*.txt
    │
    ├── RAG_Pipeline.ipynb             # Main RAG pipeline notebook
    ├── requirements.txt
    └── env_fixed.yaml
```

---

## How to Train

### 1. Chunking

**Purpose**: Split PDF documents into semantically meaningful chunks for retrieval.

**Where to add files**: Place your PDF documents in the project root or specify the path in the chunking notebook.

**Steps**:
```bash
cd RAG_for_research_papers/Chunking
```

1. Open `First_chunking_attempt.ipynb`
2. Update the file path:
```python
file_path = "../your_document.pdf"
```
3. Configure chunking parameters:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,        # Characters per chunk
    chunk_overlap=200,     # Overlap between chunks
    separators=["\n\n", "\n", " ", ""]
)
```
4. Run the notebook to generate chunks
5. Chunks are saved to `Chunking/Chunk_files/` as `.pkl` files

**Chunking Types**:
- **Rule-based**: Using `RecursiveCharacterTextSplitter`
- **Semantic**: Embedding-based coherent segmentation

---

### 2. ChromaDB Setup

**Purpose**: Store document embeddings in a persistent vector database for efficient retrieval.

**Steps**:
```bash
cd RAG_for_research_papers/VectorDB
```

1. **Quick Start** (using pre-trained embeddings):
   - Open `ChromaDB_quickstart.ipynb`
   - Uses `sentence-transformers/all-MiniLM-L6-v2` by default

2. **With Custom BERT Embeddings**:
   - Open `ChromaDB_HP_with_BERT_embeddings.ipynb`
   - Configure paths:
```python
# Path to your chunk file
file_path = "../Chunking/Chunk_files/your_chunks.pkl"

# ChromaDB storage location
Relative_Database_path = "./chroma_Data_with_BERT_embeddings"

# Collection name
collection_name = "Your_Collection_Name"
```

3. **With Fine-tuned BERT**:
   - Open `ChromaDB_HP_with_Fine_tuned_BERT.ipynb`
   - Ensure fine-tuned model weights are available

4. **Initialize and populate**:
```python
# Initialize client
client = chromadb.PersistentClient(path=Absolute_Database_path)

# Create collection
collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_function,
    metadata={"description": "Your description"}
)

# Add documents in batches
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadatas
)
```

---

### 3. BERT Encoder Training

**Purpose**: Pre-train a custom BERT encoder on your domain corpus.

**Steps**:
```bash
cd RAG_for_research_papers/Encoder
```

1. Choose your encoder variant based on requirements:
   - Basic: `encoder_only.ipynb`
   - With MoE: `bert_encoder_with_MoE_without_hyperparameter_tuning.ipynb`
   - With HP tuning: `bert_encoder_with_MoE_with_BO_CV_hyperparameter_tuning.ipynb`

2. Configure hyperparameters:
```python
# Basic configuration
VOCAB_MIN_FREQ = 1
MAX_SEQ_LEN = 128
HIDDEN_SIZE = 768
NUM_LAYERS = 12
NUM_HEADS = 12
FFN_DIM = 3072
DROPOUT = 0.1
MLM_MASK_PROB = 0.15
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
```

3. Prepare training data:
   - Load your corpus text
   - Build vocabulary
   - Create training dataset

4. Run training:
```python
# Training will perform:
# - Masked Language Modeling (MLM)
# - Next Sentence Prediction (NSP)
```

5. Save model:
```python
torch.save(model.state_dict(), "bert_encoder_weights.pt")
```

---

### 4. Encoder Fine-Tuning (LoRA)

**Purpose**: Fine-tune the pre-trained BERT encoder for query-chunk relevance using Low-Rank Adaptation.

**Steps**:
```bash
cd "RAG_for_research_papers/Encoder Fine Tuning"
```

1. Open `lora_fine_tuning.ipynb`

2. Prepare contrastive dataset:
   - CSV with columns: `query`, `positive_chunk_id`
   - Negative samples are drawn from ChromaDB

3. Configure LoRA parameters:
```python
LORA_RANK = 8
LORA_ALPHA = 8
LORA_DROPOUT = 0.1
```

4. Training uses contrastive loss:
   - Brings query embeddings closer to positive chunks
   - Pushes away from negative chunks

---

### 5. Cross-Encoder Reranking

**Purpose**: Train a cross-encoder to rerank retrieved chunks for improved accuracy.

**Steps**:
```bash
cd RAG_for_research_papers/Cross_Encoder_Reranking
```

1. **Generate training data**:
   - Open `dataset_generator_for_cross_encoder_reranking.ipynb`
   - Creates query-chunk pairs with relevance labels

2. **Train cross-encoder**:
   - Open `cross_encoder_reranking_lora_finetuning_with_hyperparameter_tuning.ipynb`
   - Uses LoRA for parameter-efficient fine-tuning
   - Grid Search with 10-Fold Cross-Validation

3. Configure training:
```python
# Hyperparameter search space
learning_rates = [2e-5, 3e-5, 5e-5]
dropout_rates = [0.1, 0.2]
```

4. Model saved to `crossenc_lora_out/`

---

## Running the RAG Application

### Using Docker (Recommended)

```bash
cd RAG_for_research_papers/RAG_app
docker-compose up --build
```

Access the API at `http://localhost:8000`

### Manual Setup

1. **Start the backend**:
```bash
cd RAG_for_research_papers/RAG_app/backend
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

2. **API Endpoints**:
   - `GET /health` - Health check
   - `GET /model_info` - Model and tokenizer info
   - `POST /query` - Run RAG query
   - `POST /evaluate` - Run RAGAS evaluation

3. **Example Query**:
```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={
        "query": "What is the main theme?",
        "top_k": 10,  # Number of chunks to retrieve
        "top_m": 3    # Number of chunks after reranking
    }
)
print(response.json())
```

---

## Evaluation

The project uses **RAGAS** (Retrieval Augmented Generation Assessment) metrics:

| Metric | Description |
|--------|-------------|
| **Answer Relevancy** | Does the answer address the question? |
| **Faithfulness** | Is the answer grounded in retrieved context? |
| **Contextual Precision** | How much of the answer uses the context? |
| **Contextual Recall** | How much of reference answer is covered? |
| **Contextual Relevancy** | Is the retrieved context useful? |

Run evaluation:
```bash
cd RAG_for_research_papers/RAG_app/backend
python ragas_eval.py
```

---

## References

1. Devlin, J., et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." NAACL, 2019.
2. Reimers, N. & Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks." EMNLP, 2019.
3. ChromaDB Documentation: https://docs.trychroma.com
4. Johnson, J., et al. "Billion-Scale Similarity Search with GPUs." IEEE Transactions on Big Data, 2019.

---

## License

This project is for educational purposes as part of an ANLP course project.
