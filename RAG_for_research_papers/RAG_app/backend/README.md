# RAG Backend

## Endpoints
- `/health`: Health check
- `/model_info`: Model and tokenizer info
- `/query`: Run cross-encoder reranking (POST: {query, chunk})
- `/evaluate`: Run RAGAS evaluation (POST: {predictions, references})

## Usage
- Build and run with Docker Compose:
  ```bash
  docker-compose up --build
  ```
- Access API at `http://localhost:8000`

## Requirements
- Model weights and tokenizer must be present in `../../Cross Encoder Reranking/crossenc_lora_out/`
