import os

def health_check():
    return {"status": "ok"}

def get_model_info():
    return {
        "model_path": os.path.abspath("../../Cross Encoder Reranking/crossenc_lora_out/model_with_lora.pt"),
        "tokenizer_path": os.path.abspath("../../Cross Encoder Reranking/crossenc_lora_out")
    }
