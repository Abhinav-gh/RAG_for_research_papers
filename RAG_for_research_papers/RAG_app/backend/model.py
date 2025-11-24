import torch
from transformers import AutoTokenizer

MODEL_PATH = "../../Cross Encoder Reranking/crossenc_lora_out/model_with_lora.pt"
TOKENIZER_PATH = "../../Cross Encoder Reranking/crossenc_lora_out"

class CrossEncoderLoRAWrapper:
    def __init__(self, model_class, model_name_or_path=TOKENIZER_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = model_class(model_name_or_path)
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, query, chunk):
        inputs = self.tokenizer(query, chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        with torch.no_grad():
            logits, _ = self.model(**inputs)
            prob = torch.sigmoid(logits).cpu().numpy().tolist()
        return prob
