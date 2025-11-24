import torch
from transformers import AutoTokenizer, AutoModel

ENCODER_MODEL_PATH = "../../Encoder Fine Tuning/lora_finetuned/lora_bert.pt"
ENCODER_NAME = "bert-base-uncased"

class BertEncoder:
    def __init__(self, model_name=ENCODER_NAME, model_path=ENCODER_MODEL_PATH, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        for k in inputs:
            inputs[k] = inputs[k].to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return emb
