import torch
from transformers import BertTokenizerFast, AutoModel
from transformers import AutoModelForSequenceClassification

import os

ENCODER_MODEL = "bert-base-uncased"   # HF model name
ENCODER_WEIGHTS = "../Encoder_Fine_Tuning/lora_finetuned/lora_bert.pt"

class BertEncoder:
    def __init__(self, model_name=ENCODER_MODEL, model_path=ENCODER_WEIGHTS, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        abs_model_path = os.path.abspath(model_path)

        print(">>> Loading BERT encoder")
        print(">>> HF Model:", model_name)
        print(">>> Weights:", abs_model_path)

        # Load tokenizer directly from HuggingFace (NO local path!)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)

        # Load base BERT model from HF and apply your finetuned weights
        self.model = AutoModel.from_pretrained(model_name)

        # Load your LoRA / finetuned weights
        state_dict = torch.load(abs_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def embed(self, text):
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]

        return cls_emb
