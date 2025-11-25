import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from peft import LoraConfig, get_peft_model
import os

BASE_MODEL = "bert-base-uncased"
MODEL_PATH = "../Cross_Encoder_Reranking/crossenc_lora_out/model_with_lora.pt"
TOKENIZER_PATH = "../Cross_Encoder_Reranking/crossenc_lora_out"

LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.0

print(">>> tokenizer_path (raw):", TOKENIZER_PATH)
print(">>> tokenizer_path (abs):", os.path.abspath(TOKENIZER_PATH))
print(">>> vocab exists:", os.path.exists(os.path.join(TOKENIZER_PATH, "vocab.txt")))
print(">>> tokenizer.json exists:", os.path.exists(os.path.join(TOKENIZER_PATH, "tokenizer.json")))

class CrossEncoderLoRAWrapper:
    def __init__(self,
                 base_model=BASE_MODEL,
                 model_path=MODEL_PATH,
                 tokenizer_path=TOKENIZER_PATH,
                 device=None):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")

        # SAFEST FIX: convert EVERYTHING to absolute paths
        model_path = os.path.abspath(model_path)
        tokenizer_path = os.path.abspath(tokenizer_path)

        print("[DEBUG] Tokenizer path:", tokenizer_path)
        print("[DEBUG] Model path:", model_path)

        # Load tokenizer LOCALLY ONLY
        from transformers import BertTokenizerFast

        # self.tokenizer = BertTokenizerFast(
        #     vocab_file=os.path.join(tokenizer_path, "vocab.txt"),
        #     tokenizer_file=os.path.join(tokenizer_path, "tokenizer.json")
        # )
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)

        # Load base encoder (from HF repo name ONLY)
        # Load base encoder (same class used during training)
        encoder = AutoModel.from_pretrained(
            base_model,
            local_files_only=False
        )


        # Recreate LoRA config
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="FEATURE_EXTRACTION",   # <--- THE FIX
            target_modules=["query", "key", "value", "dense"]
        )

        # Inject LoRA layers
        encoder = get_peft_model(encoder, lora_config)

        # Classifier head
        hidden_size = encoder.config.hidden_size
        classifier = nn.Linear(hidden_size, 1)

        # Wrap into full model identical to training
        class CrossEncoderModel(nn.Module):
            def __init__(self, encoder, classifier):
                super().__init__()
                self.encoder = encoder
                self.classifier = classifier
                self.dropout = nn.Dropout(0.1)

            def forward(self, input_ids, attention_mask=None, token_type_ids=None):
                outputs = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    return_dict=True,
                )
                cls_emb = outputs.last_hidden_state[:, 0, :]
                x = self.dropout(cls_emb)
                logits = self.classifier(x).squeeze(-1)
                return logits, cls_emb

        self.model = CrossEncoderModel(encoder, classifier)

        # Load REAL .pt checkpoint using absolute path
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict, strict=False)

        self.model.to(self.device)
        self.model.eval()

    def predict_batch(self, query, chunks):
        pairs = [(query, c) for c in chunks]

        inputs = self.tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            logits, _ = self.model(**inputs)
            probs = torch.sigmoid(logits).cpu().numpy().tolist()

        return probs

