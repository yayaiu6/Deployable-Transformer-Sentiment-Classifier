import torch
from transformers import BertTokenizer, PretrainedConfig
from .model_classes import TransformerModel, Sentiment140Config

def load_model_and_tokenizer(model_path, tokenizer_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
    config = Sentiment140Config.from_pretrained(tokenizer_path)
    model = TransformerModel(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, tokenizer

def predict(text, model, tokenizer, device="cuda" if torch.cuda.is_available() else "cpu"):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        logits = model(input_ids, mask=attention_mask)
        pred = torch.argmax(logits, dim=-1).item()
    class_names = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return class_names[pred]