import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

BASE = "google/flan-t5-small"
ADAPTER = "models/techbot-flan-t5-small-lora"

tok = AutoTokenizer.from_pretrained(BASE)
base_model = AutoModelForSeq2SeqLM.from_pretrained(BASE)
model = PeftModel.from_pretrained(base_model, ADAPTER)
model.eval()

prompt = "Instruction: What's going on with NVIDIA?\nAnswer:"
inputs = tok(prompt, return_tensors="pt")

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=180)

print(tok.decode(out[0], skip_special_tokens=True))
