import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

# -----------------------------
# CONFIG
# -----------------------------
BASE_MODEL = "google/flan-t5-small"
DATA_PATH = "data/train.jsonl"
OUT_DIR = "models/techbot-flan-t5-small-lora"

MAX_SOURCE_LEN = 256
MAX_TARGET_LEN = 256

def build_prompt(instruction: str) -> str:
    return f"Instruction: {instruction}\nAnswer:"

def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    print("🔄 Loading dataset...")
    ds = load_dataset("json", data_files=DATA_PATH, split="train")

    print("🔄 Loading tokenizer/model (CPU)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    # LoRA config (CPU-friendly)
    lora = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q", "v"],  # T5 attention projections
    )
    model = get_peft_model(model, lora)
    model.to("cpu")

    def tokenize(batch):
        source = [build_prompt(x) for x in batch["instruction"]]
        target = batch["output"]

        model_inputs = tokenizer(
            source,
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding="max_length",
        )

        labels = tokenizer(
            target,
            max_length=MAX_TARGET_LEN,
            truncation=True,
            padding="max_length",
        )["input_ids"]

        # mask padding for loss
        labels = [
            [(t if t != tokenizer.pad_token_id else -100) for t in seq]
            for seq in labels
        ]
        model_inputs["labels"] = labels
        return model_inputs

    print("🔄 Tokenizing...")
    tokenized = ds.map(tokenize, batched=True, remove_columns=ds.column_names)

    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # CPU Training args
    args = TrainingArguments(
        output_dir=OUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=6,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=collator,
    )

    print("🚀 Training...")
    trainer.train()

    print("💾 Saving adapter + tokenizer...")
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    print(f"✅ Done. Saved to: {OUT_DIR}")

if __name__ == "__main__":
    main()
