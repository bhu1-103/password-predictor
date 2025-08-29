import torch
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import (
    PreTrainedTokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

def build_char_tokenizer(dataset_path):
    tokenizer = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")

    trainer = trainers.WordLevelTrainer(
        special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
    )

    tokenizer.train([dataset_path], trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

DATASET_PATH = "./rockyou-75.txt"
OUTPUT_DIR = "./model-v1"

tokenizer = build_char_tokenizer(DATASET_PATH)

config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_positions=32,
    n_embd=512,
    n_layer=8,
    n_head=4
)
model = GPT2LMHeadModel(config=config)

print("Loading dataset...")
dataset = load_dataset("text", data_files={"train": DATASET_PATH})

def tokenize_function(examples):
    return tokenizer(
        [f"<bos>{word}<eos>" for word in examples["text"]],
        truncation=True,
        max_length=32,
        padding="max_length"
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

tokenized_dataset = tokenized_dataset.map(
    lambda batch: {"labels": batch["input_ids"]}, batched=True
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=32,
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)

print("Starting training...")
trainer.train()

print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"Model and tokenizer saved to {OUTPUT_DIR}")
