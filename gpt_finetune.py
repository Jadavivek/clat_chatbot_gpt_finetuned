from datasets import Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import json

# Load CLAT data
with open('clat_data.json', 'r') as f:
    raw_data = json.load(f)

# Preprocess into question + answer format
dataset = Dataset.from_dict({
    "text": [f"Question: {item['question']}\nAnswer: {item['answer']}" for item in raw_data]
})

# Tokenizer & model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=256)

tokenized_data = dataset.map(tokenize, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./clat_gpt_model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
)

# Train
trainer.train()

# Save fine-tuned model
trainer.save_model("./clat_gpt_model")
tokenizer.save_pretrained("./clat_gpt_model")
