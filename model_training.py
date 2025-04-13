from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json

# Load and preprocess the dataset
with open('clat_data.json', 'r') as file:
    data = json.load(file)

dataset = Dataset.from_dict({
    'question': [item['question'] for item in data],
    'answer': [item['answer'] for item in data]
})

# Format for GPT fine-tuning (concatenating question and answer)
def preprocess_function(examples):
    return {'text': [f"Question: {q}\nAnswer: {a}" for q, a in zip(examples['question'], examples['answer'])]}

dataset = dataset.map(preprocess_function, batched=True)

# Load pre-trained GPT-2 model
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=['question', 'answer'])

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=10_000,
    logging_steps=500,
    save_total_limit=2,
)

# Setup Trainer
trainer = Trainer(
    model=model,
    args=training
