import torch
import torch.nn as nn
import transformers
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import torch.optim as optim
from transformers import get_scheduler
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
import bitsandbytes as bnb
from accelerate import Accelerator

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=6
)

df = pd.read_csv("train.csv")

X = df["comment_text"]
y = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

model = get_peft_model(base_model, peft_config)

accelerator = Accelerator()
optimizer = bnb.optim.AdamW8bit(model.parameters(), min_8bit_size=16384)

if torch.cuda.is_available():
    device = torch.device("cuda")
    model.to(device)


# Tokenize the text
def tokenize_text(texts, max_length=128):
    return tokenizer(
        texts.tolist(),  # Convert pandas Series to list
        padding=True,  # Pad to max_length
        truncation=True,  # Truncate to max_length
        max_length=max_length,
        return_tensors="pt",  # Return PyTorch tensors
    )


# Tokenize the input text
tokenized_texts = tokenize_text(X)

dataset = ToxicCommentDataset(tokenized_texts, y)

batch_size = 64
data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)

# configuring hf accelerate
data_loader, model, optimizer = accelerator.prepare(data_loader, model, optimizer)

epochs = 1
training_steps = epochs * len(data_loader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=training_steps,
)

progress_bar = tqdm(range(training_steps))


# training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    total_loss = 0

    model.train()
    for batch in data_loader:
        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()  
        optimizer.zero_grad()

        progress_bar.update(1)

        total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

progress_bar.close()
