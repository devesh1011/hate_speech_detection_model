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
import pandas as pd
from peft import LoraConfig, TaskType, get_peft_model
import bitsandbytes as bnb
from accelerate import Accelerator
from evaluate import load


tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
base_model = AutoModelForSequenceClassification.from_pretrained(
    "google-bert/bert-base-cased", num_labels=6
)


train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("data/raw/test.csv")

X = train_df["comment_text"]
y = train_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]]

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
tokenized_train_texts = tokenize_text(X)
tokenized_test_texts = tokenize_text(test_df["comment_text"])

train_dataset = ToxicCommentDataset(tokenized_train_texts, y)
test_dataset = ToxicCommentDataset(
    tokenized_test_texts,
    test_df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]],
)


batch_size = 64
train_data_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True
)
test_data_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

# configuring hf accelerate
train_data_loader, test_data_loader, model, optimizer = accelerator.prepare(
    train_data_loader, test_data_loader, model, optimizer
)

epochs = 1
training_steps = epochs * len(train_data_loader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=training_steps,
)

progress_bar = tqdm(range(training_steps))

metric = load("accuracy")
# training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    total_loss = 0

    model.train()
    for batch in train_data_loader:
        outputs = model(**batch)
        loss = outputs.loss

        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        progress_bar.update(1)

        total_loss += loss.item()

    avg_loss = total_loss / len(train_data_loader)
    print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    # Evaluate the model
    model.eval()
    for batch in test_data_loader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = torch.argmax(outputs.logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    # Compute the final accuracy
    final_score = metric.compute()
    print(f"Accuracy: {final_score['accuracy']}")

progress_bar.close()
