

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import sys, os
import warnings
warnings.filterwarnings("ignore")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.trainer import TextClassifier

print("Starting training script...")

# Load dataset + tokenize
dataset = load_dataset("imdb")
classifier = TextClassifier("distilbert-base-uncased")

def preprocess(batch):
    return classifier.tokenizer(batch["text"], truncation=True)

tokenized = dataset.map(preprocess, batched=True, remove_columns=["text"])
collator = DataCollatorWithPadding(tokenizer=classifier.tokenizer)

train_loader = DataLoader(tokenized["train"], shuffle=True, batch_size=16, collate_fn=collator)
eval_loader = DataLoader(tokenized["test"], batch_size=16, collate_fn=collator)

# Train + evaluate
losses = classifier.train(train_loader, eval_loader, num_epochs=2)
classifier.save("/Models/IMDB")


