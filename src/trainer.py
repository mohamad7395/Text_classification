
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_scheduler,
    DataCollatorWithPadding,
)
from tqdm.auto import tqdm
import evaluate


class TextClassifier:
    def __init__(self, model_name="distilbert-base-uncased", num_labels=2, device=None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Default optimizer/scheduler placeholders (set in train)
        self.optimizer = None
        self.lr_scheduler = None

        # Metric (basic accuracy, can extend outside class)
        self.metric = evaluate.load("accuracy")

    def train(
        self,
        train_dataloader,
        eval_dataloader=None,
        num_epochs=2,
        learning_rate=2e-5,
        max_grad_norm=1.0,
        log_interval=50,
    ):
        """Custom training loop"""
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = num_epochs * len(train_dataloader)
        self.lr_scheduler = get_scheduler(
            "linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )

        progress_bar = tqdm(range(total_steps))
        train_losses, step_losses = [], []

        for epoch in range(num_epochs):
            self.model.train()
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                step_losses.append(loss.item())
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                progress_bar.update(1)

                if (step + 1) % log_interval == 0:
                    avg_loss = np.mean(step_losses)
                    train_losses.append(avg_loss)
                    step_losses = []

            if eval_dataloader is not None:
                acc = self.evaluate(eval_dataloader)
                print(f"Epoch {epoch+1}/{num_epochs} | Validation Accuracy: {acc:.4f}")

        return train_losses

    def evaluate(self, dataloader):
        """Run evaluation and return accuracy"""
        self.model.eval()
        preds, labels = [], []
        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            labels.extend(batch["labels"].cpu().numpy())
        result = self.metric.compute(predictions=np.array(preds), references=np.array(labels))
        return result["accuracy"]

    def predict(self, texts, batch_size=16):
        """Predict labels for a list of raw texts"""
        self.model.eval()
        preds = []
        for i in range(0, len(texts), batch_size):
            encodings = self.tokenizer(
                texts[i:i+batch_size], truncation=True, padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(**encodings)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
        return preds
    
    def predict_dataset(self, dataset, batch_size=16, return_df=True):
        """Predict on a Hugging Face dataset (with 'text' and 'label' fields)."""
        import pandas as pd

        texts, labels, preds, probs = [], [], [], []

        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            texts.extend(batch["text"])
            labels.extend(batch["label"])

            # Tokenize batch
            encodings = self.tokenizer(batch["text"], padding=True, truncation=True, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model(**encodings)
                logits = outputs.logits
                batch_probs = torch.softmax(logits, dim=-1)[:, 1]  
                batch_preds = torch.argmax(logits, dim=-1)

            preds.extend(batch_preds.cpu().numpy())
            probs.extend(batch_probs.cpu().numpy())

        if return_df:
            return pd.DataFrame({
                "text": texts,
                "label": labels,
                "pred": preds,
                "prob_pos": probs
            })
        return preds


    def save(self, path="./sentiment_model"):
        """Save model + tokenizer"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    @classmethod
    def load(cls, path, device=None):
        """Load model + tokenizer from disk"""
        obj = cls.__new__(cls)  # bypass __init__
        obj.tokenizer = AutoTokenizer.from_pretrained(path)
        obj.model = AutoModelForSequenceClassification.from_pretrained(path)
        obj.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        obj.model.to(obj.device)
        obj.metric = evaluate.load("accuracy")
        return obj
