import os, sys, json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import LabelEncoder

MODEL_NAME   = "vinai/phobert-base-v2"
DATA_PATH    = "data/processed/clean_data.csv"
MODEL_DIR    = "models"
BATCH_SIZE   = 16
MAX_LEN      = 128
EPOCHS       = 3
LR           = 2e-5
RANDOM_STATE = 42

os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


if __name__ == '__main__':
    try:
        # Step 1: Load and process data FIRST (before any torch import)
        df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
        df = df.dropna(subset=["text_clean", "label"])
        df = df[df["text_clean"].str.strip() != ""].reset_index(drop=True)
        df["text_clean"] = df["text_clean"].astype(str).str.strip()
        print(f"[phobert] Loaded {len(df)} rows"); sys.stdout.flush()

        le = LabelEncoder()
        y  = le.fit_transform(df["label"])
        label_mapping = {int(i): str(c) for i, c in enumerate(le.classes_)}
        print(f"[phobert] Classes: {label_mapping}"); sys.stdout.flush()

        X_train, X_tmp, y_train, y_tmp = train_test_split(
            df["text_clean"].tolist(), y,
            test_size=0.30, random_state=RANDOM_STATE, stratify=y)
        X_val, X_test, y_val, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_tmp)
        print(f"[phobert] Split done"); sys.stdout.flush()

        # Step 2: NOW import torch and transformers
        import torch
        from torch.utils.data import Dataset, DataLoader
        from torch.optim import AdamW
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from transformers import get_linear_schedule_with_warmup
        print("[phobert] torch imported"); sys.stdout.flush()

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[phobert] Device: {DEVICE}"); sys.stdout.flush()

        # Step 3: Define dataset class and functions here (after torch import)
        class SentimentDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                enc = self.tokenizer(
                    self.texts[idx],
                    max_length=self.max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                return {
                    "input_ids":      enc["input_ids"].squeeze(),
                    "attention_mask": enc["attention_mask"].squeeze(),
                    "label":          torch.tensor(self.labels[idx], dtype=torch.long),
                }

        def train_epoch(model, loader, optimizer, scheduler, device):
            model.train()
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                out = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["label"].to(device),
                )
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += out.loss.item()
            return total_loss / len(loader)

        def eval_epoch(model, loader, device):
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for batch in loader:
                    out = model(
                        input_ids=batch["input_ids"].to(device),
                        attention_mask=batch["attention_mask"].to(device),
                    )
                    preds.extend(torch.argmax(out.logits, dim=1).cpu().numpy())
                    trues.extend(batch["label"].numpy())
            return f1_score(trues, preds, average="macro"), preds, trues

        # Step 4: Load model and train
        print(f"[phobert] Loading {MODEL_NAME}..."); sys.stdout.flush()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
                    MODEL_NAME, num_labels=3).to(DEVICE)
        print("[phobert] Model on GPU OK"); sys.stdout.flush()

        train_loader = DataLoader(SentimentDataset(X_train, y_train, tokenizer, MAX_LEN),
                                  batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader   = DataLoader(SentimentDataset(X_val,   y_val,   tokenizer, MAX_LEN),
                                  batch_size=BATCH_SIZE, num_workers=0)
        test_loader  = DataLoader(SentimentDataset(X_test,  y_test,  tokenizer, MAX_LEN),
                                  batch_size=BATCH_SIZE, num_workers=0)
        print("[phobert] DataLoaders ready"); sys.stdout.flush()

        optimizer   = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = len(train_loader) * EPOCHS
        scheduler   = get_linear_schedule_with_warmup(
                          optimizer,
                          num_warmup_steps=total_steps // 10,
                          num_training_steps=total_steps)

        best_val_f1, best_epoch = 0, 0
        for epoch in range(1, EPOCHS + 1):
            loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
            val_f1, _, _ = eval_epoch(model, val_loader, DEVICE)
            print(f"[phobert] Epoch {epoch}/{EPOCHS} | loss={loss:.4f} | val_f1={val_f1:.4f}"); sys.stdout.flush()
            if val_f1 > best_val_f1:
                best_val_f1, best_epoch = val_f1, epoch
                os.makedirs(os.path.join(MODEL_DIR, "phobert"), exist_ok=True)
                model.save_pretrained(os.path.join(MODEL_DIR, "phobert"))
                tokenizer.save_pretrained(os.path.join(MODEL_DIR, "phobert"))
                print(f"[phobert] Saved best model at epoch {epoch}"); sys.stdout.flush()

        model = AutoModelForSequenceClassification.from_pretrained(
                    os.path.join(MODEL_DIR, "phobert")).to(DEVICE)
        test_f1, test_preds, test_trues = eval_epoch(model, test_loader, DEVICE)
        print(f"\n[phobert] Test F1-macro: {test_f1:.4f}"); sys.stdout.flush()
        print(classification_report(test_trues, test_preds, target_names=le.classes_))

        with open(os.path.join(MODEL_DIR, "phobert_config.json"), "w") as f:
            json.dump({"model": "PhoBERT", "feature": MODEL_NAME,
                       "f1_macro_test": round(test_f1, 4),
                       "best_epoch": best_epoch}, f, indent=2)
        with open(os.path.join(MODEL_DIR, "phobert_label_mapping.json"), "w") as f:
            json.dump(label_mapping, f, indent=2)
        print("[phobert] Done."); sys.stdout.flush()

    except BaseException:
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
