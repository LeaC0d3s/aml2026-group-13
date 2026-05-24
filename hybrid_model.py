import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from sklearn.metrics import precision_score, f1_score, roc_auc_score

# custom dataset matching Hugging Face expectations
class HybridFakeNewsDataset(Dataset):
    def __init__(self, csv_path, tokenizer, max_len=512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # match feature columns from linguisticfeatures.py
        self.feature_cols = ["sentiment_textblob", "sentiment_vader", "gunning_fog", "lexical_diversity"]
        
        # handle any stray NaNs just in case
        self.df["text_clean"] = self.df["text_clean"].fillna("")
        self.df[self.feature_cols] = self.df[self.feature_cols].fillna(0)
        
        self.texts = self.df["text_clean"].values
        self.linguistic_features = self.df[self.feature_cols].values.astype(np.float32)
        self.labels = self.df["label"].values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "ling_features": torch.tensor(self.linguistic_features[idx], dtype=torch.float32),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# novel architecture: Gated Attention Fusion Network
class GatedHybridClassifier(nn.Module):
    def __init__(self, transformer_model_name="distilbert-base-uncased", ling_dim=4, drop_out_rate=0.1):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(transformer_model_name)
        self.hidden_dim = self.transformer.config.hidden_size # 768 for DistilBERT
        
        self.ling_projection = nn.Sequential(
            nn.Linear(ling_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU()
        )
        
        # this outputs a value between 0 and 1 for each dimension, acting as an attention mask over the linguistic features conditioned on the contextual vector
        self.gate_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(drop_out_rate),
            nn.Linear(self.hidden_dim // 2, 2)
        )
        
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, ling_features, labels=None):
        # extract contextual embedding from DistilBERT
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = transformer_outputs.last_hidden_state[:, 0, :]
        
        # project linguistic features
        ling_embedded = self.ling_projection(ling_features)
        
        # compute dynamic Gate based on combined spaces
        combined = torch.cat([cls_embedding, ling_embedded], dim=-1)
        gate = self.gate_layer(combined)
        
        # apply gate for controlled information fusion -> the model balances raw text context with stylometric markers dynamically
        fused_vector = gate * cls_embedding + (1 - gate) * ling_embedded
        
        # final classification
        logits = self.classifier(fused_vector)
        
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels)
            
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

# evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    # extract probabilities for positive class (fake news = 1)
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    
    return {
        "precision": precision_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "auroc": roc_auc_score(labels, probs)
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running training on: {device}")
    
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = HybridFakeNewsDataset("datasets/train_features.csv", tokenizer)
    dev_dataset = HybridFakeNewsDataset("datasets/dev_features.csv", tokenizer)
    test_dataset = HybridFakeNewsDataset("datasets/test_features.csv", tokenizer)
    
    # the custom model
    model = GatedHybridClassifier(transformer_model_name=model_name, ling_dim=4)
    
    # parameters for training 
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="precision",
        greater_is_better=True,
        logging_dir="./logs",
        logging_steps=50,
        report_to="none"
    )
    
    # use Hugging Face Trainer API for smooth parameter processing
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics
    )
    
    print("Started Hybrid Model Training")

    trainer.train()
    test_results = trainer.evaluate(test_dataset)
    
    print("\n=== FINAL HYBRID MODEL TEST SET RESULTS ===")
    print(f"Test Precision : {test_results['eval_precision']:.4f}")
    print(f"Test F1 Score  : {test_results['eval_f1']:.4f}")
    print(f"Test AUROC     : {test_results['eval_auroc']:.4f}")

if __name__ == "__main__":
    main()