import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_score, f1_score, roc_auc_score, classification_report 

#Note: in some cases it might throw an error if you already have a folder called "datasets" becuase the paths get mixed up internally, so just rename that folder and it should work.
class FakeNewsTextDataset(Dataset):
    """
    Standard text-only dataset for DistilBERT.

    This uses the same CSV files as the hybrid model, but ignores the
    handcrafted linguistic feature columns.
    """

    def __init__(self, csv_path, tokenizer, max_len=512):
        self.df = pd.read_csv(csv_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.df["text_clean"] = self.df["text_clean"].fillna("")

        self.texts = self.df["text_clean"].values
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
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }

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

    #define model name
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    #process datasets for model input
    train_dataset = FakeNewsTextDataset("datasets_agressive/train_features.csv", tokenizer)
    dev_dataset = FakeNewsTextDataset("datasets_agressive/dev_features.csv", tokenizer)
    test_dataset = FakeNewsTextDataset("datasets_agressive/test_features.csv", tokenizer)
    
    # the custom model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
    )    
    # parameters for training 
    training_args = TrainingArguments(
        output_dir="./results_distilbert_baseline",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        eval_strategy="epoch", #version dependant variable name: "evaluation_strategy" -> "eval_strategy"
        save_strategy="epoch",
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="precision",
        greater_is_better=True,
        logging_dir="./logs_distilbert_baseline",
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
    
    print("Started DistilBERT Baseline Training")

    trainer.train()
    test_results = trainer.evaluate(test_dataset)
    
    print("\n=== FINAL Vanilla DistilBERT MODEL TEST SET RESULTS ===")
    print(f"Test Precision : {test_results['eval_precision']:.4f}")
    print(f"Test F1 Score  : {test_results['eval_f1']:.4f}")
    print(f"Test AUROC     : {test_results['eval_auroc']:.4f}")

    # Full classification report on the test set
    test_predictions = trainer.predict(test_dataset)

    logits = test_predictions.predictions
    labels = test_predictions.label_ids
    preds = np.argmax(logits, axis=1)

    print("\n=== FULL CLASSIFICATION REPORT ===")
    print(
    classification_report(
        labels,
        preds,
        target_names=["Real", "Fake"],
        digits=4,
        zero_division=0,
        )
    )

if __name__ == "__main__":
    main()
