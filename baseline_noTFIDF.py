import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import KFold, cross_val_score

# Import and execute your aggressive cleaning pipeline
# This ensures datasets/train_features.csv, dev_features.csv, and test_features.csv are freshly generated
from linguistic_features_aggressive import run_master_clean

run_master_clean()

# 1. Load the generated features dataframes
train_df = pd.read_csv("datasets/train_features.csv")
dev_df   = pd.read_csv("datasets/dev_features.csv")
test_df  = pd.read_csv("datasets/test_features.csv")

label_col    = "label"
feature_cols = ["sentiment_textblob", "sentiment_vader", "gunning_fog", "lexical_diversity"]

# Fill any residual NaNs in linguistic features just in case
train_df[feature_cols] = train_df[feature_cols].fillna(0)
dev_df[feature_cols]   = dev_df[feature_cols].fillna(0)
test_df[feature_cols]  = test_df[feature_cols].fillna(0)

# 2. Extract ONLY the 4 linguistic features (Completely eliminating TF-IDF)
X_train = train_df[feature_cols].values
X_dev   = dev_df[feature_cols].values
X_test  = test_df[feature_cols].values

y_train = train_df[label_col]
y_dev   = dev_df[label_col]
y_test  = test_df[label_col]

# 3. GridSearch optimization using the Random Forest classifier
param_grid = {
    "n_estimators": [100, 200], 
    "max_depth":    [10, 20, 50], 
    "min_samples_split": [2, 5], 
    "max_features": ["sqrt", "log2"],
    "max_leaf_nodes": [100, 200]
}

rf = RandomForestClassifier(random_state=42, n_jobs=1)

grid_search = GridSearchCV(
    rf,
    param_grid, 
    scoring="precision",  
    cv=3,
    verbose=1,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nBest params:", grid_search.best_params_)
print("Best precision (CV):", round(grid_search.best_score_, 4))

# 4. Model Evaluation
best_rf = grid_search.best_estimator_

y_pred      = best_rf.predict(X_test)
y_pred_prob = best_rf.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
auroc     = roc_auc_score(y_test, y_pred_prob)

print("\n=== TEST SET RESULTS ===")
print(f"Precision : {precision:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUROC     : {auroc:.4f}")
print("\nDetailed report:")
print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))

y_dev_pred = best_rf.predict(X_dev)
dev_precision = precision_score(y_dev, y_dev_pred)

print("\n=== DEV SET RESULTS ===")
print(f"Dev Precision: {dev_precision:.4f}")
print("\nDev Classification Report:")
print(classification_report(y_dev, y_dev_pred, target_names=["Real", "Fake"]))