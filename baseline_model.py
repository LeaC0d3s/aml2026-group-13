import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, f1_score, roc_auc_score, classification_report
from scipy.sparse import hstack, csr_matrix

# csv with features
train_df = pd.read_csv("train_features.csv")
dev_df   = pd.read_csv("dev_features.csv")
test_df  = pd.read_csv("test_features.csv")

text_col     = "text_clean"
label_col    = "label"
feature_cols = ["sentiment_textblob", "sentiment_vader", "gunning_fog", "lexical_diversity"]

# TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2)) # change max_features if required

X_train_tfidf = tfidf.fit_transform(train_df[text_col])  # fit + transform
X_dev_tfidf   = tfidf.transform(dev_df[text_col])         # solo transform
X_test_tfidf  = tfidf.transform(test_df[text_col])        # solo transform

# Linguistic features transfomed to sparse matrix
X_train_ling = csr_matrix(train_df[feature_cols].values) # we can remove features and do ablation studies if we want
X_dev_ling   = csr_matrix(dev_df[feature_cols].values)
X_test_ling  = csr_matrix(test_df[feature_cols].values)

# concatenate TF-IDF feature with linguistc features
X_train = hstack([X_train_tfidf, X_train_ling]) # remove features completely and do ablation doing x_train= x_train_tfidf
X_dev   = hstack([X_dev_tfidf,   X_dev_ling])
X_test  = hstack([X_test_tfidf,  X_test_ling])

y_train = train_df[label_col]
y_dev   = dev_df[label_col]
y_test  = test_df[label_col]

# gridsearch with RF (it should find the best parameters for the model)
param_grid = {
    "n_estimators": [100, 200], # higher slower but more accurate
    "max_depth":    [10, 20, 50], 
    "min_samples_split": [2, 5], # higher could prevent overfitting but also reduce performance
    "max_features": ["sqrt", "log2"],
    "max_leaf_nodes": [100, 200]
    
}

rf = RandomForestClassifier(random_state=42, n_jobs= 1)

grid_search = GridSearchCV(
    rf,
    param_grid,
    n_iter = 15, # number of combinations to try 
    scoring="precision",  
    cv=3,
    verbose=1,
    n_jobs = -1
    
)

grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
print("Best precision (CV):", round(grid_search.best_score_, 4))

# Evaluation
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