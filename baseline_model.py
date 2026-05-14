import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, f1_score, roc_auc_score, classification_report
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import KFold, cross_val_score

# csv with features
train_df = pd.read_csv("datasets/train_features.csv")
dev_df   = pd.read_csv("datasets/dev_features.csv")
test_df  = pd.read_csv("datasets/test_features.csv")

text_col     = "text_clean"
label_col    = "label"
feature_cols = ["sentiment_textblob", "sentiment_vader", "gunning_fog", "lexical_diversity"]

# Handle missing text values to prevent TF-IDF ValueError
train_df[text_col] = train_df[text_col].fillna('')
dev_df[text_col]   = dev_df[text_col].fillna('')
test_df[text_col]  = test_df[text_col].fillna('')

# Also clean linguistic features just in case
train_df[feature_cols] = train_df[feature_cols].fillna(0)
dev_df[feature_cols]   = dev_df[feature_cols].fillna(0)
test_df[feature_cols]  = test_df[feature_cols].fillna(0)

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

y_dev_pred = best_rf.predict(X_dev)
dev_precision = precision_score(y_dev, y_dev_pred)

print("\n=== DEV SET RESULTS ===")
print(f"Dev Precision: {dev_precision:.4f}")
print("\nDev Classification Report:")
print(classification_report(y_dev, y_dev_pred, target_names=["Real", "Fake"]))

'''
kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== STABILITY CHECK: 5-FOLD CROSS-VALIDATION ===")
# 2. Run the cross-validation
# This tells us if the precision is consistent across different slices of data.
cv_precision_scores = cross_val_score(best_rf, X_train, y_train, cv=kf, scoring='precision')

# 3. Print the results
print(f"Precision per fold: {cv_precision_scores}")
print(f"Mean Precision: {cv_precision_scores.mean():.4f}")
print(f"Standard Deviation (+/-): {cv_precision_scores.std():.4f}")
'''