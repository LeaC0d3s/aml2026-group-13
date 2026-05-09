import pandas as pd
from sklearn.model_selection import train_test_split


df_fake = pd.read_csv("aml2026-group-13/datasets/fake.csv")
df_true = pd.read_csv("aml2026-group-13/datasets/true.csv")
print(df_fake.head(3), f"Total rows: {len(df_fake)}")
print(df_true.head(3), f"Total rows: {len(df_true)}")
# Add labels
df_fake["label"] = 1
df_true["label"] = 0

# Combine datasets
df = pd.concat([df_fake, df_true], ignore_index=True)

# Shuffle dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# First split: 70% train, 30% temp
train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)

# Second split: 15% dev, 15% test
dev_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)

# Check proportions
print("Train size:", len(train_df))
print("Dev size:", len(dev_df))
print("Test size:", len(test_df))

print("\nClass distribution:")
print("Train:\n", train_df["label"].value_counts(normalize=True))
print("Dev:\n", dev_df["label"].value_counts(normalize=True))
print("Test:\n", test_df["label"].value_counts(normalize=True))

# Save splits
train_df.to_csv("aml2026-group-13/datasets/train.csv", index=False)
dev_df.to_csv("aml2026-group-13/datasets/dev.csv", index=False)
test_df.to_csv("aml2026-group-13/datasets/test.csv", index=False)

print("\nDatasets saved successfully.")
