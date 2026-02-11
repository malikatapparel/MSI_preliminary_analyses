
"""
Script: 02a_by_item_features_split.py
Project: Motivational Salience Index (MSI)
Authors: Marie Pittet & Malika Tapparel
Description: This script further prepares the data for ML analyses by:
    - computing Signal Detection Theory indices
    - pivoting the dataset wide
    - performing the train-test split at that stage (before dealing with missing data and normalizing) to avoid leaking 
"""
# %%
# ------------------------------------------------------------
# 0) Env
# ------------------------------------------------------------
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import GroupShuffleSplit

# ------------------------------------------------------------
# 1) Loading the dataset
# ------------------------------------------------------------
df = pd.read_csv("../data/extracted/item_df.csv")

# ------------------------------------------------------------
# 2) train-test split (no pivot here since only 1 task)
# ------------------------------------------------------------
X = df.drop(columns=["vas_score"]) # features
y = df["vas_score"]
groups = df["sbj"]

# performing a 70%/30% train-test split
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=123)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx].reset_index(drop=True), X.iloc[test_idx].reset_index(drop=True)
y_train, y_test = y.iloc[train_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

# saving that
train_df = X_train.copy()
train_df["vas_score"] = y_train

test_df = X_test.copy()
test_df["vas_score"] = y_test

train_df.to_csv("../data/preprocessed/by_item/training.csv", index=False)
test_df.to_csv("../data/preprocessed/by_item/test.csv", index=False)

# %%
