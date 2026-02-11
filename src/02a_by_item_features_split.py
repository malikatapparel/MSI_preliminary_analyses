
"""
Script: 02a_by_item_features_split.py
Project: Motivational Salience Index (MSI)
Authors: Marie Pittet & Malika Tapparel
Description: This script further prepares the data for ML analyses by:
    - computing Signal Detection Theory indices
    - pivoting the dataset wide
    - performing the train-test split at that stage (before dealing with missing data and normalizing) to avoid leaking 
"""

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
# 2) Computing Signal Detection Theory metrics (maybe more useful that raw hit/misses/etc)
# ------------------------------------------------------------
# counts of hits, miss, false alarms, correct rejections
nct, nca, nmt, nma = df["n_correct_towards"], df["n_correct_away"], df["n_miss_towards"], df["n_miss_away"]

# raw rates (will be NaN if denominator is 0)
df["correct_towards_rate"] = nct / (nct + nmt)
df["correct_away_rate"]     = nca / (nca + nma)
df["miss_towards_rate"]   = nmt / (nct + nmt)
df["miss_away_rate"]     = nma / (nca + nma)

# log-linear correction to prevents norm.ppf(0) or norm.ppf(1) from blowing up to infinity when hit/FA rates are exactly 0 or 1.
Hct = (nct  + 0.5) / (nct  + nmt  + 1.0)
Hca = (nca  + 0.5) / (nca  + nma  + 1.0)
Fct = (nmt  + 0.5) / (nct  + nmt  + 1.0)
Fca = (nma  + 0.5) / (nca  + nma  + 1.0)

# if a row has zero trials (denominator 0), set to NaN
Hct = Hct.where((nct + nmt) > 0)
Hca = Hca.where((nca + nma) > 0)
Fct = Fct.where((nct + nmt) > 0)
Fca = Fca.where((nca + nma) > 0)

# SDT metrics
df["dprime_towards"]    = norm.ppf(Hct) - norm.ppf(Fct)
df["dprime_away"]       = norm.ppf(Hca) - norm.ppf(Fca)
df["criterion_towards"] = -0.5 * (norm.ppf(Hct) + norm.ppf(Fct))
df["criterion_away"]    = -0.5 * (norm.ppf(Hca) + norm.ppf(Fca))

# ------------------------------------------------------------
# 4) train-test split (no pivot here since only 1 task)
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
